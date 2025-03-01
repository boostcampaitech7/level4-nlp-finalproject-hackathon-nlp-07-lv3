# Copyright (2024) Tsinghua University, Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import json
import logging
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteriaList,
)

from . import modeling_ced  # noqa
from .beats.BEATs import BEATs, BEATsConfig
from .modeling_ced import *  # noqa
from .modeling_whisper import WhisperModel
from .Qformer import BertConfig, BertLMHeadModel
from .utils import StoppingCriteriaSub


class SALMONN(nn.Module):
    @classmethod
    def init_speech_Qformer(cls, num_query_token, speech_width, num_hidden_layers=2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased", trust_remote_code=True)
        encoder_config.num_hidden_layers = num_hidden_layers
        encoder_config.encoder_width = speech_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 1
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(torch.zeros(1, num_query_token, encoder_config.hidden_size))
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    @property
    def device(self):
        return list(self.parameters())[0].device

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.amp.autocast(device_type="cuda", dtype=dtype, enabled=True)
        else:
            return contextlib.nullcontext()

    def __init__(
        self,
        llama_path="",
        whisper_path="",
        freeze_whisper=True,
        ced_path="",
        beats_path="",
        freeze_beats=True,
        use_speech_Qformer=True,
        num_speech_query_token=1,
        freeze_speech_QFormer=False,
        window_level_Qformer=True,
        second_per_window=0.333333,
        second_stride=0.333333,
        speech_llama_proj_model="",
        freeze_speech_llama_proj=False,
        lora=True,
        lora_rank=8,
        lora_alpha=32,
        lora_dropout=0.1,
        multi_prompt=False,
        prompt_path="",
        prompt_template="",
        max_txt_len=128,
        end_sym="</s>",
        low_resource=False,  # use 8 bit
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
        token=None,
        only_preprocessor=None,
    ):
        super().__init__()

        self.ced_path = ced_path
        self.beats_path = beats_path
        self.use_speech_Qformer = use_speech_Qformer
        self.window_level_Qformer = window_level_Qformer
        self.second_per_window = second_per_window
        self.second_stride = second_stride
        self.lora = lora
        self.multi_prompt = multi_prompt
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym
        self.low_resource = low_resource

        logging.info("Loading LLaMA Tokenizer")
        self.llama_tokenizer = AutoTokenizer.from_pretrained(llama_path,
                                                             use_fast=False,
                                                             token=token,
                                                             trust_remote_code = True,
                                                             padding_side="right",
                                                             bos_token="<|im_start|>",
                                                             eos_token="<|im_end|>")

        self.llama_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.llama_tokenizer.padding_side = "right"

        if not only_preprocessor:
            logging.info("Loading LLaMA Model")
            if self.low_resource:
                self.llama_model = AutoModelForCausalLM.from_pretrained(
                    llama_path,
                    torch_dtype=torch.float16,
                    load_in_8bit=True,
                    device_map={"": device_8bit},
                    token=token,
                    attn_implementation="sdpa",
                )
            else:
                self.llama_model = AutoModelForCausalLM.from_pretrained(
                    llama_path,
                    torch_dtype=torch.float16,
                    token=token,
                    # attn_implementation="sdpa", # flash attention
                )

            # 모델 토큰나이저의 사전에 새로운 토큰을 추가하면서 임베딩 레이어 크기가 변화할 필요가 있으니 그에 따라 사이즈를 조정하는 코드
            self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
            logging.info("Loading LLaMA Done")

            if self.lora:
                self.peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    target_modules = ["q_proj", "v_proj", "gate_proj"]
                )
                self.llama_model = get_peft_model(self.llama_model, self.peft_config)
                self.llama_model.print_trainable_parameters()
                logging.info("LoRA Training")

        assert whisper_path
        logging.info("Loading Whisper Model")
        # speech model
        self.speech_encoder = WhisperModel.from_pretrained(whisper_path).encoder
        # speech
        self.ln_speech = nn.LayerNorm(self.speech_encoder.config.d_model)
        if freeze_whisper:
            for name, param in self.speech_encoder.named_parameters():
                param.requires_grad = False
            self.speech_encoder.eval()
            logging.info("freeze Whisper")

        if self.ced_path:
            logging.info("Loading CED Model")
            self.ced = getattr(modeling_ced, self.ced_path)(pretrained=True)
            self.ln_audio = nn.LayerNorm(self.ced.embed_dim)
            if freeze_beats:
                for name, param in self.ced.named_parameters():
                    param.requires_grad = False
                self.ced.eval()
                logging.info("freeze CED")

        elif self.beats_path:
            logging.info("Loading BEATs Model")
            beats_ckpt = torch.load(self.beats_path, map_location="cpu", weights_only=True)
            beats_cfg = BEATsConfig(beats_ckpt["cfg"])

            self.beats = BEATs(beats_cfg)
            self.beats.load_state_dict(beats_ckpt["model"])
            self.ln_audio = nn.LayerNorm(self.beats.cfg.encoder_embed_dim)
            if freeze_beats:
                for name, param in self.beats.named_parameters():
                    param.requires_grad = False
                self.beats.eval()
                logging.info("freeze BEATs")

        if self.use_speech_Qformer:
            if self.ced_path:
                self.speech_Qformer, self.speech_query_tokens = self.init_speech_Qformer(
                    num_query_token=num_speech_query_token,
                    speech_width=self.speech_encoder.config.d_model + self.ced.embed_dim
                )
            elif self.beats_path:
                self.speech_Qformer, self.speech_query_tokens = self.init_speech_Qformer(
                    num_query_token=num_speech_query_token,
                    speech_width=self.speech_encoder.config.d_model + self.beats.cfg.encoder_embed_dim,
                )
            else:
                self.speech_Qformer, self.speech_query_tokens = self.init_speech_Qformer(
                    num_query_token=num_speech_query_token, speech_width=self.speech_encoder.config.d_model
                )
            self.speech_Qformer.bert.embeddings.word_embeddings = None
            self.speech_Qformer.bert.embeddings.position_embeddings = None
            for layer in self.speech_Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
            self.speech_Qformer.cls = None
            # QFormer은 학습 때 사용되기 때문에 학습 시에는 freeze하지 않음
            if freeze_speech_QFormer:
                for name, param in self.speech_Qformer.named_parameters():
                    param.requires_grad = False
                self.speech_Qformer.eval()
                self.speech_query_tokens.requires_grad = False
                logging.info("freeze Speech QFormer")

            # 해당 코드를 논문에서 소개된 두 인코더에서 나온 결과값을 concat 한 z값을 QFormer 통해서 alignment한 H 값을 LLM(LLaMA)에 넣기 위해서 project 하는 코드
            logging.info("Loading speech LLAMA proj")
            if only_preprocessor:
                config = AutoConfig.from_pretrained(llama_path, token=token)
                lm_hidden_size = config.hidden_size
            else:
                lm_hidden_size = self.llama_model.config.hidden_size
            self.speech_llama_proj = nn.Linear(self.speech_Qformer.config.hidden_size, lm_hidden_size)
            # self.speech_llama_proj 가중치를 사전에 정의된 값으로 초기화
            if speech_llama_proj_model:
                logging.info("Loading speech LLAMA proj from {}".format(speech_llama_proj_model))
                speech_llama_proj_weight = torch.load(speech_llama_proj_model, map_location="cpu")
                self.load_state_dict(speech_llama_proj_weight["model"], strict=False)
            # self.speeech_llama_proj 가중치 freeze
            if freeze_speech_llama_proj:
                for name, param in self.speech_llama_proj.named_parameters():
                    param.requires_grad = False
                self.speech_llama_proj.eval()
                logging.info("freeze speech LLAMA proj")
        else:
            # feel free to add other aligners here
            raise NotImplementedError

        # prepare prompts
        self.prompt_dict = {}
        if prompt_path:
            try:
                with open(prompt_path, "r", encoding="utf-8") as file:
                    raw_prompts = json.load(file)
            except json.JSONDecodeError as e:
                print(f"JSON decoding error with utf-8 encoding: {e}")
                raise
            except IOError as e:
                print(f"Failed to open or read the file: {e}")
                raise
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                raise

            for task, prompts in raw_prompts.items():
                filtered_prompts = [prompt for prompt in prompts if "<SpeechHere>" in prompt]
                self.prompt_dict[task] = [prompt_template.format(p) for p in filtered_prompts]

            print("Loading training prompts done!")

    def _encode_auditory_feature(self, speech_embeds, audio_embeds=None):
        with self.maybe_autocast():
            if self.use_speech_Qformer:
                speech_embeds = self.ln_speech(speech_embeds)
                if audio_embeds is not None:
                    audio_embeds = self.ln_audio(audio_embeds)
                    # 두 임베딩 값의 크기를 맞춰서 padding
                    if audio_embeds.size(1) < speech_embeds.size(1):
                        audio_embeds = F.pad(audio_embeds, (0, 0, 0, speech_embeds.size(1) - audio_embeds.size(1)))
                    elif audio_embeds.size(1) > speech_embeds.size(1):
                        speech_embeds = F.pad(speech_embeds, (0, 0, 0, audio_embeds.size(1) - speech_embeds.size(1)))
                    # speech encoder + non-speech encoder 결과를 concat 해서 Z 값
                    speech_embeds = torch.cat((speech_embeds, audio_embeds), dim=-1)
                speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long).to(speech_embeds.device)

                # 인코더 결과값들은 고정된 값이 아니라 valiable 한 값이기에 QFormer는 태생적으로 고정된 값들 만을 처리할 수 있다.
                # 이를 위해서 인코더 결과값들을 슬라이딩 윈도우 방식으로 나누어서 처리하여 가변적인 값들도 QFormer을 이용해서 처리할 수 있게 적용
                if self.window_level_Qformer:
                    B, T, C = speech_embeds.shape
                    kernel = round(1500 * self.second_per_window / 30.0)
                    stride = round(1500 * self.second_stride / 30.0)
                    kernel = (1, kernel)
                    stride = (1, stride)
                    speech_embeds_tr = speech_embeds.transpose(1, 2).unsqueeze(2)
                    # 입력 텐서에 슬라이딩 윈도우를 적용해서 커널이 지나간 요소들을 Flatten 해서 하나의 벡터로 변환 후 최종적으로 2D tensor로 반환
                    speech_embeds_overlap = F.unfold(
                        speech_embeds_tr, kernel_size=kernel, dilation=1, padding=0, stride=stride
                    )
                    _, _, L = speech_embeds_overlap.shape
                    speech_embeds_overlap = speech_embeds_overlap.view(B, -1, kernel[1], L)
                    speech_embeds_overlap = torch.permute(speech_embeds_overlap, [0, 3, 2, 1])
                    speech_embeds = speech_embeds_overlap.reshape(-1, kernel[1], C)
                    speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long, device=speech_embeds.device)

                query_tokens = self.speech_query_tokens.expand(speech_embeds.shape[0], -1, -1)
                query_output = self.speech_Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=speech_embeds,
                    encoder_attention_mask=speech_atts,
                    return_dict=True,
                )
                # QFormer 결과값을 LLM 에 들어가기 적합하게 projection
                speech_embeds = self.speech_llama_proj(query_output.last_hidden_state)

                if self.window_level_Qformer:
                    speech_embeds = speech_embeds.view(B, -1, speech_embeds.size(2)).contiguous()

                speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long).to(speech_embeds.device)
            else:
                raise NotImplementedError

        return speech_embeds, speech_atts

    def encode_speech(self, spectrogram, raw_wav=None, audio_padding_mask=None):
        with self.maybe_autocast():
            # speech
            speech_embeds = self.speech_encoder(spectrogram, return_dict=True).last_hidden_state
            # non-speech
            if self.ced_path and raw_wav is not None:
                audio_embeds = self.ced.forward(raw_wav)
            elif self.beats_path and raw_wav is not None:
                audio_embeds, _ = self.beats.extract_features(
                    raw_wav, padding_mask=audio_padding_mask, feature_only=True
                )
            else:
                audio_embeds = None

        return self._encode_auditory_feature(speech_embeds, audio_embeds=audio_embeds)

    def prompt_wrap(self, embeds, atts, prompt, multi_prompt=False):
        if prompt:
            if multi_prompt:
                p_before = []
                p_after = []
                for i, p in enumerate(prompt):
                    b, a = p.split("<SpeechHere>")
                    p_before.append(b)
                    p_after.append(a)

                # input_ids, attention_mask
                p_before_tokens = self.llama_tokenizer(p_before, return_tensors="pt", add_special_tokens=False).to(
                    embeds.device
                )
                p_before_embeds = (
                    self.llama_model.model.embed_tokens(p_before_tokens.input_ids)
                    if not self.lora
                    else self.llama_model.model.model.embed_tokens(p_before_tokens.input_ids)
                )

                # speech_embeds wrapped with prompts_embeds are padded to the same length here
                p_after_tokens = self.llama_tokenizer(
                    p_after, return_tensors="pt", padding="longest", add_special_tokens=False
                ).to(embeds.device)
                p_after_embeds = (
                    self.llama_model.model.embed_tokens(p_after_tokens.input_ids)
                    if not self.lora
                    else self.llama_model.model.model.embed_tokens(p_after_tokens.input_ids)
                )

                wrapped_embeds = torch.cat([p_before_embeds, embeds, p_after_embeds], dim=1)
                wrapped_atts = torch.cat([p_before_tokens.attention_mask, atts, p_after_tokens.attention_mask], dim=1)
            else:
                batch_size = embeds.shape[0]
                p_before, p_after = prompt.split("<SpeechHere>")

                p_before_tokens = self.llama_tokenizer(p_before, return_tensors="pt", add_special_tokens=False).to(
                    embeds.device
                )
                p_after_tokens = self.llama_tokenizer(p_after, return_tensors="pt", add_special_tokens=False).to(
                    embeds.device
                )
                p_before_embeds = (
                    self.llama_model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
                    if not self.lora
                    else self.llama_model.model.model.embed_tokens(p_before_tokens.input_ids).expand(
                        batch_size, -1, -1
                    )
                )
                p_after_embeds = (
                    self.llama_model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
                    if not self.lora
                    else self.llama_model.model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
                )

                wrapped_embeds = torch.cat([p_before_embeds, embeds, p_after_embeds], dim=1)
                wrapped_atts = torch.cat([p_before_tokens.attention_mask, atts, p_after_tokens.attention_mask], dim=1)
            return wrapped_embeds, wrapped_atts
        else:
            return embeds, atts

    def forward(self, samples, verbose=False):
        # detect whether there are multi tasks in this batch
        task = list(set(samples["task"]))
        if len(task) > 1 or "QA" in task:
            self.multi_prompt = True

        # prepare prompts
        if self.prompt_dict:
            if self.multi_prompt:
                prompt = [random.choice(self.prompt_dict[task]) for task in samples["task"]]
                # Q = Qusestion
                if "Q" in samples:
                    prompt = [p.format(q) if "{}" in p else p for p, q in zip(prompt, samples["Q"])]
            else:
                prompt = random.choice(self.prompt_dict[samples["task"][0]])

        # use speech/audio encoder to encode speech/audio
        spectrogram = samples["spectrogram"]
        raw_wav = samples.get("raw_wav", None)
        audio_padding_mask = samples.get("padding_mask", None)

        # speech encoder + non-speech encoder 의 결과물을 합쳐서 QFormer를 통과 후에 LLM에 들어가기 위해서 proj 까지 완료된 결과물
        # LLM에 input으로 들어가기 위한 값들
        speech_embeds, speech_atts = self.encode_speech(
            spectrogram, raw_wav=raw_wav, audio_padding_mask=audio_padding_mask
        )

        # wrap speech_embeds with prompts
        # LLM instruction을 위한 prompt와 결합
        if self.prompt_dict:
            speech_embeds, speech_atts = self.prompt_wrap(
                speech_embeds, speech_atts, prompt, multi_prompt=self.multi_prompt
            )

        # prepare inputs for LLM
        # to_regress_tokens은 LLM이 예측할 출력 결과값
        # self.end_sym은 </s> 으로 EOS 토큰역할을 함
        text = [t + self.end_sym for t in samples["text"]]
        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False,
        ).to(spectrogram.device)
        to_regress_embeds = (
            self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
            if not self.lora
            else self.llama_model.model.model.embed_tokens(to_regress_tokens.input_ids)
        )

        # -100 은 어텐션에서 무시해야하는 토큰 지정하여 loss 계산 시에도 무시
        # 길이를 맞추기 위해서 적용된 Padding token은 Cross-Entropy Loss 계산 시에는 필요가 없으니 이는 마스킹 처리
        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )

        # 마찬가지로 오디오 인코더에서 들어온 값들은 LLM이 출력하는 값이 아니기에 여기도 Loss 계산 시에 무시하기 위해서
        # 오디오 인코더 길이 만큼의 -100 으로 마스킹 처리
        empty_targets = (
            torch.ones([speech_atts.shape[0], speech_atts.shape[1] + 1], dtype=torch.long)
            .to(spectrogram.device)
            .fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        batch_size = speech_embeds.shape[0]
        # bos(begin of sentence) 토큰들을 배치 사이즈 만큼 생성하여 배치 내의 모든 요소들을 위한 bos 토큰 생성
        bos = (
            torch.ones(
                [batch_size, 1],
                dtype=to_regress_tokens.input_ids.dtype,
                device=to_regress_tokens.input_ids.device,
            )
            * self.llama_tokenizer.bos_token_id
        )
        bos_embeds = (
            self.llama_model.model.embed_tokens(bos)
            if not self.lora
            else self.llama_model.model.model.embed_tokens(bos)
        )
        atts_bos = speech_atts[:, :1]

        inputs_embeds = torch.cat([bos_embeds, speech_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, speech_atts, to_regress_tokens.attention_mask], dim=1)

        # calulate loss
        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
            loss = outputs.loss

        if verbose:
            nvocab = self.llama_model.config.vocab_size
            results = outputs.logits[:, empty_targets.size(1) - 1 : -1, :].contiguous().view(-1, nvocab).argmax(dim=-1)
            labels = targets[:, empty_targets.size(1) :].contiguous().view(-1)
            mask = labels != -100
            correct = (results[mask] == labels[mask]).float().sum()
            total = len(labels[mask])

        if verbose:
            return {"loss": loss, "correct": correct, "total": total}

        return {"loss": loss}

    def generate(self, samples, generate_cfg, prompts=None):
        batch_size = samples["spectrogram"].shape[0]

        spectrogram = samples["spectrogram"]
        raw_wav = samples.get("raw_wav", None)
        audio_padding_mask = samples.get("padding_mask", None)

        # speech encoder + non-speech encoder 의 결과물을 합쳐서 QFormer를 통과 후에 LLM에 들어가기 위해서 proj 까지 완료된 결과물
        # LLM에 input으로 들어가기 위한 값들
        speech_embeds, speech_atts = self.encode_speech(
            spectrogram, raw_wav=raw_wav, audio_padding_mask=audio_padding_mask
        )

        # LLM instruction을 위한 prompt와 결합
        if prompts is not None:
            speech_embeds, speech_atts = self.prompt_wrap(speech_embeds, speech_atts, prompts, multi_prompt=True)

        # bos(begin of sentence) 토큰들을 배치 사이즈 만큼 생성하여 배치 내의 모든 요소들을 위한 bos 토큰 생성
        bos = (
            torch.ones(
                [batch_size, 1],
                dtype=torch.int32,
                device=speech_embeds.device,
            )
            * self.llama_tokenizer.bos_token_id
        )
        bos_embeds = (
            self.llama_model.model.embed_tokens(bos)
            if not self.lora
            else self.llama_model.model.model.embed_tokens(bos)
        )
        atts_bos = speech_atts[:, :1]

        embeds = torch.cat([bos_embeds, speech_embeds], dim=1)
        attns = torch.cat([atts_bos, speech_atts], dim=1)

        # 해당 토큰이 생성되면 생성을 종료한다
        stop_words_ids = [torch.tensor([2]).to(speech_embeds.device)]  # TODO: fix this heuristics
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

        outputs = self.llama_model.generate(
            inputs_embeds=embeds,
            max_new_tokens=generate_cfg.get("max_new_tokens", 200),
            stopping_criteria=stopping_criteria,
            num_beams=generate_cfg.get("num_beams", 4),
            do_sample=generate_cfg.get("do_sample", False),
            min_length=generate_cfg.get("min_length", 1),
            temperature=generate_cfg.get("temperature", 1.0),
            top_p=generate_cfg.get("top_p", 0.9),
            repetition_penalty=generate_cfg.get("repetition_penalty", 1.0),
            length_penalty=generate_cfg.get("length_penalty", 1.0),
            attention_mask=attns,
        )
        # 토큰값들을 다시 원본 텍스트로 decoding
        text = self.llama_tokenizer.batch_decode(outputs, add_special_tokens=False)

        return text

    @classmethod
    def from_config(cls, config):
        from dotenv import load_dotenv
        load_dotenv()
        token = os.environ['HF_KEY']

        llama_path = config.get("llama_path")
        whisper_path = config.get("whisper_path")
        freeze_whisper = config.get("freeze_whisper", True)
        beats_path = config.get("beats_path", "")
        ced_path = config.get("ced_path", "")
        freeze_beats = config.get("freeze_beats", True)

        use_speech_Qformer = config.get("use_speech_Qformer", True)
        num_speech_query_token = config.get("num_speech_query_token", 1)
        freeze_speech_QFormer = config.get("freeze_speech_QFormer", False)
        window_level_Qformer = config.get("window_level_Qformer", True)
        second_per_window = config.get("second_per_window", 0.333333)
        second_stride = config.get("second_stride", 0.333333)

        speech_llama_proj_model = config.get("speech_llama_proj_model", "")
        freeze_speech_llama_proj = config.get("freeze_speech_llama_proj", False)

        lora = config.get("lora", True)
        lora_rank = config.get("lora_rank", 8)
        lora_alpha = config.get("lora_alpha", 32)
        lora_dropout = config.get("lora_dropout", 0.1)

        multi_prompt = config.get("multi_prompt", False)
        prompt_path = config.get("prompt_path", "")
        prompt_template = config.get("prompt_template", "")
        max_txt_len = config.get("max_txt_len", 128)
        end_sym = config.get("end_sym", "</s>")
        low_resource = config.get("low_resource", False)
        device_8bit = config.get("device_8bit", 0)

        only_preprocessor = config.get("only_preprocessor", None)

        model = cls(
            llama_path=llama_path,
            whisper_path=whisper_path,
            freeze_whisper=freeze_whisper,
            beats_path=beats_path,
            ced_path = ced_path,
            freeze_beats=freeze_beats,
            use_speech_Qformer=use_speech_Qformer,
            num_speech_query_token=num_speech_query_token,
            freeze_speech_QFormer=freeze_speech_QFormer,
            window_level_Qformer=window_level_Qformer,
            second_per_window=second_per_window,
            second_stride=second_stride,
            speech_llama_proj_model=speech_llama_proj_model,
            freeze_speech_llama_proj=freeze_speech_llama_proj,
            lora=lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            multi_prompt=multi_prompt,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
            token=token,
            only_preprocessor=only_preprocessor,
        )

        ckpt_path = config.get("ckpt", "")
        if ckpt_path:
            logging.info("Load SALMONN ckpt from: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt["model"], strict=False)

        return model
