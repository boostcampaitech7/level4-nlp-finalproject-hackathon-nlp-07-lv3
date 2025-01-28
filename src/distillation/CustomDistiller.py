import sys

from textbrewer import GeneralDistiller
from textbrewer.distiller_utils import *
from textbrewer.distiller_basic import BasicDistiller

import torch.distributed as dist
import torch.nn as nn
import wandb

sys.path.append('./src/Multi_Level_OT_main')
from Multi_Level_OT_main.models.distillation_model import DistillationLoss
from distillation.utils import (
                                dynamic_temperature,
                                minmax_normalize,
                                softmax_normalize,
                                standardize_tensor,
                                pad_logits,
                                read_teacher_outputs,
                                custom_post_adaptor,
                                CustomDict
                            )
from distillation.losses import dynamic_kd_loss, encoder_kd_loss, KL_divergence, KL_divergence_token_level
from utils import move_to_cuda

class CustomDistiller(GeneralDistiller):

    def __init__(self,
                 train_config,
                 distill_config,
                 model_T,
                 model_S,
                 adaptor_T,
                 adaptor_S,
                 logits_pro,
                 global_step_start,
                 use_softmax,
                 use_encoder_embeds=False,
                 dt_normalization_type : Optional[str] = "softmax",
                 intermediate_normalization_type : Optional[str] = "softmasx",
                 kd_type : Optional[str] = "kl_divergence_token_level",
                 intermediate_control_config='',
                 layer_weight=0.1,
                 padding_value=0
    ):

        super(CustomDistiller, self).__init__(train_config, distill_config, model_T, model_S, adaptor_T, adaptor_S)

        self.global_step_start = global_step_start
        self.use_softmax = use_softmax
        self.use_encoder_embeds = use_encoder_embeds
        assert kd_type in ['original_kd','dynamic_kd','dynamic_temperature', 'kl_divergence', 'kl_divergence_token_level'],"kd_type is not in ['original_kd','dynamic_kd','dynamic_temperature', 'kl_divergence', 'kl_divergence_token_level]"
        self.kd_type = kd_type
        self.normalization_type = dt_normalization_type
        assert dt_normalization_type in ['','minmax','softmax','standardize'],"normalization_type is not in ['','minmax','softmax','standardize']"
        self.intermediate_normalization_type =  intermediate_normalization_type
        assert intermediate_normalization_type in ['','minmax','softmax'],"intermediate_normalization_type is not in ['','minmax','softmax']"
        self.padding_value = padding_value
        self.dynamic_kd_loss = dynamic_kd_loss
        self.dynamic_temperature= dynamic_temperature
        self.KL_divergence = KL_divergence
        self.KL_divergence_token_level = KL_divergence_token_level
        if use_encoder_embeds:
            self.encoder_kd_loss = encoder_kd_loss
        else:
            self.encoder_kd_loss = None
        

        self.projs = []
        self.projs_group = []

        '''
        /textbrewer/presets/PROJ_MAP
        PROJ_MAP에 어떠한 projection이 가능한 다양한 레이어를 담아놓고 있음
        예를 들어 teacher 와 student 간의 Hidden Size가 차이가 나서 이를 차원에 맞게 수정이 필요하여 projection을 위해서 nn.Linear을 사용하면
        PROJ_MAP에 저장된 nn.Linear 방식을 사용해서 dim_in, dim_out을 활용해서 이를 선언
        PROJ_MAP은 외부에서 선언되어서 가져온 값으로 새로 선언해서 만들 혹은 수정의 필요가 있음 여기서는 nn.Linear로 고정하되 추후에 projection 방법론 다양화에 따라서 따로 수정해서 진행 필요
        '''
        # for im in self.d_config.intermediate_matches:
        #     if im.proj is not None:
        #         projection = im.proj[0]
        #         dim_in = im.proj[1] 
        #         dim_out = im.proj[2]
        #         self.projs_group.append(im.proj[3])
        #         self.projs.append(PROJ_MAP[projection](dim_in, dim_out))
        #         self.projs[-1].to(self.t_config.device)
        #     else:
        #         self.projs.append(None)
        #         self.projs_group.append(None)`

        self.logits_projs=[]
        if  logits_pro is not None:
                projection = logits_pro[0]
                dim_in = logits_pro[1]
                dim_out = logits_pro[2]
                self.logits_projs.append(PROJ_MAP[projection](dim_in, 4096))
                self.logits_projs.append(PROJ_MAP[projection](4096, dim_out))
        for layer in self.logits_projs:
                layer.to(self.t_config.device)

        self.d_config.is_caching_logits = False

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            handler = logging.FileHandler('distill.log')
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def train_on_batch(self, batch_S, batch_T, args=None, T_outputs_path=None):
        args = args or {}

        if self.model_T is not None:
            results_T, encoder_embeds_T, _ = self.model_T(batch_T)
            results_S, encoder_embeds_S, _ = self.model_S(batch_S)
            teacher_batch = batch_T
            student_batch = batch_S
        else:
            results_S = self.model_S(batch_S)
            results_T = CustomDict(read_teacher_outputs(T_outputs_path. self.t_config.device))
            teacher_batch = batch_S
            student_batch = batch_S

        
        '''
        /textbrewer/distiller_utils/post_adaptor

        self.adaptor_T 예시 :

        def simple_adaptor(batch, model_outputs):
            return {'logits': model_outputs.logits, 'hidden': model_outputs.hidden_states, 'losses': model_outputs.loss}

        post_adaptor 에서 해당 dict 값으로 반환값들을 파싱 후 후처리해서 반환 (각 value 를 list로 묶어서 반환)
        '''
        results_T = custom_post_adaptor(self.adaptor_T(results_T))
        results_S = custom_post_adaptor(self.adaptor_S(results_S))

        total_loss, losses_dict = self.compute_loss(results_S, results_T, encoder_embeds_S, encoder_embeds_T, teacher_batch, student_batch)

        return total_loss, losses_dict

    def compute_loss(self, results_S, results_T, encoder_embeds_S, encoder_embeds_T, teacher_batch, student_batch):
        # logit-based feature-based cross-entropy loss의 각각의 값들을 저장
        losses_dict = dict()

        total_loss = 0

        # Logit-Based
        if 'logits' in results_T and 'logits' in results_S:
            logits_list_T = results_T['logits']  # list of tensor
            logits_list_S = results_S['logits']  # list of tensor

            if self.kd_type == 'original_kd' and self.use_softmax:
                logits_list_T = F.softmax(logits_list_T[0], dim=-1)
                logits_list_S = F.softmax(logits_list_S[0], dim=-1)

            total_kd_loss = 0

            '''
            textbrewer/distiller_utils/select_logits_with_mask
            select_logits_with_mask 에서 attention_mask 을 활용해서 Logits 값 중에서 유효한 값들만 남김
            '''
            if 'logits_mask' in results_S:
                masks_list_S = results_S['logits_mask']
                logits_list_S = select_logits_with_mask(logits_list_S, masks_list_S)  # (mask_sum, num_of_class)
            if 'logits_mask' in results_T:
                masks_list_T = results_T['logits_mask']
                logits_list_T = select_logits_with_mask(logits_list_T, masks_list_T)  # (mask_sum, num_of_class)

            '''
            textbrewer/distiller_utils/probability_shift_
            tensor 내부에서 dim 별로 가장 높은 값을 찾아서 해당 값들을 정답이 되는 label 이 되는 값들과 바꾸어서
            teacher-forcing 느낌으로 l_T의 logits 중 정답에 해당 하는 값들이 가장 높게 고정 시킴
            '''
            if self.d_config.probability_shift is True:
                labels_list = results_S['labels']
                for l_T, l_S, labels in zip(logits_list_T, logits_list_S, labels_list):
                    l_T, l_S = l_T.to(self.t_config.device), l_S.to(self.t_config.device)
                    l_T = probability_shift_(l_T, labels)

                    # student -> teacher 의 차원으로 projection
                    for logits_layer in self.logits_projs:
                        l_S = logits_layer(l_S)

                    l_S, l_T = pad_logits(l_S, l_T, self.padding_value)
                    mask_S = (l_S != self.padding_value).any(dim=-1).float()
                    mask_T = (l_T != self.padding_value).any(dim=-1).float()
                    valid_mask = mask_S * mask_T

                    if self.d_config.temperature_scheduler is not None:
                        temperature = self.d_config.temperature_scheduler(l_S, l_T, self.d_config.temperature)
                    else:
                        temperature = self.d_config.temperature
                    if self.kd_type == 'original_kd':
                        total_kd_loss += self.kd_loss(l_S, l_T, temperature) # AbstractDistiller: self.kd_loss = KD_LOSS_MAP[self.d_config.kd_loss_type]
                    elif self.kd_type == 'dynamic_kd':
                        total_kd_loss += self.dynamic_kd_loss(l_S, l_T, temperature)
                    elif self.kd_type == 'dynamic_temperature':
                        total_kd_loss += self.dynamic_temperature(l_S, l_T,self.normalization_type)
                    elif self.kd_type == 'kl_divergence':
                        total_kd_loss += self.KL_divergence(l_S, l_T, mask_S, mask_T, padding_value=self.padding_value)
                    elif self.kd_type == 'kl_divergence_token_level':
                        total_kd_loss += self.KL_divergence_token_level(l_S, l_T, valid_mask, temperature) 

            else:
                for l_T, l_S in zip(logits_list_T, logits_list_S):
                    l_T, l_S = l_T.to(self.t_config.device), l_S.to(self.t_config.device)

                    for logits_layer in self.logits_projs:
                        l_S = logits_layer(l_S)

                    l_S, l_T = pad_logits(l_S, l_T, self.padding_value)
                    mask_S = (l_S != self.padding_value).any(dim=-1).float()
                    mask_T = (l_T != self.padding_value).any(dim=-1).float()
                    valid_mask = mask_S * mask_T

                    if self.d_config.temperature_scheduler is not None:
                        temperature = self.d_config.temperature_scheduler(l_S, l_T, self.d_config.temperature)
                    else:
                        temperature = self.d_config.temperature

                    if self.kd_type == 'original_kd':
                        total_kd_loss += self.kd_loss(l_S, l_T, temperature) # AbstractDistiller: self.kd_loss = KD_LOSS_MAP[self.d_config.kd_loss_type]
                    elif self.kd_type == 'dynamic_kd':
                        total_kd_loss += self.dynamic_kd_loss(l_S, l_T, temperature)
                    elif self.kd_type == 'dynamic_temperature':
                        total_kd_loss += self.dynamic_temperature(l_S, l_T,self.normalization_type)
                    elif self.kd_type == 'kl_divergence':
                        total_kd_loss += self.KL_divergence(l_S, l_T, mask_S, mask_T, padding_value=self.padding_value)
                    elif self.kd_type == 'kl_divergence_token_level':
                        total_kd_loss += self.KL_divergence_token_level(l_S, l_T, valid_mask) 


            losses_dict['logit_based_kd_loss'] = total_kd_loss

        # Encoder-embeds based 
        if self.encoder_kd_loss is not None:
            total_enkd_loss = 0

            encoder_embeds_S, encoder_embeds_T = encoder_embeds_S.to(self.t_config.device), encoder_embeds_T.to(self.t_config.device)
            total_enkd_loss = self.encoder_kd_loss(encoder_embeds_S, encoder_embeds_T)
            losses_dict['encoder_embeds_based_kd_loss'] = total_enkd_loss

        # Cross-Entropy Loss
        if 'losses' in results_S:
            total_hl_loss = 0
            for loss in results_S['losses']:
                # in case of multi-GPU
                total_hl_loss += loss.mean()
            losses_dict['cross_entropy_output_loss'] = total_hl_loss

        # Feature-Based
        # inters_T = {feature: results_T.get(feature, []) for feature in FEATURES}
        # inters_S = {feature: results_S.get(feature, []) for feature in FEATURES}
        # inputs_mask_T = results_T.get('inputs_mask', None)
        # inputs_mask_S = results_S.get('inputs_mask', None)
        # # pdb.set_trace()
        # for ith, inter_match in enumerate(self.d_config.intermediate_matches):
        #     layer_T = inter_match.layer_T
        #     layer_S = inter_match.layer_S
        #     feature = inter_match.feature
        #     loss_type = inter_match.loss
        #     match_weight = inter_match.weight
        #     match_loss = MATCH_LOSS_MAP[loss_type]


        #     if type(layer_S) is list and type(layer_T) is list:
        #         inter_S = [inters_S[feature][s] for s in layer_S]
        #         inter_T = [inters_T[feature][t] for t in layer_T]
        #         name_S = '-'.join(map(str, layer_S))
        #         name_T = '-'.join(map(str, layer_T))
        #         if self.projs[ith]:
        #             # inter_T = [self.projs[ith](t) for t in inter_T]
        #             # student -> teacher 의 차원으로 projection
        #             inter_S = [self.projs[ith](s) for s in inter_S]
        #     else:
        #         inter_S = inters_S[feature][layer_S]
        #         inter_T = inters_T[feature][layer_T]
        #         name_S = str(layer_S)
        #         name_T = str(layer_T)
        #         if self.projs[ith]:
        #             # inter_T = self.projs[ith](inter_T)
        #             # student -> teacher 의 차원으로 projection
        #             inter_S = inter_S.float()
        #             inter_S = self.projs[ith](inter_S)

        #     # normalize
        #     if len(self.intermediate_normalization_type)>0:
        #         if self.intermediate_normalization_type=='minmax':
        #             inter_S = minmax_normalize(inter_S)
        #             inter_T = minmax_normalize(inter_T)
        #         elif self.intermediate_normalization_type=='softmax':
        #             inter_S = softmax_normalize(inter_S)
        #             inter_T = softmax_normalize(inter_T)
        #         elif self.intermediate_normalization_type=='standardize':
        #             inter_S= standardize_tensor(inter_S)
        #             inter_T= standardize_tensor(inter_T)

        #     intermediate_loss = match_loss(inter_S, inter_T, mask=inputs_mask_S)
        #     total_loss += intermediate_loss * match_weight
        #     losses_dict[f'unweighted_{feature}_{loss_type}_{name_S}_{name_T}'] = intermediate_loss

        total_loss =  0.95 * losses_dict['cross_entropy_output_loss'] + 0.05 * losses_dict['logit_based_kd_loss']
        formated_losses_dict={}
        for key, value in losses_dict.items():
            if isinstance(value, torch.Tensor):
                # tensor -> float
                value = value.item()
            formated_losses_dict[key]=value
        self.logger.info(formated_losses_dict)
        wandb.log(formated_losses_dict)
        print(losses_dict)
        return total_loss, losses_dict


class CustomDistiller2:

    def __init__(self, adaptor_T, adaptor_S, use_encoder_output=True, batch_limit=100, store_path='teacher_logits_partial.npy', crossentropy_weight=1, distillation_weight=1, student_temperature=1, teacher_temperature=1, skip_student_eos=False, skip_teacher_eos=False, ignore_index=-100, debug=False, debug_rank=0, tokenizer_student=None, tokenizer_teacher=None, f=1):
        self.use_encoder_output = use_encoder_output
        self.adaptor_T = adaptor_T
        self.adaptor_S = adaptor_S

        '''
        reference:
            Multi-Level Optimal Transport for Universal Cross-Tokenizer Knowledge Distillation on Language Models
            https://arxiv.org/abs/2412.14528
            https://github.com/2018cx/Multi-Level-OT/tree/main
        '''
        self.multi_level_OT_distiller = DistillationLoss(
            batch_limit=batch_limit,
            store_path=store_path,
            crossentropy_weight=crossentropy_weight,
            distillation_weight=distillation_weight, 
            student_temperature=student_temperature, 
            teacher_temperature=teacher_temperature, 
            skip_student_eos=skip_student_eos, 
            skip_teacher_eos=skip_teacher_eos, 
            ignore_index=ignore_index, 
            debug=debug,
            debug_rank=debug_rank,
            tokenizer_student=tokenizer_student, 
            tokenizer_teacher=tokenizer_teacher, f=f
        )
        self.encoder_kd_loss = encoder_kd_loss

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            handler = logging.FileHandler('distill.log')
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def train_on_batch(self, epoch, output_T, output_S, target_T, target_S, encoder_output_T=None, encoder_output_S=None, device="cuda"):
        losses_dict={}
        output_T, output_S, target_T, target_S = CustomDict(move_to_cuda(self.adaptor_T(output_T), device)), CustomDict(move_to_cuda(self.adaptor_S(output_S), device)), move_to_cuda(target_T, device), move_to_cuda(target_S, device)
        loss, crossentropy_loss, distillation_loss = self.multi_level_OT_distiller(epoch, output_S, output_T, target_S, target_T)
        if self.use_encoder_output:
            encoder_output_S, encoder_output_T = encoder_output_S.last_hidden_state.to(device), encoder_output_T.last_hidden_state.to(device)
            encoder_loss = self.encoder_kd_loss(encoder_output_S, encoder_output_T)
            loss += encoder_loss

        losses_dict['crossentropy'] = crossentropy_loss
        losses_dict['multi_level_OT'] = distillation_loss
        losses_dict['encoder_embeds_based_kd_loss'] = encoder_loss
        self.logger.info(losses_dict)
        wandb.log(losses_dict)
        print(losses_dict)
        return loss

class MultiGranularFeatureAlignment(nn.Module):
    """
    - low-level: CNN 필터 출력의 평균 풀링 결과 비교
    - mid-level: self-attention 맵 비교
    - high-level: 최종 임베딩 혹은 로짓 KL 비교
    """
    def __init__(self, audio_dim=1024, text_dim=4096):
        super().__init__()
        # 계층 매핑 시 필요한 투영 레이어
        self.low_proj = nn.Linear(audio_dim, text_dim)
        self.mid_proj = nn.Linear(audio_dim, text_dim)
        self.high_proj = nn.Linear(audio_dim, text_dim)

    def forward(self, audio_feats, text_feats):
        """
        audio_feats: [B, T, A_dim]
        text_feats:  [B, T, T_dim]
        """

        # (1) 저수준 특성 정렬 (예: CNN 출력의 평균 풀링)
        audio_low = F.avg_pool1d(audio_feats.transpose(1,2), kernel_size=3).transpose(1,2)
        text_low  = F.avg_pool1d(text_feats.transpose(1,2), kernel_size=3).transpose(1,2)
        loss_low = F.mse_loss(self.low_proj(audio_low), text_low)

        # (2) 중간 단계: self-attention 맵 유사도
        #   예: audio_feats, text_feats 각각 Attention 연산 수행 후 결과 비교
        audio_attn = torch.softmax(audio_feats @ audio_feats.transpose(1,2), dim=-1)
        text_attn  = torch.softmax(text_feats @ text_feats.transpose(1,2), dim=-1)
        loss_mid   = F.kl_div(audio_attn.log(), text_attn, reduction='batchmean')

        # (3) 고수준: 로짓 정렬(KL divergence)
        audio_high = self.high_proj(audio_feats)  # [B, T, text_dim]
        loss_high  = F.kl_div(
            F.log_softmax(audio_high, dim=-1),
            F.softmax(text_feats, dim=-1),
            reduction='batchmean'
        )

        # 가중 합산(임의로 0.3:0.4:0.3으로 설정)
        return 0.3 * loss_low + 0.4 * loss_mid + 0.3 * loss_high
    
class DifficultyAwareDistiller:
    """
    - 교사 모델 로짓에서 계산된 엔트로피 혹은 손실을 바탕으로
      샘플별로 distillation loss 가중치를 자동 조정.
    """
    def __init__(self, base_alpha=0.2, max_alpha=0.4, ema=0.9):
        self.base_alpha = base_alpha
        self.max_alpha = max_alpha
        self.ema = ema
        self.prev_entropy = None

    def calc_distillation_loss(self, teacher_logits, student_logits, gold_labels=None):
        """
        teacher_logits: [B, T, C]
        student_logits: [B, T, C]
        """
        # (1) 교사 모델의 엔트로피 계산
        teacher_probs = torch.softmax(teacher_logits, dim=-1)
        teacher_ent   = -torch.sum(teacher_probs * torch.log(teacher_probs + 1e-9), dim=-1)
        batch_ent     = teacher_ent.mean()

        # (2) EMA(지수 가중 이동평균)로 난이도 추적
        if self.prev_entropy is None:
            self.prev_entropy = batch_ent
        else:
            self.prev_entropy = self.ema * self.prev_entropy + (1 - self.ema) * batch_ent

        # (3) 난이도에 따른 alpha 조정
        #     예: 난이도가 높을수록 alpha가 커진다
        dynamic_alpha = self.base_alpha + torch.sigmoid(self.prev_entropy - 2.0) * (self.max_alpha - self.base_alpha)
        
        # (4) distillation loss: KL(teacher || student)
        distill_loss = F.kl_div(
            F.log_softmax(student_logits, dim=-1),
            torch.softmax(teacher_logits, dim=-1),
            reduction='batchmean'
        )
        
        # (5) CE 손실(ground truth와 student 예측 비교)
        ce_loss = None
        if gold_labels is not None:
            ce_loss = F.cross_entropy(student_logits.transpose(1,2), gold_labels)
        else:
            # gold_labels가 없는 경우, distil loss만
            ce_loss = 0

        # (6) 최종 손실 결합
        total_loss = (1 - dynamic_alpha) * ce_loss + dynamic_alpha * distill_loss
        return total_loss, distill_loss, ce_loss, dynamic_alpha.item()
    
class StreamingAdapter:
    """
    주어진 chunk_size와 lookahead로 오디오 스트리밍을 처리
    AAC 실시간 태스크에 활용 가능
    """
    def __init__(self, model, teacher_model=None, chunk_size=16000, lookahead=4000):
        self.model = model
        self.teacher_model = teacher_model  # 없을 경우 None
        self.buffer = torch.zeros(chunk_size + lookahead)
        self.chunk_size = chunk_size
        self.lookahead = lookahead

    def stream_step(self, new_audio):
        """
        new_audio: [N] mono waveform
        """
        # 버퍼 업데이트
        self.buffer = torch.cat([self.buffer[self.chunk_size:], new_audio])
        chunk = self.buffer[:self.chunk_size]
        look = self.buffer[-self.lookahead:]

        # 모델 추론(학생)
        student_out = self.model.forward_chunk(chunk, look)

        distill_loss = 0
        if self.teacher_model is not None:
            with torch.no_grad():
                teacher_out = self.teacher_model.forward_chunk(chunk, look)
            distill_loss = F.kl_div(
                F.log_softmax(student_out, dim=-1),
                torch.softmax(teacher_out, dim=-1),
                reduction='batchmean'
            )
        return student_out, distill_loss
    
class StreamingAdapter:
    """
    주어진 chunk_size와 lookahead로 오디오 스트리밍을 처리
    AAC 실시간 태스크에 활용 가능
    """
    def __init__(self, model, teacher_model=None, chunk_size=16000, lookahead=4000):
        self.model = model
        self.teacher_model = teacher_model  # 없을 경우 None
        self.buffer = torch.zeros(chunk_size + lookahead)
        self.chunk_size = chunk_size
        self.lookahead = lookahead

    def stream_step(self, new_audio):
        """
        new_audio: [N] mono waveform
        """
        # 버퍼 업데이트
        self.buffer = torch.cat([self.buffer[self.chunk_size:], new_audio])
        chunk = self.buffer[:self.chunk_size]
        look = self.buffer[-self.lookahead:]

        # 모델 추론(학생)
        student_out = self.model.forward_chunk(chunk, look)

        distill_loss = 0
        if self.teacher_model is not None:
            with torch.no_grad():
                teacher_out = self.teacher_model.forward_chunk(chunk, look)
            distill_loss = F.kl_div(
                F.log_softmax(student_out, dim=-1),
                torch.softmax(teacher_out, dim=-1),
                reduction='batchmean'
            )
        return student_out, distill_loss