import sys
import wandb

from textbrewer import GeneralDistiller
from textbrewer.distiller_utils import *

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
from distillation.losses import dynamic_kd_loss, encoder_kd_loss, KL_divergence, KL_divergence_token_level, QFormerDistiller
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
                 use_encoder_embeds=True,
                 dt_normalization_type : Optional[str] = "softmax",
                 kd_type : Optional[str] = "kl_divergence_token_level",
                 layer_weight=0.1,
                 padding_value=0,
                 base_alpha=0.2, max_alpha=0.4, ema=0.9
    ):

        super(CustomDistiller, self).__init__(train_config, distill_config, model_T, model_S, adaptor_T, adaptor_S)

        self.global_step_start = global_step_start
        self.use_softmax = use_softmax
        self.use_encoder_embeds = use_encoder_embeds
        assert kd_type in ['original_kd','dynamic_kd','dynamic_temperature', 'kl_divergence', 'kl_divergence_token_level'],"kd_type is not in ['original_kd','dynamic_kd','dynamic_temperature', 'kl_divergence', 'kl_divergence_token_level]"
        self.kd_type = kd_type
        self.normalization_type = dt_normalization_type
        assert dt_normalization_type in ['','minmax','softmax','standardize'],"normalization_type is not in ['','minmax','softmax','standardize']"
        self.padding_value = padding_value
        self.dynamic_kd_loss = dynamic_kd_loss
        self.dynamic_temperature= dynamic_temperature
        self.KL_divergence = KL_divergence
        self.KL_divergence_token_level = KL_divergence_token_level
        if use_encoder_embeds:
            self.encoder_kd_loss = encoder_kd_loss
        else:
            self.encoder_kd_loss = None
        self.base_alpha = base_alpha
        self.max_alpha = max_alpha
        self.ema = ema
        self.prev_entropy = None
        

        self.projs = []
        self.projs_group = []
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
            results_T, encoder_embeds_T, _, _, _ = self.model_T(batch_T)
            results_S, encoder_embeds_S, _, _, _ = self.model_S(batch_S)
            teacher_batch = batch_T
            student_batch = batch_S
        else:
            results_S, _, _, _, _ = self.model_S(batch_S)
            results_T = CustomDict(read_teacher_outputs(T_outputs_path, self.t_config.device))
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

            # (1) 교사 모델의 엔트로피 계산
            teacher_probs = torch.softmax(logits_list_T[0], dim=-1)
            teacher_ent   = -torch.sum(teacher_probs * torch.log(teacher_probs + 1e-9), dim=-1)
            batch_ent     = teacher_ent.mean()
            batch_ent_detached = batch_ent.detach()

            # (2) EMA(지수 가중 이동평균)로 난이도 추적
            if self.prev_entropy is None:
                self.prev_entropy = batch_ent_detached
            else:
                self.prev_entropy = self.ema * self.prev_entropy + (1 - self.ema) * batch_ent_detached

            # (3) 난이도에 따른 alpha 조정
            #     예: 난이도가 높을수록 alpha가 커진다
            dynamic_alpha = self.base_alpha + torch.sigmoid(self.prev_entropy - 2.0) * (self.max_alpha - self.base_alpha)
            dynamic_alpha = dynamic_alpha.to(self.t_config.device).item()

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

            encoder_embeds_S, encoder_embeds_T = encoder_embeds_S.last_hidden_state.to(self.t_config.device), encoder_embeds_T.last_hidden_state.to(self.t_config.device)
            total_enkd_loss = self.encoder_kd_loss(encoder_embeds_S, encoder_embeds_T)
            losses_dict['encoder_embeds_based_kd_loss'] = total_enkd_loss

        # Cross-Entropy Loss
        if 'loss' in results_S:
            total_hl_loss = 0
            for loss in results_S['loss']:
                total_hl_loss += loss.mean()
            losses_dict['cross_entropy_output_loss'] = total_hl_loss

        # total_loss =  0.95 * losses_dict['cross_entropy_output_loss'] + 0.05 * losses_dict['logit_based_kd_loss']
        total_loss = (1 - dynamic_alpha) * losses_dict['cross_entropy_output_loss'] + dynamic_alpha * losses_dict['logit_based_kd_loss'] + losses_dict['encoder_embeds_based_kd_loss']
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
    
class CustomDistiller3:

    def __init__(self, adaptor_T, adaptor_S, qformer_dim_T, qformer_dim_S, device='cuda'):
        self.adaptor_T = adaptor_T
        self.adaptor_S = adaptor_S
        self.qformer_dim_T = qformer_dim_T
        self.qformer_dim_S = qformer_dim_S
        self.device = device


        self.qformer_distiller = QFormerDistiller(
                teacher_dim=self.qformer_dim_T, student_dim=self.qformer_dim_S, student_device=self.device
        )
        self.encoder_kd_loss = encoder_kd_loss
        

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            handler = logging.FileHandler('distill.log')
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def train_on_batch(self, epoch, output_T, output_S, qformer_output_T, qformer_output_S, whisper_output_T, whisper_output_S, beats_output_T, beats_output_S, device="cuda"):
        losses_dict={}
        output_T, output_S = custom_post_adaptor(self.adaptor_T(output_T)), custom_post_adaptor(self.adaptor_S(output_S))
        qformer_output_S, qformer_output_T = qformer_output_S.last_hidden_state.to(device), qformer_output_T.last_hidden_state.to(device)
        whisper_output_S, whisper_output_T = whisper_output_S.last_hidden_state.to(device), whisper_output_T.last_hidden_state.to(device)
        beats_output_S, beats_output_T = beats_output_S.last_hidden_state.to(device), beats_output_T.last_hidden_state.to(device)

        total_loss = 0

        if 'logits' in output_T and 'logits' in output_S:
            logits_list_T = output_T['logits']  # list of tensor
            logits_list_S = output_S['logits']  # list of tensor

            output_logits_loss = 0

            # (1) 교사 모델의 엔트로피 계산
            teacher_probs = torch.softmax(logits_list_T[0], dim=-1)
            teacher_ent   = -torch.sum(teacher_probs * torch.log(teacher_probs + 1e-9), dim=-1)
            batch_ent     = teacher_ent.mean()
            batch_ent_detached = batch_ent.detach()

            # (2) EMA(지수 가중 이동평균)로 난이도 추적
            if self.prev_entropy is None:
                self.prev_entropy = batch_ent_detached
            else:
                self.prev_entropy = self.ema * self.prev_entropy + (1 - self.ema) * batch_ent_detached

            # (3) 난이도에 따른 alpha 조정
            #     예: 난이도가 높을수록 alpha가 커진다
            dynamic_alpha = self.base_alpha + torch.sigmoid(self.prev_entropy - 2.0) * (self.max_alpha - self.base_alpha)
            dynamic_alpha = dynamic_alpha.to(device).item()

            for l_T, l_S in zip(logits_list_T, logits_list_S):
                l_T, l_S = l_T.to(device), l_S.to(device)

                for logits_layer in self.logits_projs:
                    l_S = logits_layer(l_S)

                l_S, l_T = pad_logits(l_S, l_T, self.padding_value)
                mask_S = (l_S != self.padding_value).any(dim=-1).float()
                mask_T = (l_T != self.padding_value).any(dim=-1).float()
                valid_mask = mask_S * mask_T
                output_logits_loss += self.KL_divergence_token_level(l_S, l_T, valid_mask)

        qformer_loss = self.qformer_distiller(qformer_output_T, qformer_output_S)
        whisper_loss = self.encoder_kd_loss(whisper_output_S, whisper_output_T, student_device=self.device, use_contrasive_loss=False)
        beats_loss = self.encoder_kd_loss(beats_output_S, beats_output_T, student_device=self.device, use_contrasive_loss=False)

        if 'loss' in output_S:
            ce_loss = 0
            for loss in output_S['loss']:
                ce_loss += loss.mean()

        losses_dict['cross_entropy'] = ce_loss
        losses_dict['qformer'] = qformer_loss
        losses_dict['whisper_loss'] = whisper_loss
        losses_dict['beats_loss'] = beats_loss
        losses_dict['output_logits_kl'] = output_logits_loss

        total_loss = (1 - dynamic_alpha) * losses_dict['cross_entropy'] + dynamic_alpha * losses_dict['output_logits_kl'] + 0.8 * losses_dict['qformer'] + 0.8 * losses_dict['whisper_loss'] + 0.8 * losses_dict['beats_loss'] 
        self.logger.info(losses_dict)
        wandb.log(losses_dict)
        print(losses_dict)
        return total_loss

         
    
