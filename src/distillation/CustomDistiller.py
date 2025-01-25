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
            encoder_output_S, encoder_output_T = encoder_output_S.to(device), encoder_output_T.to(device)
            encoder_loss = self.encoder_kd_loss(encoder_output_S, encoder_output_T)
            loss += encoder_loss

        losses_dict['crossentropy'] = crossentropy_loss
        losses_dict['multi_level_OT'] = distillation_loss
        losses_dict['encoder_embeds_based_kd_loss'] = encoder_loss
        self.logger.info(losses_dict)
        wandb.log(losses_dict)
        print(losses_dict)
        return loss
