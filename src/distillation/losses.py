from typing import Optional, Any, Union

import torch
import torch.nn as nn
from torch.nn import functional as F


def dynamic_kd_loss(student_logits, teacher_logits, temperature=1.0):
    with torch.no_grad():
        student_probs = F.softmax(student_logits, dim=-1)
        student_entropy = -torch.sum(student_probs * torch.log(student_probs + 1e-6), dim=-1) # student entropy, (bsz, )
        # normalized entropy score by student uncertainty:
        # i.e.,  entropy / entropy_upper_bound
        # higher uncertainty indicates the student is more confusing about this instance
        instance_weight = student_entropy / torch.log(torch.ones_like(student_entropy) * student_logits.size(1))
    student_input = F.log_softmax(student_logits / temperature, dim=-1)
    target = F.softmax(teacher_logits / temperature, dim=-1)
    batch_loss = F.kl_div(student_input, target, reduction="none").sum(-1) * temperature ** 2  # bsz
    weighted_kld = torch.mean(batch_loss * instance_weight)
    return weighted_kld

def KL_divergence_token_level(logits_S, logits_T, valid_mask, temperature=3.0):
    """
    logits_S, logits_T : (L, V)
    valid_mask         : (L), 유효 토큰은 1, 패딩 토큰은 0
    temperature        : distillation temperature
    """
    p_T     = F.softmax(logits_T / temperature, dim=-1)       # (L, V)
    log_p_T = F.log_softmax(logits_T / temperature, dim=-1)   # (L, V)
    log_p_S = F.log_softmax(logits_S / temperature, dim=-1)   # (L, V)

    # KL = Σ p_T * (log_p_T - log_p_S)
    kl_per_token = p_T * (log_p_T - log_p_S)  # (L, V)
    kl_per_token = kl_per_token.sum(dim=-1)   # (L)

    # 유효 위치만 골라서(마스크 곱) 합산
    kl_per_token = kl_per_token * valid_mask  # 패딩(0)은 0이 됨

    # 유효 토큰 개수로 평균
    kl_loss = kl_per_token.sum() / (valid_mask.sum() + 1e-9)

    # KD 논문에서 T^2를 곱해주는 관행이 있음
    return kl_loss * (temperature ** 2)

def KL_divergence(logits_S, logits_T, mask_S, mask_T, scaling_temperatures=1, padding_value=-100):
    # masking 된 값에 대해서는 softmax 시에 값에 포함이 거의 되지 않게 -inf 로 초기화
    masked_logits_S = logits_S.masked_fill(mask_S == 0, float('-inf'))
    masked_logits_T = logits_T.masked_fill(mask_T == 0, float('-inf'))

    teacher_probs = F.softmax(masked_logits_T / scaling_temperatures, dim=-1)
    student_log_probs = F.log_softmax(masked_logits_S / scaling_temperatures, dim=-1)

    loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (scaling_temperatures ** 2)
    return loss

def encoder_kd_loss(encoder_embeds_S, encoder_embeds_T, scaling_temerature=1, student_device='cuda'):
    '''
    reference:
        Efficient Audio Captioning with Encoder-Level Knowledge Distillation 
        https://arxiv.org/abs/2407.14329
    '''
    emd_s_size, emd_t_size = encoder_embeds_S.size(-1), encoder_embeds_T.size(-1)
    dim_in = min(emd_s_size, emd_t_size)
    dim_out = max(emd_s_size, emd_t_size)
    projection_layer = nn.Linear(dim_in, dim_out).to(student_device)

    encoder_embeds_S = torch.mean(encoder_embeds_S, dim=-2)
    encoder_embeds_T = torch.mean(encoder_embeds_T, dim=-2)

    if emd_s_size > emd_t_size:
        encoder_embeds_T = projection_layer(encoder_embeds_T)
    elif emd_s_size < emd_t_size:
        encoder_embeds_S = projection_layer(encoder_embeds_S)

    if encoder_embeds_S.size(0) == 1:
        loss = kd_mse_loss(encoder_embeds_S, encoder_embeds_T, scaling_temerature)
    else:
        loss = contrastive_loss(encoder_embeds_S, encoder_embeds_T, scaling_temerature)

    return loss

def cosine_similarity(q_vec, c_vec):
    q_vec = q_vec / q_vec.norm(dim=1, keepdim=True)
    c_vec = c_vec / c_vec.norm(dim=1, keepdim=True)
    return torch.mm(q_vec, c_vec.T)

def contrastive_loss(encoder_embeds_S, encoder_embeds_T, scaling_temperature=1):
    sim_matrix = cosine_similarity(encoder_embeds_T, encoder_embeds_S)  
    pos_sim = torch.diag(sim_matrix)  #
    exp_sim_matrix = torch.exp(sim_matrix / scaling_temperature) 
    row_sum = exp_sim_matrix.sum(dim=1)
    L_i1 = -torch.log(torch.exp(pos_sim / scaling_temperature) / row_sum)

    sim_matrix_transposed = sim_matrix.T 
    pos_sim_transposed = torch.diag(sim_matrix_transposed)  
    row_sum_transposed = exp_sim_matrix.T.sum(dim=1) 
    L_i2 = -torch.log(torch.exp(pos_sim_transposed / scaling_temperature) / row_sum_transposed)
    L_contra = torch.mean(L_i1 + L_i2)

    return L_contra

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

    def calc_distillation_loss(self, teacher_output, student_output):
        teacher_logits = teacher_output.logits
        student_logits = student_output.logits
        
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
        ce_loss = student_output.loss

        # (6) 최종 손실 결합
        total_loss = (1 - dynamic_alpha) * ce_loss + dynamic_alpha * distill_loss
        return total_loss, distill_loss, ce_loss, dynamic_alpha.item()