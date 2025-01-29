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
    def __init__(self, audio_dim=1024, text_dim=4096, device='cuda'):
        super().__init__()
        self.low_proj = nn.Linear(audio_dim, text_dim).to(device)
        self.mid_proj = nn.Linear(audio_dim, text_dim).to(device)
        self.high_proj = nn.Linear(audio_dim, text_dim).to(device)

    def safe_avg_pool(self, x, kernel_size):
        seq_len = x.size(-1)
        if seq_len < kernel_size:
            pad_total = kernel_size - seq_len
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            x = F.pad(x, (pad_left, pad_right), "constant", 0)
        return F.avg_pool1d(x, kernel_size=kernel_size)

    def forward(self, audio_feats, text_feats):
        # 배치 크기 강제 일치
        assert audio_feats.size(0) == text_feats.size(0), \
            f"Batch size mismatch: audio {audio_feats.size(0)} vs text {text_feats.size(0)}"

        # (1) 저수준 특성 정렬
        B = audio_feats.size(0)  # 배치 크기 고정
        
        # 오디오 처리
        audio_t = audio_feats.transpose(1, 2)  # [B, A_dim, T_audio]
        kernel_audio = min(3, audio_t.size(-1))
        audio_low = self.safe_avg_pool(audio_t, kernel_audio)
        audio_low = audio_low.transpose(1, 2)  # [B, T_audio_pool, A_dim]
        
        # 텍스트 처리
        text_t = text_feats.transpose(1, 2)  # [B, T_text, D_text]
        kernel_text = min(3, text_t.size(-1))
        text_low = self.safe_avg_pool(text_t, kernel_text)
        text_low = text_low.transpose(1, 2)  # [B, T_text_pool, D_text]

        # 시퀀스 길이 동기화
        min_seq = min(audio_low.size(1), text_low.size(1))
        loss_low = F.mse_loss(
            self.low_proj(audio_low[:, :min_seq, :]), 
            text_low[:, :min_seq, :]
        )

        # (2) 중간 단계: Attention 맵 유사도
        audio_attn = torch.softmax(audio_feats @ audio_feats.transpose(1,2), dim=-1)
        text_attn = torch.softmax(text_feats @ text_feats.transpose(1,2), dim=-1)
        loss_mid = F.kl_div(audio_attn.log(), text_attn, reduction='batchmean')

        # (3) 고수준: 로짓 정렬
        audio_high = self.high_proj(audio_feats)
        loss_high = F.kl_div(
            F.log_softmax(audio_high, dim=-1),
            F.softmax(text_feats, dim=-1),
            reduction='batchmean'
        )

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