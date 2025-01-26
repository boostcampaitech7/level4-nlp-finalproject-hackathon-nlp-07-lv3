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
    kl_per_token = p_T * (log_p_T - log_p_S)  # ( L, V)
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

