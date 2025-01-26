import torch
import torch.nn as nn

# Mock encoder embeddings for testing
def generate_mock_data(batch_size=4, embedding_dim=128, seq_length=10):
    """
    Generates mock data for teacher and student embeddings.
    """
    teacher_embeds = torch.randn(batch_size, seq_length, embedding_dim, requires_grad=True)
    student_embeds = torch.randn(batch_size, seq_length, embedding_dim, requires_grad=True)
    return teacher_embeds, student_embeds

# Cosine similarity function
def cosine_similarity(q_vec, c_vec):
    """
    Computes cosine similarity between two matrices.
    """
    q_vec = q_vec / q_vec.norm(dim=1, keepdim=True)
    c_vec = c_vec / c_vec.norm(dim=1, keepdim=True)
    return torch.mm(q_vec, c_vec.T)


# Contrastive loss function
def contrastive_loss(encoder_embeds_S, encoder_embeds_T, scaling_temperature=1):
    """
    Computes contrastive loss as per the given formula.
    """
    # Compute similarity matrix
    sim_matrix = cosine_similarity(encoder_embeds_T, encoder_embeds_S)

    # Positive similarity (diagonal elements)
    pos_sim = torch.diag(sim_matrix)

    # Compute exp(similarity/temperature) for all elements
    exp_sim_matrix = torch.exp(sim_matrix / scaling_temperature)

    # Row-wise sum for teacher -> student
    row_sum = exp_sim_matrix.sum(dim=1)
    L_i1 = -torch.log(torch.exp(pos_sim / scaling_temperature) / row_sum)

    # Transposed similarity for student -> teacher
    sim_matrix_transposed = sim_matrix.T
    pos_sim_transposed = torch.diag(sim_matrix_transposed)
    row_sum_transposed = exp_sim_matrix.T.sum(dim=1)
    L_i2 = -torch.log(torch.exp(pos_sim_transposed / scaling_temperature) / row_sum_transposed)

    # Combine and compute mean
    L_contra = torch.mean(L_i1 + L_i2)


    return L_contra

# Main encoder knowledge distillation loss function
def encoder_kd_loss(encoder_embeds_S, encoder_embeds_T, scaling_temperature=1, student_device='cuda'):
    """
    Encoder KD Loss function as described in the reference.
    """
    # Dimension adjustment for projection
    emd_s_size, emd_t_size = encoder_embeds_S.size(-1), encoder_embeds_T.size(-1)
    dim_in = min(emd_s_size, emd_t_size)
    dim_out = max(emd_s_size, emd_t_size)
    projection_layer = nn.Linear(dim_in, dim_out).to(student_device)

    # Mean pooling over sequence dimension
    encoder_embeds_S = torch.mean(encoder_embeds_S, dim=-2)
    encoder_embeds_T = torch.mean(encoder_embeds_T, dim=-2)

    # Apply projection if dimensions mismatch
    if emd_s_size > emd_t_size:
        encoder_embeds_T = projection_layer(encoder_embeds_T)
    elif emd_s_size < emd_t_size:
        encoder_embeds_S = projection_layer(encoder_embeds_S)

    # Compute contrastive loss
    loss = contrastive_loss(encoder_embeds_S, encoder_embeds_T, scaling_temperature)
    loss2 = contrastive_loss2(encoder_embeds_S, encoder_embeds_T, scaling_temperature)
    return loss, loss2

   

# Testing both implementations
if __name__ == "__main__":
    # Generate mock data
    teacher_embeds, student_embeds = generate_mock_data(batch_size=4, embedding_dim=128, seq_length=10)

    # Set scaling temperature
    scaling_temperature = 0.1

    # Compute loss
    loss, loss2 = encoder_kd_loss(student_embeds, teacher_embeds, scaling_temperature)
    print(f"Computed Encoder KD Loss: {loss.item()}")
    print(f"Computed Encoder KD Loss: {loss2.item()}")
