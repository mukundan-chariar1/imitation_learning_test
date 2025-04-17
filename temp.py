import torch
import torch.nn.functional as F
import torch.linalg as linalg

# X = torch.tensor([[0.2, 0.8],
#                  [0.5, 0.5],
#                  [1.0, 0.5]], dtype=torch.float32)

# Wq = torch.tensor([[0.5, 0.5],
#                   [0, 1]], dtype=torch.float32)
# Wk = torch.tensor([[1, 0],
#                   [-0.5, 0.5]], dtype=torch.float32)
# Wv = torch.tensor([[1, 0],
#                   [0, 1]], dtype=torch.float32)

# Q=X@Wq
# K=X@Wk
# V=X@Wv

# attn_scores=Q@K.T

# mask = torch.triu(torch.ones_like(attn_scores, dtype=torch.bool), diagonal=1)
# attn_scores_masked = attn_scores.masked_fill(mask, float('-inf'))

# A = F.softmax(attn_scores_masked, dim=-1)

# O=A@V

# print("Output O:\n", O)

# mu_real = torch.tensor([0.5, 0.2, 0.7, 0.4], dtype=torch.float32)
# Sigma_real = torch.tensor([
#     [0.2, 0.1, 0.05, 0.02],
#     [0.1, 0.4, 0.1, 0.05],
#     [0.05, 0.1, 0.4, 0.1],
#     [0.02, 0.05, 0.1, 0.2]
# ], dtype=torch.float32)

# mu1 = torch.tensor([0.4, 0.3, 0.5, 0.6], dtype=torch.float32)
# Sigma1 = torch.tensor([
#     [0.7, 0.1, 0.05, 0.02],
#     [0.1, 0.2, 0.1, 0.05],
#     [0.05, 0.1, 0.3, 0.1],
#     [0.02, 0.05, 0.1, 0.15]
# ], dtype=torch.float32)

# mu2 = torch.tensor([0.7, 0.1, 0.3, 0.5], dtype=torch.float32)
# Sigma2 = torch.tensor([
#     [0.4, 0.05, 0.05, 0.01],
#     [0.05, 0.3, 0.1, 0.05],
#     [0.05, 0.1, 0.2, 0.05],
#     [0.01, 0.05, 0.05, 0.2]
# ], dtype=torch.float32)

# def calculate_fid(mu_real, Sigma_real, mu_fake, Sigma_fake):

#     import pdb; pdb.set_trace()


#     diff = mu_real - mu_fake
#     sqdiff = torch.sum(diff ** 2)
    
#     covmean = torch.sqrt(Sigma_real @ Sigma_fake)
    
#     tr_term = torch.trace(Sigma_real + Sigma_fake - 2 * covmean)
    
#     fid = sqdiff + tr_term
#     return fid.item()

# fid1 = calculate_fid(mu_real, Sigma_real, mu1, Sigma1)
# fid2 = calculate_fid(mu_real, Sigma_real, mu2, Sigma2)

# print(f"FID for Set I: {fid1:.4f}")
# print(f"FID for Set II: {fid2:.4f}")

# if fid1 < fid2:
#     print("Set I is better (lower FID score)")
# else:
#     print("Set II is better (lower FID score)")

x0 = torch.tensor([[0.8, 0.5, 0.9],
                   [0.1, 1.0, 0.6],
                   [0.4, 0.6, 0.3]], dtype=torch.float32)

epsilon1 = torch.tensor([[1.0, -1.0, 1.0],
                         [-1.0, 1.0, 1.0],
                         [1.0, -1.0, 1.0]], dtype=torch.float32)

epsilon2 = torch.tensor([[2.0, -2.0, 2.0],
                         [-2.0, 2.0, 2.0],
                         [2.0, -2.0, 2.0]], dtype=torch.float32)

# Linear beta schedule
beta0 = 0.0001
beta1000 = 0.02

def linear_beta(t):
    """Linear schedule for beta_t"""
    return beta0 + (t / 1000) * (beta1000 - beta0)

# Compute beta1 and beta2
beta1 = linear_beta(1)  # t=1
beta2 = linear_beta(2)  # t=2

def forward_diffusion_step(xt_1, beta_t, epsilon_t):
    """Perform one forward diffusion step using reparameterization"""

    import pdb; pdb.set_trace()

    
    sqrt_one_minus_beta = torch.sqrt(1 - torch.tensor(beta_t))
    sqrt_beta = torch.sqrt(torch.tensor(beta_t))
    xt = sqrt_one_minus_beta * xt_1 + sqrt_beta * epsilon_t
    return xt

# Step 1: Compute x1 from x0
x1 = forward_diffusion_step(x0, beta1, epsilon1)

# Step 2: Compute x2 from x1
x2 = forward_diffusion_step(x1, beta2, epsilon2)

print("x1 (after first diffusion step):")
print(x1.numpy())
print("\nx2 (after second diffusion step):")
print(x2.numpy())