import torch
from torch import nn

class VQVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_embeddings):
        super(VQVAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.codebook = nn.Embedding(num_embeddings, hidden_dim)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, _ = self.vector_quantization(z_e.detach())
        x_recon = self.decoder(z_q)

        return x_recon, z_e, z_q

    def vector_quantization(self, z):
        z_e = z.unsqueeze(-1)
        distances = torch.sum((z_e - self.codebook.weight)**2, dim=2)
        b = torch.argmin(distances, dim=1)
        z_q = self.codebook(b).detach()

        return z_q, b