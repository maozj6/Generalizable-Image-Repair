import torch
import torch.nn as nn
import time

# 编码器：CNN -> mu, logvar
class CNNEncoder(nn.Module):
    def __init__(self, latent_dim=32):
        super(CNNEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(24, 32, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU()
        )
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(64 * 13 * 8, latent_dim)
        self.fc_logvar = nn.Linear(64 * 13 * 8, latent_dim)

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

# 解码器：latent -> reconstructed image
class CNNDecoder(nn.Module):
    def __init__(self, latent_dim=32):
        super(CNNDecoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 64 * 13 * 8),
            nn.ReLU()
        )
        self.deconv = nn.Sequential(
            nn.Unflatten(1, (64, 13, 8)),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 24, kernel_size=5, stride=2, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(24, 3, kernel_size=5, stride=2, output_padding=1),
            nn.Sigmoid()  # 输出范围0-1
        )

    def forward(self, z):
        x = self.fc(z)
        return self.deconv(x)

# VAE 模型：Encoder + Sampling + Decoder
class VAEController(nn.Module):
    def __init__(self, latent_dim=32):
        super(VAEController, self).__init__()
        self.encoder = CNNEncoder(latent_dim)
        self.decoder = CNNDecoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 初始化模型
latent_dim = 32
model = VAEController(latent_dim=latent_dim).to(device)
model.eval()

# 创建随机输入
input_sample = torch.randn(1, 3, 160, 120).to(device)

# GPU 预热
with torch.no_grad():
    for _ in range(10):
        _ = model(input_sample)

# 正式计时：1000 次 encode+decode
start_time = time.time()
with torch.no_grad():
    for _ in range(1000):
        _ = model(input_sample)
end_time = time.time()

# 输出统计
total_time = end_time - start_time
avg_time = total_time / 1000
print(f"\nTotal VAE inference time (1000 runs): {total_time:.4f} s")
print(f"Average per sample (encode + decode): {avg_time * 1000:.4f} ms")

