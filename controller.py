import torch
import torch.nn as nn
import time

# 定义 CNN 控制器模型
class CNNController(nn.Module):
    def __init__(self, num_outputs=2):
        super(CNNController, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(24, 32, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 13 * 8, 100), nn.ReLU(),
            nn.Linear(100, 50), nn.ReLU(),
            nn.Linear(50, 10), nn.ReLU(),
            nn.Linear(10, num_outputs)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 实例化模型并移动到 GPU
model = CNNController().to(device)
model.eval()

# 创建一个随机输入样本 (1, 3, 160, 120)
input_sample = torch.randn(1, 3, 160, 120).to(device)

# GPU warm-up（避免首次慢热）
with torch.no_grad():
    for _ in range(10):
        _ = model(input_sample)

# 正式计时推理 1000 次
start_time = time.time()
with torch.no_grad():
    for _ in range(1000):
        _ = model(input_sample)
end_time = time.time()

# 输出运行时间
total_time = end_time - start_time
avg_time = total_time / 1000
print(f"\nTotal time: {total_time:.4f} s")
print(f"Average time per inference: {avg_time * 1000:.4f} ms")


