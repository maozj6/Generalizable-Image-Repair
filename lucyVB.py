import torch
import torch.nn.functional as F
import time

# Lucy-Richardson去卷积（单通道版本，3通道循环处理）
def lucy_richardson_torch(blurred, psf, iterations=10):
    eps = 1e-7
    psf_mirror = torch.flip(psf, [2, 3])
    estimate = torch.full_like(blurred, 0.5)

    for _ in range(iterations):
        conv_est = F.conv2d(estimate, psf, padding='same')
        relative_blur = blurred / (conv_est + eps)
        estimate *= F.conv2d(relative_blur, psf_mirror, padding='same')

    return estimate

# TV 去噪（近似变分贝叶斯）
def tv_denoise_torch(image, weight=0.1, iterations=20):
    img = image.clone()

    for _ in range(iterations):
        grad_x = img[..., :, 1:] - img[..., :, :-1]
        grad_y = img[..., 1:, :] - img[..., :-1, :]

        # 对 grad 做 padding 以保持尺寸一致
        grad_x = F.pad(grad_x, (0, 1), mode='replicate')  # pad right
        grad_y = F.pad(grad_y, (0, 0, 0, 1), mode='replicate')  # pad bottom

        # 计算散度（divergence）
        div_x = grad_x[..., :, :-1] - grad_x[..., :, 1:]
        div_y = grad_y[..., :-1, :] - grad_y[..., 1:, :]
        div_x = F.pad(div_x, (0, 1), mode='replicate')  # pad right
        div_y = F.pad(div_y, (0, 0, 0, 1), mode='replicate')  # pad bottom

        div = div_x + div_y
        img += weight * div

    return img

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_trials = 1000
    image_shape = (3, 160, 120)
    inputs = torch.rand(num_trials, *image_shape, device=device)

    # 模糊核：3通道分开使用
    psf = torch.ones((3, 1, 5, 5), device=device) / 25.0

    # ========== Lucy-Richardson ==========
    start = time.time()
    for i in range(num_trials):
        result = []
        for c in range(3):
            img = inputs[i, c:c+1].unsqueeze(0)  # shape: (1,1,H,W)
            res = lucy_richardson_torch(img, psf[c:c+1], iterations=10)
            result.append(res.squeeze(0))
        _ = torch.cat(result, dim=0)
    end = time.time()
    lucy_avg_time = (end - start) / num_trials
    print(f"Lucy-Richardson avg time per image: {lucy_avg_time * 1000:.4f} ms")

    # ========== Variational Bayes (TV Denoising) ==========
    start = time.time()
    for i in range(num_trials):
        result = []
        for c in range(3):
            img = inputs[i, c]
            res = tv_denoise_torch(img.unsqueeze(0), weight=0.1, iterations=20)
            result.append(res)
        _ = torch.cat(result, dim=0)
    end = time.time()
    tv_avg_time = (end - start) / num_trials
    print(f"TV Denoising (Variational) avg time per image: {tv_avg_time * 1000:.4f} ms")

if __name__ == "__main__":
    main()


