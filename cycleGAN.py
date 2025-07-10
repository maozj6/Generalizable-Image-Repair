import torch
import time
import functools
import torch.nn as nn

# ---------- 1. 工具函数与组件 ----------

class Identity(nn.Module):
    def forward(self, x):
        return x

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        return functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        return functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        return lambda x: Identity()
    else:
        raise NotImplementedError(f'Normalization layer [{norm_type}] not found')


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super().__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        block = []
        p = 1 if padding_type == 'zero' else 0
        pad = {
            'reflect': nn.ReflectionPad2d(1),
            'replicate': nn.ReplicationPad2d(1),
            'zero': None
        }[padding_type]
        if pad:
            block += [pad]
        block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            block += [nn.Dropout(0.5)]
        if pad:
            block += [pad]
        block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]
        return nn.Sequential(*block)

    def forward(self, x):
        return x + self.conv_block(x)

# ---------- 2. CycleGAN 生成器 ----------

class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, n_blocks=9, padding_type='reflect'):
        assert(n_blocks >= 0)
        super().__init__()
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        # Downsampling
        for i in range(2):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        # ResNet blocks
        mult = 2 ** 2
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type, norm_layer, use_dropout, use_bias)]

        # Upsampling
        for i in range(2):
            mult = 2 ** (2 - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

# ---------- 3. 参数类 ----------

class Opt:
    input_nc = 3
    output_nc = 3
    ngf = 64
    netG = 'resnet_9blocks'
    norm = 'instance'
    no_dropout = True
    init_type = 'normal'
    init_gain = 0.02

# ---------- 4. 构建并测速 G_A ----------

if __name__ == "__main__":
    opt = Opt()
    assert torch.cuda.is_available(), "CUDA is not available!"
    device = torch.device('cuda')

    norm_layer = get_norm_layer(opt.norm)
    netG_A = ResnetGenerator(opt.input_nc, opt.output_nc, opt.ngf, norm_layer,
                             not opt.no_dropout, n_blocks=9).to(device)
    netG_A.eval()

    dummy_input = torch.randn(1, 3, 120, 160).to(device)

    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            _ = netG_A(dummy_input)

    # Timed run
    n_runs = 1000
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = netG_A(dummy_input)
    torch.cuda.synchronize()
    end = time.time()

    avg_time_ms = (end - start) / n_runs * 1000
    print(f"Average inference time over {n_runs} runs: {avg_time_ms:.3f} ms")


