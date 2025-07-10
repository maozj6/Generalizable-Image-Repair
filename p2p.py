import torch
import torch.nn as nn
import time


class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super().__init__()
        self.outermost = outermost
        use_bias = (norm_layer == nn.InstanceNorm2d)
        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                              stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            out = self.model(x)
            # Center crop input x to match out
            x_cropped = x
            if x.size(2) > out.size(2) or x.size(3) > out.size(3):
                h_diff = x.size(2) - out.size(2)
                w_diff = x.size(3) - out.size(3)
                x_cropped = x[:, :, h_diff // 2: x.size(2) - (h_diff - h_diff // 2),
                                 w_diff // 2: x.size(3) - (w_diff - w_diff // 2)]
            return torch.cat([x_cropped, out], dim=1)


class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super().__init__()

        # Build unet generator structure recursively
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, innermost=True, norm_layer=norm_layer)
        for _ in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

    def forward(self, input):
        return self.model(input)


if __name__ == '__main__':
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create the generator
    input_nc = 3
    output_nc = 3
    image_size = (128, 160)  # height, width
    netG = UnetGenerator(input_nc, output_nc, num_downs=7).to(device)

    # Dummy input
    dummy_input = torch.randn(1, input_nc, *image_size).to(device)

    # Warm up
    _ = netG(dummy_input)

    # Measure inference time
    with torch.no_grad():
        start = time.time()
        for _ in range(100):
            _ = netG(dummy_input)
        end = time.time()

    print(f"Average inference time per image: {(end - start)/100:.6f} seconds")


