### adapted from https://raw.githubusercontent.com/sherwinbahmani/cc3d/master/training/networks_3d.py

import torch
import numpy as np

@torch.jit.script
def clamp_gain(x: torch.Tensor, g: float, c: float):
    return torch.clamp(x * g, -c, c)


def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


def identity(x):
    return x


def leaky_relu_0_2(x):
    return torch.nn.functional.leaky_relu(x, 0.2)


activation_funcs = {
    "linear": {
        "fn": identity,
        "def_gain": 1
    },
    "lrelu": {
        "fn": leaky_relu_0_2,
        "def_gain": np.sqrt(2)
    }
}


class FullyConnectedLayer(torch.nn.Module):

    def __init__(self, in_features, out_features, bias=True, activation='linear', lr_multiplier=1, bias_init=0):
        super().__init__()
        self.activation = activation_funcs[activation]['fn']
        self.activation_gain = activation_funcs[activation]['def_gain']
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight * self.weight_gain
        b = self.bias
        if b is not None and self.bias_gain != 1:
            b = b * self.bias_gain
        x = self.activation(torch.addmm(b.unsqueeze(0), x, w.t())) * self.activation_gain
        return x


class SmoothUpsample(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
        return x

def modulated_conv3d(x, weight, styles, padding=0, demodulate=True):
    batch_size = x.shape[0]
    out_channels, in_channels, kd, kh, kw = weight.shape

    # Calculate per-sample weights and demodulation coefficients.
    w = weight.unsqueeze(0)  # [NOIkkk]
    w = w * styles.reshape(batch_size, 1, -1, 1, 1, 1)  # [NOIkkk]

    if demodulate:
        dcoefs = (w.square().sum(dim=[2, 3, 4, 5]) + 1e-8).rsqrt()  # [NO]
        w = w * dcoefs.reshape(batch_size, -1, 1, 1, 1, 1)  # [NOIkk]

    # Execute as one fused op using grouped convolution.
    batch_size = int(batch_size)
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kd, kh, kw)
    x = torch.nn.functional.conv3d(x, w, padding=padding, groups=batch_size)
    x = x.reshape(batch_size, -1, *x.shape[2:])
    return x


class Generator(torch.nn.Module):

    def __init__(self, z_dim, w_dim, num_layers, img_resolution, img_channels, c_dim = 0, channel_multiplier = 1, bias_resolution = 64, **unused_kwargs):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = SynthesisNetwork3D(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, channel_multiplier=channel_multiplier, bias_resolution=bias_resolution)
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim = c_dim, w_dim=w_dim, num_ws=self.num_ws, num_layers=num_layers)

    def forward(self, z, c = None, truncation_psi=1, truncation_cutoff=None, noise_mode='random', report_stats = False):

        if report_stats:
            return_dict = {
                "res_4": torch.norm(self.synthesis.first_block.conv1.affine.weight, p=2).item(),
                "res_8": torch.norm(self.synthesis.blocks[0].conv0.affine.weight, p=2).item(),
                "res_16": torch.norm(self.synthesis.blocks[1].conv0.affine.weight, p=2).item(),
                "res_32": torch.norm(self.synthesis.blocks[2].conv0.affine.weight, p=2).item(),
                "res_64": torch.norm(self.synthesis.blocks[3].conv0.affine.weight, p=2).item(),
                "res_128": torch.norm(self.synthesis.blocks[4].conv0.affine.weight, p=2).item(),
            }

        else:
            ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
            img = self.synthesis(ws, noise_mode = noise_mode)
            return_dict = {
                "image": img
            }
        return return_dict

class SynthesisNetwork3D(torch.nn.Module):

    def __init__(self, w_dim, img_resolution, img_channels, enc_feats=None, channel_multiplier = 1, bias_resolution = 64, **unused_kwargs):
        super().__init__()

        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        start_res = min(list(enc_feats.keys())) if enc_feats is not None else 4
        self.block_resolutions = [2 ** i for i in range(int(np.log2(start_res)), self.img_resolution_log2 + 1)]
        self.num_ws = 2 * (len(self.block_resolutions) + 1)
        # channels_dict = {4: 512, 8: 512, 16: 512, 32: 256, 64: 128}
        channels_dict = {
            4: 512, 
            8: 512, 
            16: 512, 
            32: 256 * channel_multiplier,
            64: 128 * channel_multiplier,
            128: 64 * channel_multiplier,
            256: 32 * channel_multiplier
        }
        # channels_dict = {4: 256, 8: 128, 16: 64, 32: 32, 64: 16, 128: 8, 256: 4}
        self.blocks = torch.nn.ModuleList()
        self.biases = torch.nn.ParameterList()
        self.num_ws = 0
        self.first_block = SynthesisPrologue(channels_dict[self.block_resolutions[0]], w_dim=w_dim,
                                             resolution=self.block_resolutions[0], img_channels=img_channels,
                                             enc_dim=enc_feats[self.block_resolutions[0]] if enc_feats is not None else 0)
        self.num_ws += self.first_block.num_ws
        for res in self.block_resolutions[1:]:
            in_channels = channels_dict[res // 2]
            out_channels = channels_dict[res]
            block = SynthesisBlock(
                in_channels, out_channels, w_dim=w_dim, resolution=res, 
                img_channels=img_channels, enc_dim=enc_feats[res] if enc_feats is not None else 0,
                use_const_bias = res <= bias_resolution
            )
            self.blocks.append(block)
            self.num_ws += block.num_ws

    def forward(self, ws, feats=None, noise_mode='random', update_emas=False):
        split_ws = [ws[:, 0:2, :]] + [ws[:, 2 * n + 1: 2 * n + 4, :] for n in range(len(self.block_resolutions))]
        res = self.block_resolutions[0]
        feat = feats[res] if feats is not None else None
        x, img = self.first_block(split_ws[0], feat = feat, noise_mode = noise_mode)
        # print_template = "res: {}, norm: {}" # TODO
        # print_list = [] # TODO
        for i in range(len(self.block_resolutions) - 1):
            res = self.block_resolutions[i+1]
            feat = feats[res] if feats is not None else None
            x, img = self.blocks[i](x, img, split_ws[i + 1], feat, noise_mode)
            # print_list.append(print_template.format(res, torch.norm(x, p=2).item())) # TODO
        # print(", ".join(print_list))    # TODO
        return img


class SynthesisPrologue(torch.nn.Module):

    def __init__(self, out_channels, w_dim, resolution, img_channels, enc_dim=0):
        super().__init__()
        self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.img_channels = img_channels
        self.const = torch.nn.Parameter(torch.randn([out_channels, resolution, resolution, resolution]))
        self.conv1 = SynthesisLayer(out_channels+enc_dim, out_channels, w_dim=w_dim, resolution=resolution)
        self.torgb = ToRGBLayer(out_channels, img_channels, w_dim=w_dim)
        self.num_ws = 2

    def forward(self, ws, feat=None, noise_mode='random'):
        w_iter = iter(ws.unbind(dim=1))
        x = self.const.unsqueeze(0).repeat([ws.shape[0], 1, 1, 1, 1])
        if feat is not None:
            x = torch.cat((x, feat), 1)
        x = self.conv1(x, next(w_iter), noise_mode=noise_mode)
        img = self.torgb(x, next(w_iter))
        return x, img


class SynthesisBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, w_dim, resolution, img_channels, enc_dim=0, use_const_bias = False):
        super().__init__()
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.out_channels = out_channels
        self.num_conv = 0
        self.num_torgb = 0
        self.resampler = SmoothUpsample()
        self.conv0 = SynthesisLayer(in_channels, out_channels, w_dim=w_dim, resolution=resolution, resampler=self.resampler)
        self.conv1 = SynthesisLayer(out_channels+enc_dim, out_channels, w_dim=w_dim, resolution=resolution)
        self.torgb = ToRGBLayer(out_channels, img_channels, w_dim=w_dim)
        self.num_ws = 3

        # add constant bias
        if use_const_bias:
            self.const_bias = torch.nn.Parameter(torch.randn([out_channels, resolution, resolution, resolution]) / np.sqrt(out_channels))
        else:
            self.const_bias = None

    def forward(self, x, img, ws, feat, noise_mode='random'):
        w_iter = iter(ws.unbind(dim=1))

        x = self.conv0(x, next(w_iter), noise_mode=noise_mode)
        if feat is not None:
            x = torch.cat((x, feat), 1)
        if self.const_bias is not None:
            x = x + self.const_bias
        x = self.conv1(x, next(w_iter), noise_mode=noise_mode)

        y = self.torgb(x, next(w_iter))
        img = self.resampler(img)
        img = img.add_(y)

        return x, img


class ToRGBLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, w_dim, kernel_size=1):
        super().__init__()
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size, kernel_size]))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.weight_gain = 1 / np.sqrt(in_channels) * (kernel_size ** 3)

    def forward(self, x, w):
        styles = self.affine(w) * self.weight_gain
        x = modulated_conv3d(x=x, weight=self.weight, styles=styles, demodulate=False)
        return x + self.bias[None, :, None, None, None] #torch.clamp(x + self.bias[None, :, None, None, None], -256, 256)


class SynthesisLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, w_dim, resolution, kernel_size=3, resampler=identity, activation='lrelu'):
        super().__init__()
        self.resolution = resolution
        self.resampler = resampler
        self.activation = activation_funcs[activation]['fn']
        self.activation_gain = activation_funcs[activation]['def_gain']
        self.padding = kernel_size // 2
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size, kernel_size]))

        self.register_buffer('noise_const', torch.randn([resolution, resolution, resolution]))
        self.noise_strength = torch.nn.Parameter(torch.zeros([1]))

        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))

    def forward(self, x, w, noise_mode, gain=1):
        styles = self.affine(w)

        noise = None
        if noise_mode == 'random':
            noise = torch.randn([x.shape[0], 1, self.resolution, self.resolution, self.resolution], device=x.device) * self.noise_strength
        if noise_mode == 'const':
            noise = self.noise_const * self.noise_strength

        x = modulated_conv3d(x=x, weight=self.weight, styles=styles, padding=self.padding)
        x = self.resampler(x)
        x = x + noise

        return clamp_gain(self.activation(x + self.bias[None, :, None, None, None]), self.activation_gain * gain, 256 * gain)


class MappingNetwork(torch.nn.Module):

    def __init__(self, z_dim, c_dim, w_dim, num_ws, num_layers=8, activation='lrelu', lr_multiplier=0.01, w_avg_beta=0.995):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        features_list = [z_dim ] + [w_dim] * num_layers

        self.layers = torch.nn.ModuleList()
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            self.layers.append(
                FullyConnectedLayer(
                    in_features, out_features, activation=activation, 
                    lr_multiplier=lr_multiplier if self.c_dim == 0 else 1, # in text to image gan, lr_multiplier is 1 for the mapping network
                    )
                )
        self.embed = FullyConnectedLayer(self.c_dim + out_features, self.w_dim) if self.c_dim > 0 else None

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer('w_avg', torch.zeros([w_dim]))

    def forward(self, z, c = None, truncation_psi=1, truncation_cutoff=None, skip_w_avg_update=False):
        # Embed, normalize, and concat inputs.
        x = normalize_2nd_moment(z)

        # Main layers.
        for idx in range(self.num_layers):
            x = self.layers[idx](x)
        
        # Update moving average of W.
        # if self.w_avg_beta is not None and self.training and not skip_w_avg_update:
            # self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        if self.c_dim > 0:
            x = torch.cat((x, c), dim=1)
            x = self.embed(x)
        else:
            x = x

        # Broadcast.
        if self.num_ws is not None:
            x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        # Apply truncation.
        if truncation_psi != 1:
            raise NotImplementedError("Truncation is not implemented")
            if self.num_ws is None or truncation_cutoff is None:
                x = self.w_avg.lerp(x, truncation_psi)
            else:
                x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)

        return x
    
