from torch import nn
import torch
import torch.nn.functional as F
import imageio
from skimage.transform import resize
import numpy as np

from modules.util import SameBlock2d, DownBlock2d, UpBlock2d

img = "source_image/image.jpg"
source_image = imageio.imread(img)
source_image = resize(source_image, (256, 256))[..., :3]  # (256, 256, 3)
source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)  # torch.Size([1, 3, 256, 256])

class AntiAliasInterpolation2d(nn.Module):
    """
    Band-limited downsampling, for better preservation of the input signal.
    """
    def __init__(self, channels, scale):
        super(AntiAliasInterpolation2d, self).__init__()
        sigma = (1 / scale - 1) / 2  # 1.5
        kernel_size = 2 * round(sigma * 4) + 1  # 13
        self.ka = kernel_size // 2  # 6
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka  # 6

        kernel_size = [kernel_size, kernel_size]
        sigma = [sigma, sigma]
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
                ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-(mgrid - mean) ** 2 / (2 * std ** 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels
        self.scale = scale
        inv_scale = 1 / scale
        self.int_inv_scale = int(inv_scale)

    def forward(self, input):
        if self.scale == 1.0:
            return input

        out = F.pad(input, (self.ka, self.kb, self.ka, self.kb))
        out = F.conv2d(out, weight=self.weight, groups=self.groups)
        out = out[:, :, ::self.int_inv_scale, ::self.int_inv_scale]

        return out


down = AntiAliasInterpolation2d(3, 0.25)
src_img = down(source)  # torch.Size([1, 3, 64, 64])

first = SameBlock2d(3, 64, kernel_size=(7, 7), padding=(3, 3))
out = first(src_img)  # torch.Size([1, 64, 64, 64])

num_down_blocks = 2
block_expansion = 64
max_features = 1024
down_blocks = []
for i in range(num_down_blocks):
    in_features = min(max_features, block_expansion * (2 ** i))
    out_features = min(max_features, block_expansion * (2 ** (i + 1)))
    down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
down_blocks = nn.ModuleList(down_blocks)
for i in range(len(down_blocks)):
    out = down_blocks[i](out)

# torch.Size([1, 256, 16, 16])
up_blocks = []
for i in range(num_down_blocks):
    in_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i)))
    out_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i - 1)))
    up_blocks.append(UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
up_blocks = nn.ModuleList(up_blocks)
for i in range(len(up_blocks)):
    out = up_blocks[i](out)
print(out.shape)