from models.base import BaseModel
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.cuda.amp as amp


class DoubleConv(nn.Module):
    """
    (3x3 conv -> BN -> ReLU) ** 2

    Attributes:
        in_channels (int): number of input channels.
        out_channels (int): number of the output channels.
        kernel_size (int): size of the convolutional kernel.
        padding (int): padding to be applied to the input.

    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super(DoubleConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size,
                      padding=self.padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=self.out_channels, kernel_size=self.kernel_size,
                      padding=self.padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """
    2x2 MaxPool -> doubleConv

    Attributes:
        in_channels (int): number of input channels.
        out_channels (int): number of the output channels.
        maxpool_kernel_size (int): size of the MaxPooling kernel.

    """

    def __init__(self, in_channels: int, out_channels: int):
        super(Down, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.maxpool_kernel_size = 2

        self.double_conv = DoubleConv(in_channels=self.in_channels, out_channels=self.out_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=self.maxpool_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_doubleconv = self.double_conv(x)  # out_doubleconv: save the convolution output for the skip connection
        out_maxpool = self.maxpool(out_doubleconv)
        return (out_doubleconv, out_maxpool)


class Up(nn.Module):
    """
    2x2 upconv -> concatenate --> doubleConv
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(Up, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upconv_kernel_size = 2

        self.upconv = nn.ConvTranspose2d(in_channels=self.in_channels - self.out_channels,
                                         out_channels=self.in_channels - self.out_channels,
                                         kernel_size=self.upconv_kernel_size, stride=2)
        self.double_conv = DoubleConv(in_channels=self.in_channels, out_channels=self.out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upconv(x)

        # resize x if is not the same size as skip
        if x.shape != skip.shape:
            x = TF.resize(x, size=skip.shape[2:], antialias=True)

        x = torch.cat([x, skip], axis=1)
        return self.double_conv(x)


class UNet(BaseModel):
    """
    UNet model for segmentation of images with num_classes classes
    """

    def __init__(self, input_channels = 3, num_classes= 1):
        # type: (int, int) -> None

        super(UNet, self).__init__()

        self.input_channels = input_channels
        print('input channels: ', self.input_channels)
        self.num_classes = num_classes
        print('output channels: ', self.num_classes)

        # downsampling
        self.down_conv1 = Down(in_channels=self.input_channels, out_channels=64)
        self.down_conv2 = Down(in_channels=64, out_channels=128)
        self.down_conv3 = Down(in_channels=128, out_channels=256)
        self.down_conv4 = Down(in_channels=256, out_channels=512)

        # bottleneck
        self.double_conv = DoubleConv(in_channels=512, out_channels=1024)

        # upsampling
        self.up_conv1 = Up(in_channels=(512 + 1024), out_channels=512)
        self.up_conv2 = Up(in_channels=(256 + 512), out_channels=256)
        self.up_conv3 = Up(in_channels=(128 + 256), out_channels=128)
        self.up_conv4 = Up(in_channels=(64 + 128), out_channels=64)

        # final
        self.up_conv5 = nn.Conv2d(in_channels=64, out_channels=self.num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # with amp.autocast():
        """
        downsampling -> bottleneck -> upsampling
        """

        skip_1, x = self.down_conv1(x)  # skip: output of the double convolution, x: output of the maxpool
        skip_2, x = self.down_conv2(x)
        skip_3, x = self.down_conv3(x)
        skip_4, x = self.down_conv4(x)

        x = self.double_conv(x)

        x = self.up_conv1(x, skip_4)
        x = self.up_conv2(x, skip_3)
        x = self.up_conv3(x, skip_2)
        x = self.up_conv4(x, skip_1)
        x = self.up_conv5(x) #fROM FINAL CONV THE RECONSTUCTED IMAGES IS OBTAINED (0, 255) , LOGIT USED TO FEED CROSS ENTROPY LOSS or BCEwithLogitsLoss
        x = torch.sigmoid(x) #APPLY SIGMOID TO OBTAIN THE PROBABILITY TO USE WITH BCE LOSS

        return x


def test():

    batch_size = 1
    input_channels = 3
    input_size_h = 256
    input_size_w = 256

    num_classes = 10

    model = UNet(input_channels = input_channels, num_classes=num_classes)
    x = torch.randn(batch_size, input_channels, input_size_h, input_size_w)
    print(x.shape)
    y = model(x)
    print(y.shape)

if __name__ == "__main__":
    test()




# import torch
# import torch.nn as nn
# import torchvision.transforms.functional as TF
#
# from models import BaseModel
#
#
# class DoubleConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(DoubleConv, self).__init__()
#         self.double_conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, x):
#         return self.double_conv(x)
#
#
# class AttentionBlock(nn.Module):
#     """
#     Attention mechanism for UNet to focus on important regions.
#     """
#     def __init__(self, g_channels, x_channels):
#         super(AttentionBlock, self).__init__()
#         self.gate_conv = nn.Conv2d(g_channels, x_channels, kernel_size=1)
#         self.input_conv = nn.Conv2d(x_channels, x_channels, kernel_size=1)
#         self.attn_conv = nn.Conv2d(x_channels, 1, kernel_size=1)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, g, x):
#         # Compute attention weights
#         g1 = self.gate_conv(g)
#         x1 = self.input_conv(x)
#         attn = self.sigmoid(self.attn_conv(g1 + x1))
#         return x * attn
#
#
# class Down(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(Down, self).__init__()
#         self.maxpool = nn.MaxPool2d(kernel_size=2)
#         self.double_conv = DoubleConv(in_channels, out_channels)
#
#     def forward(self, x):
#         x = self.maxpool(x)
#         return self.double_conv(x)
#
#
# class Up(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(Up, self).__init__()
#         self.upconv = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
#         self.double_conv = DoubleConv(in_channels, out_channels)
#         self.attention = AttentionBlock(g_channels=out_channels, x_channels=out_channels)
#
#     def forward(self, x, skip):
#         x = self.upconv(x)
#         if x.shape != skip.shape:
#             x = TF.resize(x, size=skip.shape[2:])
#         skip = self.attention(x, skip)
#         x = torch.cat([x, skip], dim=1)
#         return self.double_conv(x)
#
#
# class UNet1(BaseModel):
#     def __init__(self, input_channels=3, num_classes=10):
#         super(UNet1, self).__init__()
#
#         self.down1 = DoubleConv(input_channels, 64)
#         self.down2 = Down(64, 128)
#         self.down3 = Down(128, 256)
#         self.down4 = Down(256, 512)
#         self.bottleneck = DoubleConv(512, 1024)
#
#         self.up1 = Up(1024, 512)
#         self.up2 = Up(512, 256)
#         self.up3 = Up(256, 128)
#         self.up4 = Up(128, 64)
#
#         self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
#         self.sigmoid = nn.Sigmoid()
#         self.num_classes = num_classes
#
#
#     def forward(self, x):
#         skip1 = self.down1(x)
#         skip2 = self.down2(skip1)
#         skip3 = self.down3(skip2)
#         skip4 = self.down4(skip3)
#         bottleneck = self.bottleneck(skip4)
#
#         x = self.up1(bottleneck, skip4)
#         x = self.up2(x, skip3)
#         x = self.up3(x, skip2)
#         x = self.up4(x, skip1)
#
#         x = self.final_conv(x)
#         return self.sigmoid(x)  # For BCEWithLogitsLoss, remove this and use raw logits


def test():
    batch_size = 1
    input_channels = 3
    input_size = (256, 256)
    num_classes = 1

    model = UNet(input_channels=input_channels, num_classes=num_classes)
    x = torch.randn(batch_size, input_channels, *input_size)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")


if __name__ == "__main__":
    test()
