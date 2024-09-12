import torch
import torch.nn as nn

# filters, kernel, stride
darknet53 = [
    (32, 3, 1),
    (64, 3, 2),
    ("ResidualBlock", 1),
    (128, 3, 2),
    ("ResidualBlock", 2),
    (256, 3, 2),
    ("ResidualBlock", 8),
    (512, 3, 2),
    ("ResidualBlock", 8),
    (1024, 3, 2),
    ("ResidualBlock", 4)
]
cfg = [*darknet53,
       (512, 1, 1), (1024, 3, 1),
       (512, 1, 1), (1024, 3, 1),
       (512, 1, 1),
       "ScaleBlock",
       (256, 1, 1),
       "UpSampleBlock",
       (256, 1, 1), (512, 3, 1),
       (256, 1, 1), (512, 3, 1),
       (256, 1, 1),
       "ScaleBlock",
       (128, 1, 1),
       "UpSampleBlock",
       (128, 1, 1), (256, 3, 1),
       (128, 1, 1), (256, 3, 1),
       (128, 1, 1),
       "ScaleBlock"]


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, num_repeats):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_repeats):
            self.layers += [nn.Sequential(
                ConvBlock(in_channels, in_channels // 2, kernel_size=1, stride=1),
                ConvBlock(in_channels // 2, in_channels, kernel_size=3, stride=1, padding=1)
            )]
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ScaleBlock(nn.Module):
    def __init__(self, in_channels, num_classes=1):
        super().__init__()
        self.conv_b = nn.Sequential(
            ConvBlock(in_channels, in_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels * 2, 5 * 3, kernel_size=1, stride=1)
        )
        self.num_classes = num_classes

    def forward(self, x):
        x = self.conv_b(x)
        # .permute gives the common shape for yolo prediction (batch_size, 3, s, s, 6)
        # will it be better to use reshape without permute?
        # x.reshape(x.shape[0], 3, x.shape[2], x.shape[3], 6)?
        # they should have different effect, but which do we need?
        x = x.reshape(x.shape[0], 3, 5, x.shape[2], x.shape[3]).permute(0, 1, 3, 4, 2)

        return x


class YOLOv3(nn.Module):

    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self._create_layers()

    def forward(self, x):

        out = []
        route_connections = []
        for layer in self.layers:
            if isinstance(layer, ScaleBlock):
                out.append(layer(x))
                continue

            x = layer(x)
            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)

            elif isinstance(layer, nn.Upsample):
                x = torch.cat((x, route_connections[-1]), dim=1)
                route_connections.pop()

        return out

    def _create_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for layer in cfg:

            if isinstance(layer, tuple):
                if layer[0] != "ResidualBlock":
                    out_channels, kernel_size, stride = layer
                    padding = 1 if kernel_size == 3 else 0
                    layers.append(
                        ConvBlock(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
                    )
                    in_channels = out_channels
                else:
                    layers.append(ResidualBlock(in_channels, num_repeats=layer[1]))

            elif isinstance(layer, str):
                if layer == "ScaleBlock":
                    layers.append(
                        ScaleBlock(in_channels, num_classes=self.num_classes)
                    )

                elif layer == "UpSampleBlock":
                    layers.append(nn.Upsample(scale_factor=2))
                    in_channels *= 3

        return layers
