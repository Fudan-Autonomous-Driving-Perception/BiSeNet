import torch
import torch.nn as nn
import torch.nn.functional as F

class SCNN(nn.Module):
    def __init__(
            self,
            ms_ks=9,channel=128,
            pretrained=False
    ):
        """
        Argument
            ms_ks: kernel size in message passing conv
        """
        super(SCNN, self).__init__()
        self.pretrained = pretrained

        self.layer1 = nn.Sequential(
            nn.Conv2d(128, 1024, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(1024),nn.ReLU(),
            nn.Conv2d(1024, 128, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU()  # (nB, 128, 36, 100)
        )

        # add message passing
        self.message_passing = nn.ModuleList()
        self.message_passing.add_module('up_down',    nn.Conv2d(channel, channel, (1, ms_ks), padding=(0, ms_ks // 2), bias=False))
        self.message_passing.add_module('down_up',    nn.Conv2d(channel, channel, (1, ms_ks), padding=(0, ms_ks // 2), bias=False))
        self.message_passing.add_module('left_right', nn.Conv2d(channel, channel, (ms_ks, 1), padding=(ms_ks // 2, 0), bias=False))
        self.message_passing.add_module('right_left', nn.Conv2d(channel, channel, (ms_ks, 1), padding=(ms_ks // 2, 0), bias=False))
        # (nB, channel, 36, 100)

        if not pretrained:
            self.weight_init()

    def forward(self, x):#x the result of backbone
        x = self.layer1(x)
        x = self.message_passing_forward(x)
        x = self.layer1(x)

        return x

    def message_passing_forward(self, x):
        Vertical = [True, True, False, False]
        Reverse = [False, True, False, True]
        for ms_conv, v, r in zip(self.message_passing, Vertical, Reverse):
            x = self.message_passing_once(x, ms_conv, v, r)
        return x

    def message_passing_once(self, x, conv, vertical=True, reverse=False):
        """
        Argument:
        ----------
        x: input tensor
        vertical: vertical message passing or horizontal
        reverse: False for up-down or left-right, True for down-up or right-left
        """
        nB, C, H, W = x.shape
        if vertical:
            slices = [x[:, :, i:(i + 1), :] for i in range(H)]
            dim = 2
        else:
            slices = [x[:, :, :, i:(i + 1)] for i in range(W)]
            dim = 3
        if reverse:
            slices = slices[::-1]

        out = [slices[0]]
        for i in range(1, len(slices)):
            out.append(slices[i] + F.relu(conv(out[i - 1])))
        if reverse:
            out = out[::-1]
        return torch.cat(out, dim=dim)

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.reset_parameters()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data[:] = 1.
                m.bias.data.zero_()


if __name__=='__main__':
    pass
    a=torch.randn((3,128,64,64))
    scnn=SCNN()
    res=scnn(a)
    print(res.shape)