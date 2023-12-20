import torch
import torch.nn as nn


class SegNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_conv0 = nn.Sequential(
                            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                            nn.ReLU(inplace=True),
                            nn.BatchNorm2d(64),
                         )
        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) # 256 -> 128
        self.enc_conv1 = nn.Sequential(
                            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                            nn.ReLU(inplace=True),
                            nn.BatchNorm2d(128),
                         )
        self.pool1 =  nn.MaxPool2d(kernel_size=2, stride=2, padding=0) # 128 -> 64
        self.enc_conv2 = nn.Sequential(
                            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                            nn.ReLU(inplace=True),
                            nn.BatchNorm2d(256),
                         )
        self.pool2 =  nn.MaxPool2d(kernel_size=2, stride=2, padding=0) # 64 -> 32
        self.enc_conv3 = nn.Sequential(
                            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                            nn.ReLU(inplace=True),
                            nn.BatchNorm2d(512),
                         )
        self.pool3 =  nn.MaxPool2d(kernel_size=2, stride=2, padding=0) # 32 -> 16

        # bottleneck
        self.bottleneck_conv = nn.Sequential(
                            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                            nn.ReLU(inplace=True),
                            nn.BatchNorm2d(512),
                         )

        # decoder (upsampling)
        self.upsample0 = nn.Upsample(scale_factor=2, mode='bilinear')# 16 -> 32
        self.dec_conv0 = nn.Sequential(
                            nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                            nn.ReLU(inplace=True),
                            nn.BatchNorm2d(256),
                         )
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear') # 32 -> 64
        self.dec_conv1 = nn.Sequential(
                            nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                            nn.ReLU(inplace=True),
                            nn.BatchNorm2d(128),
                         )
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear')  # 64 -> 128
        self.dec_conv2 = nn.Sequential(
                            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                            nn.ReLU(inplace=True),
                            nn.BatchNorm2d(64),
                         )
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear')  # 128 -> 256
        self.dec_conv3 = nn.Sequential(
                            nn.Conv2d(64, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),

                            nn.BatchNorm2d(1),
                         )

    def forward(self, x):
        # encoder
        e0 = self.pool0(self.enc_conv0(x))
        e1 = self.pool1(self.enc_conv1(e0))
        e2 = self.pool2(self.enc_conv2(e1))
        e3 =self.pool3(self.enc_conv3(e2))

        # bottleneck
        b = self.bottleneck_conv(e3)

        # decoder
        d0 = self.dec_conv0(self.upsample0(b))
        d1 = self.dec_conv1(self.upsample1(d0))
        d2 = self.dec_conv2(self.upsample2(d1))
        d3 = self.dec_conv3(self.upsample3(d2)) 
        return d3

