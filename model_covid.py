import torch
import torch.nn as nn
import torch.nn.functional as F

class PEPX(nn.Module):
    def __init__(self, in_ch, out_ch, p1, first_proj_fact=1/4, first_exp_fact=2, sec_proj_fac=1/4):
        super(PEPX, self).__init__()
        first_proj = int(in_ch * first_proj_fact)
        first_exp = int(first_proj * first_exp_fact)
        sec_proj = int(first_exp * sec_proj_fac)

        self.con1 = nn.Conv2d(in_channels=in_ch, out_channels=first_proj, kernel_size=1,
                              stride=1, padding=0, bias=True)
        self.con2 = nn.Conv2d(in_channels=first_proj, out_channels=first_exp, kernel_size=1,
                              stride=1, padding=0, bias=True)
        self.dwcConv = nn.Conv2d(in_channels=first_exp, out_channels=first_exp, kernel_size=3,
                                 stride=1, padding=1, bias=True)
        self.con3 = nn.Conv2d(in_channels=first_exp, out_channels=sec_proj, kernel_size=1,
                              stride=1, padding=0, bias=True)

        if p1:
            self.con4 = nn.Conv2d(in_channels=sec_proj, out_channels=out_ch, kernel_size=1,
                                  stride=2, padding=0, bias=True)
        else:
            self.con4 = nn.Conv2d(in_channels=sec_proj, out_channels=out_ch, kernel_size=1,
                                  stride=1, padding=0, bias=True)

    def forward(self, mapping_filters):
        out = self.con1(mapping_filters)
        out = self.con2(out)
        out = self.dwcConv(out)
        out = self.con3(out)
        out = self.con4(out)

        return out


class CovidNet(nn.Module):
    # Inputs an image and ouputs the prediction for the class

    def __init__(self, num_class):
        super(CovidNet, self).__init__()

        # CONV 7X7
        self.conv7 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)

        # CONV 1X1 (56x56x256)
        self.conv_top1 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=1, stride=2, padding=0, bias=False)

        # PEPX 1
        layers = []
        for i in range(1,4):
            if i == 1:
                layers.append(PEPX(64, 256, first_exp_fact=8, sec_proj_fac=1/2, p1=True))
            else:
                layers.append(PEPX(256 * i, 256, first_exp_fact=4, p1=False))

        self.pepx_1 = nn.ModuleList(layers)

        # CONV 1X1 (28x28x512)
        self.conv_top2 = nn.Conv2d(in_channels=256 + 3*256, out_channels=512, kernel_size=1, stride=2, padding=0, bias=False)

        # PEPX 2
        layers = []
        for i in range(1,5):
            if i == 1:
                layers.append(PEPX(3*256 + 256, 512, first_exp_fact=8, sec_proj_fac=1/2, p1=True))
            else:
                layers.append(PEPX(512 * i, 512, first_exp_fact=4, p1=False))
        self.pepx_2 = nn.ModuleList(layers)

        # CONV 1X1 (14x14x1024)
        self.conv_top3 = nn.Conv2d(in_channels=512 + 4*512, out_channels=1024, kernel_size=1, stride=2, padding=0, bias=False)
# 9899
        # PEPEX 3
        layers = []
        for i in range(1,7):
            if i == 1:
                layers.append(PEPX(4*512 + 512, 1024, first_exp_fact=4, sec_proj_fac=1/2, p1=True))
            else:
                layers.append(PEPX(1024 * i, 1024, p1=False))

        self.pepx_3 = nn.ModuleList(layers)

        # CONV 1X1 (7x7x2048)
        self.conv_top4 = nn.Conv2d(in_channels=1024 + 6*1024, out_channels=2048, kernel_size=1, stride=2, padding=0, bias=False)

        # PEPEX 4
        layers = []
        for i in range(1,4):
            if i == 1:
                layers.append(PEPX(6*1024 + 1024, 2048, sec_proj_fac=1/2, p1=True))
            else:
                layers.append(PEPX(2048 * i, 2048, p1=False))

        self.pepx_4 = nn.ModuleList(layers)

        # FC
        self.fc1 = nn.Sequential(nn.Linear(4*100352, 1024))
        self.fc2 = nn.Sequential(nn.Linear(1024, num_class))


    def forward(self, img):
        # CHUNK 0
        out_conv7 = self.conv7(img)

        # CHUNK 1
        top_1 = self.conv_top1(out_conv7)
        out_pepx1 = top_1
        for i, func in enumerate(self.pepx_1):
            if i == 0:
                out = func(out_conv7)
            else:
                out = func(out_pepx1)

            out_pepx1 = torch.cat((out_pepx1, out), 1)

        # CHUNK 2
        top_2 = self.conv_top2(out_pepx1)
        out_pepx2 = top_2
        for i, func in enumerate(self.pepx_2):
            if i == 0:
                out = func(out_pepx1)
            else:
                out = func(out_pepx2)
            out_pepx2 = torch.cat((out_pepx2, out), 1)

        # CHUNK 3
        top_3 = self.conv_top3(out_pepx2)
        out_pepx3 = top_3
        for i, func in enumerate(self.pepx_3):
            if i == 0:
                out = func(out_pepx2)
            else:
                out = func(out_pepx3)
            out_pepx3 = torch.cat((out_pepx3, out), 1)

        # CHUNK 4
        top_4 = self.conv_top4(out_pepx3)
        out_pepx4 = top_4
        for i, func in enumerate(self.pepx_4):
            if i == 0:
                out = func(out_pepx3)
            else:
                out = func(out_pepx4)
            out_pepx4 = torch.cat((out_pepx4, out), 1)

        # Final chunk
        flatten = torch.flatten(out_pepx4, start_dim=1)

        out = F.relu(self.fc1(flatten))
        out = F.relu(self.fc2(out))

        probs = torch.softmax(out, dim=1)
        # winners = probs.argmax(dim=1)

        return probs

    def get_n_params(self):
        """

        :return: number of parameters of this model
        """
        pp = 0
        for p in list(self.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn

        return pp