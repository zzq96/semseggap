# Model implementation in PyTorch
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

##########################################
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, d1, d2, skip=False, stride = 1):
        super(ResidualBlock, self).__init__()
        self.skip = skip

        self.conv1 = nn.Conv2d(in_channels, d1, 1, stride = stride,bias = False)
        self.bn1 = nn.BatchNorm2d(d1)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(d1, d1, 3, padding = 1,bias = False)
        self.bn2 = nn.BatchNorm2d(d1)

        self.conv3 = nn.Conv2d(d1, d2, 1,bias = False)
        self.bn3 = nn.BatchNorm2d(d2)

        if not self.skip:
            self.conv4 = nn.Conv2d(in_channels, d2, 1, stride=stride,bias = False)
            self.bn4 = nn.BatchNorm2d(d2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.skip:
            residual = x
        else:
            residual = self.conv4(x)
            residual = self.bn4(residual)

        out += residual
        out = self.relu(out)
        
        return out

class UpProj_Block(nn.Module):
    def __init__(self, in_channels, out_channels, batch_size):
        super(UpProj_Block, self).__init__()
        self.batch_size = batch_size

        self.conv1 = nn.Conv2d(in_channels, out_channels, (3,3))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (2,3))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (3,2))
        self.conv4 = nn.Conv2d(in_channels, out_channels, (2,2))

        self.conv5 = nn.Conv2d(in_channels, out_channels, (3,3))
        self.conv6 = nn.Conv2d(in_channels, out_channels, (2,3))
        self.conv7 = nn.Conv2d(in_channels, out_channels, (3,2))
        self.conv8 = nn.Conv2d(in_channels, out_channels, (2,2))

        self.bn1_1 = nn.BatchNorm2d(out_channels)
        self.bn1_2 = nn.BatchNorm2d(out_channels)

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv9 = nn.Conv2d(out_channels, out_channels , 3,padding = 1)

    def prepare_indices(self, before, row, col, after, dims):

        x0, x1, x2, x3 = np.meshgrid(before, row, col, after)
        dtype = torch.cuda.FloatTensor
        x_0 = torch.from_numpy(x0.reshape([-1]))
        x_1 = torch.from_numpy(x1.reshape([-1]))
        x_2 = torch.from_numpy(x2.reshape([-1]))
        x_3 = torch.from_numpy(x3.reshape([-1]))

        linear_indices = x_3 + dims[3] * x_2  + 2 * dims[2] * dims[3] * x_0 * 2 * dims[1] + 2 * dims[2] * dims[3] * x_1
        linear_indices_int = linear_indices.int()
        return linear_indices_int

    def forward(self, x, BN=True):
        out1 = self.unpool_as_conv(x, id=1)
        out1 = self.conv9(out1)

        if BN:
            out1 = self.bn2(out1)

        out2 = self.unpool_as_conv(x, ReLU=False, id=2)

        out = out1+out2

        out = self.relu(out)
        return out

    def unpool_as_conv(self, x, BN=True, ReLU=True, id=1):
        if(id==1):
            out1 = self.conv1(torch.nn.functional.pad(x,(1,1,1,1)))
            out2 = self.conv2(torch.nn.functional.pad(x,(1,1,1,0)))
            out3 = self.conv3(torch.nn.functional.pad(x,(1,0,1,1)))
            out4 = self.conv4(torch.nn.functional.pad(x,(1,0,1,0)))
        else:
            out1 = self.conv5(torch.nn.functional.pad(x,(1,1,1,1)))
            out2 = self.conv6(torch.nn.functional.pad(x,(1,1,1,0)))
            out3 = self.conv7(torch.nn.functional.pad(x,(1,0,1,1)))
            out4 = self.conv8(torch.nn.functional.pad(x,(1,0,1,0)))

        out1 = out1.permute(0,2,3,1)
        out2 = out2.permute(0,2,3,1)
        out3 = out3.permute(0,2,3,1)
        out4 = out4.permute(0,2,3,1)

        dims = out1.size()
        dim1 = dims[1] * 2
        dim2 = dims[2] * 2

        A_row_indices = range(0, dim1, 2)
        A_col_indices = range(0, dim2, 2)
        B_row_indices = range(1, dim1, 2)
        B_col_indices = range(0, dim2, 2)
        C_row_indices = range(0, dim1, 2)
        C_col_indices = range(1, dim2, 2)
        D_row_indices = range(1, dim1, 2)
        D_col_indices = range(1, dim2, 2)

        all_indices_before = range(int(self.batch_size))
        all_indices_after = range(dims[3])

        A_linear_indices = self.prepare_indices(all_indices_before, A_row_indices, A_col_indices, all_indices_after, dims)
        B_linear_indices = self.prepare_indices(all_indices_before, B_row_indices, B_col_indices, all_indices_after, dims)
        C_linear_indices = self.prepare_indices(all_indices_before, C_row_indices, C_col_indices, all_indices_after, dims)
        D_linear_indices = self.prepare_indices(all_indices_before, D_row_indices, D_col_indices, all_indices_after, dims)

        A_flat = (out1.permute(1, 0, 2, 3)).contiguous().view(-1)
        B_flat = (out2.permute(1, 0, 2, 3)).contiguous().view(-1)
        C_flat = (out3.permute(1, 0, 2, 3)).contiguous().view(-1)
        D_flat = (out4.permute(1, 0, 2, 3)).contiguous().view(-1)

        size_ = A_linear_indices.size()[0] + B_linear_indices.size()[0]+C_linear_indices.size()[0]+D_linear_indices.size()[0]

        Y_flat = torch.cuda.FloatTensor(size_).zero_()

        Y_flat.scatter_(0, A_linear_indices.type(torch.cuda.LongTensor).squeeze(),A_flat.data)
        Y_flat.scatter_(0, B_linear_indices.type(torch.cuda.LongTensor).squeeze(),B_flat.data)
        Y_flat.scatter_(0, C_linear_indices.type(torch.cuda.LongTensor).squeeze(),C_flat.data)
        Y_flat.scatter_(0, D_linear_indices.type(torch.cuda.LongTensor).squeeze(),D_flat.data)


        Y = Y_flat.view(-1, dim1, dim2, dims[3])
        Y=Variable(Y.permute(0,3,1,2))
        Y=Y.contiguous()

        if(id==1):
            if BN:
                Y = self.bn1_1(Y)
        else:
            if BN:
                Y = self.bn1_2(Y)

        if ReLU:
            Y = self.relu(Y)

        return Y

class SkipUp(nn.Module):
    def __init__(self, in_size, out_size, scale):
        super(SkipUp,self).__init__()
        self.unpool = nn.Upsample(scale_factor=scale,mode='bilinear')
        self.conv = nn.Conv2d(in_size,out_size,3,1,1)
        self.conv2 = nn.Conv2d(out_size,out_size,3,1,1)
    def forward(self,inputs):
        outputs = self.unpool(inputs)
        outputs = self.conv(outputs)
        # outputs = self.conv2(outputs)
        return outputs
##########################################

class Model_2b_depgd_GAP_MS(nn.Module):
    def __init__(self, block1, block2, batch_size,n_class):
        super(Model_2b_depgd_GAP_MS, self).__init__()
        self.batch_size=batch_size
        self.n_class=n_class

        # backbone
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 7, stride=2, padding=4)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(3,stride=2)

        self.proj_layer1 = self.make_proj_layer(block1, 64 , d1 = 64, d2 = 256, stride = 1)
        self.skip_layer1_1 = self.make_skip_layer(block1, 256, d1 = 64, d2 = 256, stride=1)
        self.skip_layer1_2 = self.make_skip_layer(block1, 256, d1 = 64, d2 = 256, stride=1)

        self.proj_layer2 = self.make_proj_layer(block1, 256 , d1 = 128, d2 = 512, stride = 2)
        self.skip_layer2_1 = self.make_skip_layer(block1, 512, d1 = 128, d2 = 512)
        self.skip_layer2_2 = self.make_skip_layer(block1, 512, d1 = 128, d2 = 512)
        self.skip_layer2_3 = self.make_skip_layer(block1, 512, d1 = 128, d2 = 512)

        self.proj_layer3 = self.make_proj_layer(block1, 512 , d1 = 256, d2 = 1024, stride=2)
        self.skip_layer3_1 = self.make_skip_layer(block1, 1024, d1 = 256, d2 = 1024)
        self.skip_layer3_2 = self.make_skip_layer(block1, 1024, d1 = 256, d2 = 1024)
        self.skip_layer3_3 = self.make_skip_layer(block1, 1024, d1 = 256, d2 = 1024)
        self.skip_layer3_4 = self.make_skip_layer(block1, 1024, d1 = 256, d2 = 1024)
        self.skip_layer3_5 = self.make_skip_layer(block1, 1024, d1 = 256, d2 = 1024)

        self.proj_layer4 = self.make_proj_layer(block1, 1024 , d1 = 512, d2 = 2048, stride=2)
        self.skip_layer4_1 = self.make_skip_layer(block1, 2048, d1 = 512, d2 = 2048)
        self.skip_layer4_2 = self.make_skip_layer(block1, 2048, d1 = 512, d2 = 2048)

        self.conv2 = nn.Conv2d(2048,1024,1)
        self.bn2 = nn.BatchNorm2d(1024)

        #depth
        self.dep_up_conv1 = self.make_up_conv_layer(block2, 1024, 512, self.batch_size)
        self.dep_up_conv2 = self.make_up_conv_layer(block2, 512, 256, self.batch_size)
        self.dep_up_conv3 = self.make_up_conv_layer(block2, 256, 128, self.batch_size)
        self.dep_up_conv4 = self.make_up_conv_layer(block2, 128, 64, self.batch_size)
        self.dep_skip_up1 = SkipUp(512,64,8)
        self.dep_skip_up2 = SkipUp(256,64,4)
        self.dep_skip_up3 = SkipUp(128,64,2)
        self.dep_conv3 = nn.Conv2d(64,1,3, padding=1)#64,1
        # self.upsample = nn.UpsamplingBilinear2d(size = (480,640))
        self.upsample = nn.Upsample(size = (480,640),mode='bilinear')
        #sem
        self.sem_up_conv1 = self.make_up_conv_layer(block2, 1024, 512, self.batch_size)
        self.sem_up_conv2 = self.make_up_conv_layer(block2, 512, 256, self.batch_size)
        self.sem_up_conv3 = self.make_up_conv_layer(block2, 256, 128, self.batch_size)
        self.sem_up_conv4 = self.make_up_conv_layer(block2, 128, 64, self.batch_size)
        self.sem_skip_up1 = SkipUp(512,64,8)
        self.sem_skip_up2 = SkipUp(256,64,4)
        self.sem_skip_up3 = SkipUp(128,64,2)
        self.sem_conv3_2 = nn.Conv2d(64,self.n_class,3, padding=1)#64,1

        self.midn=32
        self.GAP_convSem=nn.Conv2d(64,self.midn,1)
        self.GAP_convDep1=nn.Conv2d(64,self.midn,1)
        self.GAP_convDep2=nn.Conv2d(64,self.midn,1)
        self.GAP_convCom=nn.Conv2d(self.midn,64,1)
        self.GAP_BNSem=nn.BatchNorm2d(self.midn)
        self.GAP_BNDep1=nn.BatchNorm2d(self.midn)
        self.GAP_BNDep2=nn.BatchNorm2d(self.midn)
        self.GAP_BNCom=nn.BatchNorm2d(64)

        self.sideup1=nn.Upsample((8,10),mode='bilinear')
        self.sideup2=nn.Upsample((15,19),mode='bilinear')
        self.sideup3=nn.Upsample((29,38),mode='bilinear')
        self.sideup4=nn.Upsample((57,76),mode='bilinear')
        self.sidedconv1=nn.Conv2d(2048,64,1)
        self.sidedconv2=nn.Conv2d(1024,64,1)
        self.sidedconv3=nn.Conv2d(512,64,1)
        self.sidedconv4=nn.Conv2d(256,64,1)
        self.sidecconv1=nn.Conv2d(128,64,3,padding=1)
        self.sidecconv2=nn.Conv2d(128,64,3,padding=1)
        self.sidecconv3=nn.Conv2d(128,64,3,padding=1)
        self.sidecconv4=nn.Conv2d(128,64,3,padding=1)
        self.sideoconv1=nn.Conv2d(64,self.n_class,3,padding=1)
        self.sideoconv2=nn.Conv2d(64,self.n_class,3,padding=1)
        self.sideoconv3=nn.Conv2d(64,self.n_class,3,padding=1)
        self.sideoconv4=nn.Conv2d(64,self.n_class,3,padding=1)

    def make_proj_layer(self, block, in_channels, d1, d2, stride = 1, pad=0):
        return block(in_channels, d1, d2, skip=False, stride = stride)

    def make_skip_layer(self, block, in_channels, d1, d2, stride=1, pad=0):
        return block(in_channels, d1, d2, skip=True, stride=stride)

    def make_up_conv_layer(self, block, in_channels, out_channels, batch_size):
        return block(in_channels, out_channels, batch_size)

    def forward(self,x_1):
        out_1 = self.conv1(x_1)
        out_1 = self.bn1(out_1)
        out_1 = self.relu(out_1)
        out_1 = self.max_pool(out_1)
        out_76_57 = self.proj_layer1(out_1)
        out_1 = self.skip_layer1_1(out_76_57)
        out_1 = self.skip_layer1_2(out_1)
        out_38_29 = self.proj_layer2(out_1)
        out_1 = self.skip_layer2_1(out_38_29)
        out_1 = self.skip_layer2_2(out_1)
        out_1 = self.skip_layer2_3(out_1)
        out_19_15 = self.proj_layer3(out_1)
        out_1 = self.skip_layer3_1(out_19_15)
        out_1 = self.skip_layer3_2(out_1)
        out_1 = self.skip_layer3_3(out_1)
        out_1 = self.skip_layer3_4(out_1)
        out_1 = self.skip_layer3_5(out_1)
        out_10_8 = self.proj_layer4(out_1)
        out_1 = self.skip_layer4_1(out_10_8)
        out_1 = self.skip_layer4_2(out_1)
        out_1 = self.conv2(out_1)
        out_1 = self.bn2(out_1)

        #Upconv section
        #Depth Prediction Branch
        dep_out_1up1 = self.dep_up_conv1(out_1)
        dep_out_1up2 = self.dep_up_conv2(dep_out_1up1)
        dep_out_1up3 = self.dep_up_conv3(dep_out_1up2)
        dep_out_1 = self.dep_up_conv4(dep_out_1up3)
        dep_skipup1=self.dep_skip_up1(dep_out_1up1)
        dep_skipup2=self.dep_skip_up2(dep_out_1up2)
        dep_skipup3=self.dep_skip_up3(dep_out_1up3)
        dep_out_feat=dep_out_1 + dep_skipup1+dep_skipup2+dep_skipup3
        dep_out_1 = self.dep_conv3(dep_out_feat)
        dep_out_1 = self.upsample(dep_out_1)
        #Sem Prediction Branch
        sem_out_1up1 = self.sem_up_conv1(out_1)
        sem_out_1up2 = self.sem_up_conv2(sem_out_1up1+dep_out_1up1)
        sem_out_1up3 = self.sem_up_conv3(sem_out_1up2+dep_out_1up2)
        sem_out_1 = self.sem_up_conv4(sem_out_1up3+dep_out_1up3)
        sem_skipup1=self.sem_skip_up1(sem_out_1up1)
        sem_skipup2=self.sem_skip_up2(sem_out_1up2)
        sem_skipup3=self.sem_skip_up3(sem_out_1up3)
        sem_out_feat=sem_out_1 + sem_skipup1+sem_skipup2+sem_skipup3
        # GAPd
        Dep1=self.GAP_BNDep1(self.GAP_convDep1(dep_out_feat))
        Dep2=self.GAP_BNDep2(self.GAP_convDep2(dep_out_feat))
        theta_d=Dep1.view(self.batch_size,self.midn,-1)
        theta_d=theta_d.permute(0,2,1)
        phi_d=Dep2.view(self.batch_size,self.midn,-1)
        DepGuid=torch.matmul(theta_d,phi_d)
        #DepGuid=torch.einsum('bik,bkj->bij',[theta_d,phi_d])
        DepGuid=F.softmax(DepGuid,dim=-1)
        Sem=self.GAP_BNSem(self.GAP_convSem(sem_out_feat))
        g_s=Sem.view(self.batch_size,self.midn,-1)
        g_s=g_s.permute(0,2,1)
        Com=torch.matmul(DepGuid,g_s)
        #Com=torch.einsum('bij,bjk->bik',[DepGuid,g_s])
        Com=Com.permute(0,2,1).contiguous()
        Com=Com.view(self.batch_size,self.midn,*dep_out_feat.size()[2:])
        GAP=self.GAP_BNCom(self.GAP_convCom(Com))+sem_out_feat

        sem_out_1 = self.sem_conv3_2(GAP)
        sem_out_1 = self.upsample(sem_out_1)

        side_feat1 = torch.cat((self.sidedconv1(out_10_8),self.sideup1(GAP)),1)
        side_feat1 = self.sidecconv1(side_feat1)
        side_out1 = self.sideoconv1(side_feat1)
        side_out1 = self.upsample(side_out1)
        side_feat2 = torch.cat((self.sidedconv2(out_19_15),self.sideup2(side_feat1)),1)
        side_feat2 = self.sidecconv2(side_feat2)
        side_out2 = self.sideoconv2(side_feat2)
        side_out2 = self.upsample(side_out2)
        side_feat3 = torch.cat((self.sidedconv3(out_38_29),self.sideup3(side_feat2)),1)
        side_feat3 = self.sidecconv3(side_feat3)
        side_out3 = self.sideoconv3(side_feat3)
        side_out3 = self.upsample(side_out3)
        side_feat4 = torch.cat((self.sidedconv4(out_76_57),self.sideup4(side_feat3)),1)
        side_feat4 = self.sidecconv4(side_feat4)
        side_out4 = self.sideoconv4(side_feat4)
        side_out4 = self.upsample(side_out4)

        return dep_out_1, sem_out_1, side_out1,side_out2,side_out3,side_out4
