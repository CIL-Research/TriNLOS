import torch
import torch.nn as nn
from .trinlos_module import *


class Pipeline(nn.Module):

    def __init__(self, tra_dim=4, spatial=128, crop = 512, bin_len=0.01):
        super().__init__()
        ts_stride = [2,1,1]
        multiplue = 8
        self.tra_dim=tra_dim
        self.trafeature=Transient_TDown_2(1,int(tra_dim/2),ts_stride=ts_stride)
        self.makevol=tra2vol(tra_dim, spatial = spatial//ts_stride[1], crop = crop//ts_stride[0] , bin_len=bin_len*ts_stride[0])
        dim=tra_dim*multiplue
        self.volfeature=nn.Sequential(
                        MsFeat_2(tra_dim,multiplue),
        )
        self.sig_feat =Transient_TDown_3(dim,dim)
        self.feat=FeatureNet2D_xyz(in_ch=dim,dim=128)
        self.fuse_hyper_2_vol=AxisAttentionHybrid(dim,dim)
        self.scaling=nn.Sequential(
            nn.Conv3d(dim,dim,1,1,0),
        )
        self.fuse_volume = nn.Sequential(
            nn.ReplicationPad3d(1),
            nn.Conv3d(dim*2,dim,3,1,0),
            nn.LeakyReLU(0.2),
            nn.Conv3d(dim,tra_dim,1,1,0),
        )
        self.xy_inten=nn.Sequential(
                                    nn.Conv2d(dim,dim,1,1,0),
                                    nn.ReplicationPad2d(1),
                                    nn.Conv2d(dim,tra_dim,3,1,0),
                                    nn.LeakyReLU(0.2),
                                    )
        self.xy_fuse =  FuseXY(tra_dim)
            
        self.project = VisbleNet()
        self.inten_refine = Rendering(tra_dim*2, out_channels=1,factor=1)
        self.dep_refine = Rendering(tra_dim*2, out_channels=1, isdep=True,factor=1)
        self.pos_x = nn.Parameter(torch.randn(1,dim,128,1,1) * 0.01)
        self.pos_y = nn.Parameter(torch.randn(1,dim,1,128,1) * 0.01)
        self.pos_z = nn.Parameter(torch.randn(1,dim,1,1,128) * 0.01)
        
    def normalize(self, data_bxcxdxhxw):   #  min max scaling
        b, c, d, h, w = data_bxcxdxhxw.shape
        data_bxcxk = data_bxcxdxhxw.reshape(b, c, -1)
        data_min = data_bxcxk.min(2, keepdim=True)[0]
        data_zmean = data_bxcxk - data_min
        # most are 0
        data_max = data_zmean.max(2, keepdim=True)[0].clamp(min=1e-15)
        data_norm = data_zmean / (data_max)
        return data_norm.view(b, c, d, h, w)
    def forward(self, x):
        x= self.normalize(x)
        x = self.trafeature(x)
        x=self.makevol(x)
        x = self.volfeature(x)
        x = self.sig_feat(x)
        B, _, D, W, H = x.shape
        pos_x = x + self.pos_x + self.pos_y + self.pos_z
        proj_xy = pos_x.max(dim=2)[0] 
        proj_xz = pos_x.max(dim=3)[0] 
        proj_yz = pos_x.max(dim=4)[0] 
        
        feat_xy, feat_xz, feat_yz = self.feat(proj_xy, proj_xz,proj_yz)
        xy=self.xy_inten(feat_xy)
        feat_xy = feat_xy.unsqueeze(2).expand(-1, -1, D, -1, -1)
        feat_xz = feat_xz.unsqueeze(3).expand(-1, -1, -1, W, -1)
        feat_yz = feat_yz.unsqueeze(4).expand(-1, -1, -1, -1, H)
        fused= self.fuse_hyper_2_vol(feat_xy,feat_xz,feat_yz)
        x=self.scaling(x)
        x=torch.cat((fused,x),dim=1)
        x = self.fuse_volume(x)
        raw = self.project(x)
        raw_inten = self.xy_fuse(raw[:, :self.tra_dim], xy)
        raw = torch.cat([raw_inten, raw[:, self.tra_dim:]], dim=1)
        img = self.inten_refine(raw)
        dep = self.dep_refine(raw.detach()) 
        return img,dep, x

