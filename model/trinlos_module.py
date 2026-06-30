import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from typing import Optional
from timm.models.layers import DropPath
import torch.nn.init as init
from typing import Union, Tuple
from .elct import Elct


class ConvNeXtBlock2D(nn.Module):
    def __init__(self, dim, drop_path=0.,layer_scale_init_value=-1,norm=True):
        super().__init__()
        self.pad = nn.ReplicationPad2d(3)
        
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=0, groups=dim)
        self.use_norm=norm
        if norm:
            self.norm   = nn.LayerNorm(dim, eps=1e-6)
        self.use_norm=norm
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act    = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma   = nn.Parameter(layer_scale_init_value * torch.ones(dim)) \
                       if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.pad(x)
        x = self.dwconv(x)
        x = x.permute(0,2,3,1)
        if self.use_norm:
            x = self.norm(x)
        x = self.pwconv1(x); x = self.act(x); x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0,3,1,2)
        x = shortcut + self.drop_path(x)
        return x

class ResConvNeXtBlock2D(nn.Module):
    def __init__(self, dim,drop_path=0.0):
        super().__init__()
        
        self.conv = nn.Sequential(
                                nn.ReplicationPad2d(1),
                                nn.Conv2d(dim,dim,3,1,0),
                                 )
        self.out = nn.Sequential(
                                nn.ReplicationPad2d(1),
                                nn.Conv2d(dim,dim,3,1,0),
                         )
        self.block1 = ConvNeXtBlock2D(dim)
        self.block2 = ConvNeXtBlock2D(dim)
        self.act = nn.LeakyReLU(0.2)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
    def forward(self, x):
        identity = x
        x = self.conv(x)
        out = self.block1(x)
        out = self.block2(out)
        out=self.out(out)
        out = self.drop_path(out)  + identity
        # return self.act(out)
        return out

def get_valid_num_groups(num_channels: int, desired_groups: int = 8) -> int:
    max_groups = max(1, num_channels // 4)
    
    if desired_groups >= num_channels:
        G0 = num_channels - 1
    else:
        G0 = desired_groups
    
    G = min(G0, max_groups)
    
    while G > 1:
        if num_channels % G == 0:
            return G
        G -= 1
    
    return 1

def replicate_conv2d(
    in_channels: int,
    out_channels: int,
    kernel_size: Union[int, Tuple[int, int]] = 3,
    stride:      Union[int, Tuple[int, int]] = 1,
    padding:     Union[int, Tuple[int, int]] = 1,
    dilation:    Union[int, Tuple[int, int]] = 1,
    groups: int = 1,
    bias: bool = True,
) -> nn.Sequential:
    if isinstance(padding, int):
        pad_h = pad_w = padding
    else:
        pad_h, pad_w = padding

    pad_layer  = nn.ReplicationPad2d((pad_w, pad_w, pad_h, pad_h)) if (pad_h or pad_w) else nn.Identity()
    conv_layer = nn.Conv2d(
        in_channels, out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=0,
        dilation=dilation,
        groups=groups,
        bias=bias,
    )
    return nn.Sequential(pad_layer, conv_layer)

class Conv_block2d(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int,
        kernel_size=(3, 3), stride=1, padding=1,
        drop_rate: float = 0., desired_groups: int = 8, norm=False
    ):
        super().__init__()
        self.conv = replicate_conv2d(
            in_channels, out_channels,
            kernel_size, stride,
            padding=padding
        )
        G = get_valid_num_groups(out_channels, desired_groups)
        if norm:
            self.gn = nn.GroupNorm(num_groups=G, num_channels=out_channels)
        else:
            self.gn =nn.Identity()
        self.relu =  nn.LeakyReLU(negative_slope=0.2, inplace=False)
        self.drop = nn.Dropout2d(drop_rate) if drop_rate > 0.0 else nn.Identity()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)
        return self.drop(x)

class Downsample2d(nn.Module):
    def __init__(self, in_channels,out_channels, kernel=(3,3), stride=(2,2), drop_rate=0.3):
        super().__init__()

        self.pad = nn.ReplicationPad2d(1)
        self.conv = nn.Conv2d(in_channels,out_channels,kernel,stride)
        
        # self.conv = Conv_block2d(in_channels, out_channels, kernel, stride, drop_rate=drop_rate)
        self.relu =  nn.LeakyReLU(negative_slope=0.2, inplace=False)
    def forward(self, x):
        x=self.pad(x)
        return self.relu(self.conv(x))

class UpBilinear2d(nn.Module):

    def __init__(self, in_channels: int,out_channels: int,  drop_rate: float = 0.1,kernel_size=3,norm=False):
        super().__init__()

        self.conv = nn.Sequential(
                        nn.ReplicationPad2d(1),
                        nn.Conv2d(in_channels,out_channels,3,1,0)
        )
        if norm:
            self.gn   = nn.GroupNorm(get_valid_num_groups(in_channels, 4), in_channels)
        else:
            self.gn= nn.Identity()
        self.act  = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.conv(x)
        x = self.act(self.gn(x))
        
        return x

class BiasFreeLayerNorm2d(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, c, 1, 1))  # γ

    def forward(self, x):
        mu  = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        xhat = (x - mu) / torch.sqrt(var + 1e-5)
        return xhat * self.weight

class WithBiasLayerNorm2d(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, c, 1, 1))  # γ
        self.bias   = nn.Parameter(torch.zeros(1, c, 1, 1)) # β

    def forward(self, x):
        mu  = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        xhat = (x - mu) / torch.sqrt(var + 1e-5)
        return xhat * self.weight + self.bias


# -------- MDTA: Multi-DConv Head Transposed Self-Attention --------

class MDTA(nn.Module):
    def __init__(self, dim: int, num_heads: int, bias: bool = False):
        super().__init__()
        assert dim % num_heads == 0
        self.h = num_heads
        self.c_per_head = dim // num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv    = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dw = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1,
                                groups=dim * 3, bias=bias)
        self.proj   = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def _reshape(self, x):  # (B,C,H,W) -> (B,h,c,N)
        B, C, H, W = x.shape
        return x.view(B, self.h, C // self.h, H * W)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv_dw(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = self._reshape(q); k = self._reshape(k); v = self._reshape(v)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        # (B,h,c,c)x(B,h,c,N)->(B,h,c,N) -> (B,C,H,W)
        out = torch.matmul(attn, v).view(B, C, H, W)
        return self.proj(out)


# -------- GDFN: Gated-DConv Feed-Forward --------

class GDFN(nn.Module):
    def __init__(self, dim: int, expansion: float = 2.66, bias: bool = False):
        super().__init__()
        hidden = int(dim * expansion)
        self.pw_in  = nn.Conv2d(dim, hidden * 2, kernel_size=1, bias=bias)
        self.dw     = nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3, padding=1,
                                groups=hidden * 2, bias=bias)
        self.pw_out = nn.Conv2d(hidden, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.pw_in(x)
        x1, x2 = self.dw(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.pw_out(x)

class RestormerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, expansion: float = 2.66, bias: bool = False, norm_bias: bool = False,drop_path: float = 0.0):
        super().__init__()
        if norm_bias:
            self.norm1 = WithBiasLayerNorm2d(dim)
            self.norm2 = WithBiasLayerNorm2d(dim)
        else:
            self.norm1 = BiasFreeLayerNorm2d(dim)
            self.norm2 = BiasFreeLayerNorm2d(dim)
            
        self.attn  = MDTA(dim=dim, num_heads=heads, bias=bias)
        self.ffn   = GDFN(dim=dim, expansion=expansion, bias=bias)
        self.drop_path_attn = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.drop_path_ffn = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path_attn(self.attn(self.norm1(x)))
        x = x + self.drop_path_ffn(self.ffn(self.norm2(x)))
        return x

class Global_attn(nn.Module):
    def __init__(self,dim: int, heads: int = 4, expansion: float = 2, bias: bool = False, norm_bias: bool = False,drop_path: float = 0.0):
        super().__init__()
        self.patch_embed = Conv_block2d(dim,dim)
        self.enc = RestormerBlock(dim,heads=heads)
        self.bottle = nn.Sequential(RestormerBlock(dim*2,heads=int(2*heads)),
                                    RestormerBlock(dim*2,heads=int(2*heads)))
        self.down=Downsample2d(dim,dim*2)
        self.up=UpBilinear2d(dim*2,dim)
        self.dec = RestormerBlock(dim,heads=heads)
        self.concat_conv1=Conv_block2d(dim*2,dim)
        self.concat_conv2=Conv_block2d(dim*2,dim)
        self.out = nn.Conv2d(dim,dim,1,1,0)
        # self.scale = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, x):
        res1=x
        x= self.patch_embed(x)
        x=self.enc(x)
        res2=x
        x=self.down(x)
        x=self.bottle(x)
        x=self.up(x)
        x = self.concat_conv1(torch.cat([res2,x],dim=1))
        x=self.dec(x)
        x = self.concat_conv2(torch.cat([res1,x],dim=1))
        return self.out(x)

class ResConv3D(nn.Module):

    def __init__(self, nf0, inplace=False,drop_path=0):
        super(ResConv3D, self).__init__()
        # self.scale=nn.Parameter(torch.full((1, nf0, 1, 1, 1), 0.01))
        
        self.tmp = nn.Sequential(
                
                nn.ReplicationPad3d(1),
                nn.Conv3d(nf0 * 1,
                          nf0 * 1,
                          kernel_size=[3, 3, 3],
                          padding=0,
                          stride=[1, 1, 1],
                          bias=True),
                nn.LeakyReLU(negative_slope=0.2, inplace=inplace),
                # nn.Dropout3d(0.1, inplace),
                nn.ReplicationPad3d(1),
                nn.Conv3d(nf0 * 1,
                          nf0 * 1,
                          kernel_size=[3, 3, 3],
                          padding=0,
                          stride=[1, 1, 1],
                          bias=True),
        )
        self.inplace = inplace
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
    def forward(self, x):
        re = F.leaky_relu(self.drop_path(self.tmp(x)) + x, negative_slope=0.2, inplace=self.inplace)
        return re

class Interpsacle2d(nn.Module):
    
    def __init__(self, factor=2, gain=1, align_corners=False):
        """
            the first upsample method in G_synthesis.
        :param factor:
        :param gain:
        """
        super(Interpsacle2d, self).__init__()
        self.gain = gain
        self.factor = factor
        self.align_corners = align_corners

    def forward(self, x):
        if self.gain != 1:
            x = x * self.gain
        
        x = nn.functional.interpolate(x, scale_factor=self.factor, mode='bilinear', align_corners=self.align_corners)
        
        return x

class ResConv2D(nn.Module):

    def __init__(self, nf0, inplace=False,drop_path=0.):
        super(ResConv2D, self).__init__()
        
        self.tmp = nn.Sequential(
                
                nn.ReplicationPad2d(1),
                nn.Conv2d(nf0 * 1,
                          nf0 * 1,
                          kernel_size=[3, 3],
                          padding=0,
                          stride=[1, 1],
                          bias=True),
                
                nn.LeakyReLU(negative_slope=0.2, inplace=inplace),
                # nn.Dropout3d(0.1, inplace),
                
                nn.ReplicationPad2d(1),
                nn.Conv2d(nf0 * 1,
                          nf0 * 1,
                          kernel_size=[3, 3],
                          padding=0,
                          stride=[1, 1],
                          bias=True),
        )
        
        self.inplace = inplace
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
    def forward(self, x):
        re = F.leaky_relu(self.drop_path(self.tmp(x))+ x, negative_slope=0.2, inplace=self.inplace)
        return re

class FuseXY(nn.Module):
    def __init__(self, C):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(2*C, C, kernel_size=3, stride=1, padding=0),
            ResConv2D(C, inplace=False),
            ResConv2D(C, inplace=False)
        )
    def forward(self, raw, xy):
        x = torch.cat([raw, xy], dim=1)
        out = self.conv(x)
        return  out

class Rendering(nn.Module):
    
    def __init__(self, nf0, out_channels, factor,\
                 norm=nn.InstanceNorm2d, isdep=False):
        super(Rendering, self).__init__()
        
        ######################################
        assert out_channels == 1
        
        weights = np.zeros((1, 2, 1, 1), dtype=np.float32)
        if isdep:
            weights[:, 1:, :, :] = 1.0
        else:
            weights[:, :1, :, :] = 1.0
        tfweights = torch.from_numpy(weights)
        tfweights.requires_grad = True
        self.weights = nn.Parameter(tfweights)
        
        self.resize = Interpsacle2d(factor=factor, gain=1, align_corners=False)
        drop_rate =0.
        add =True
        #######################################
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(nf0 * 1,
                      nf0 * 1,
                      kernel_size=3,
                      padding=0,
                      stride=1,
                      bias=True),
            ResConv2D(nf0 * 1, inplace=False),
            ResConv2D(nf0 * 1, inplace=False),
            
        )
        
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(nf0 * 1 + 1,
                      nf0 * 2,
                      kernel_size=3,
                      padding=0,
                      stride=1,
                      bias=True),
            
            ResConv2D(nf0 * 2, inplace=False),
            ResConv2D(nf0 * 2, inplace=False),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(nf0 * 2,
                      out_channels,
                      kernel_size=3,
                      padding=0,
                      stride=1,
                      bias=True),
        )
        # self.scale=nn.Parameter(torch.tensor([1.]))
    def forward(self, x0):
        
        dim = x0.shape[1] // 2
        x0_im = x0[:, 0:1, :, :]
        x0_dep = x0[:, dim:dim + 1, :, :]
        x0_raw_128 = torch.cat([x0_im, x0_dep], dim=1)
        x0_raw_256 = self.resize(x0_raw_128)
        x0_conv_256 = F.conv2d(x0_raw_256, self.weights, \
                               bias=None, stride=1, padding=0, dilation=1, groups=1)
        
        ###################################
        x1 = self.conv1(x0)
        x1_up = self.resize(x1)
        
        x2 = torch.cat([x0_conv_256, x1_up], dim=1)
        x2 = self.conv2(x2)
        
        re = x0_conv_256 + 1 * x2 
        
        return re

class VisbleNet(nn.Module):
    def __init__(self):
        super(VisbleNet, self).__init__()

    def forward(self, x): 
        inten, idx = torch.max(x,dim=2)
        d = x.size(2)
        depth = (d -1 - idx.float()) / (d - 1)
        out = torch.cat([inten, depth], dim=1)
        return out

class FeatureNet2D_enc(nn.Module):
    def __init__(self, in_ch: int = 32,dim=128):
        super().__init__()
        self.dim_ext = nn.Sequential(
            nn.Conv2d(in_ch, dim,1,1,0),
            nn.LeakyReLU(0.2),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim,3,1,0)
        )
        self.resconvnext =ResConvNeXtBlock2D(dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dim_ext(x)
        x = self.resconvnext(x)
        return x

class FeatureNet2D_bottle(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.restform = Global_attn(dim)
        self.resconvnext =ResConvNeXtBlock2D(dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x=self.restform(x)
        x=self.resconvnext(x)
        return x

class FeatureNet2D_dec(nn.Module):
    def __init__(self, in_ch: int = 32,dim: int = 128):
        super().__init__()
        self.concat_conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim*2, dim,3,1,0)
        )
        self.resconvnext =ResConvNeXtBlock2D(dim)
        self.out = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, in_ch,3,1,0),
            nn.LeakyReLU(0.2)
        )
        
    def forward(self, res: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x, res], dim=1)
        x = self.concat_conv(x)          
        x = self.resconvnext(x)
        x = self.out(x)          
        return x

class FeatureNet2D_enc_bot(nn.Module):
    def __init__(self,  in_ch: int = 32,dim: int = 128):
        super().__init__()
        self.enc=FeatureNet2D_enc(in_ch,dim)
        self.bot=FeatureNet2D_bottle(dim)
    def forward(self, x):
        res=self.enc(x)
        x=self.bot(res)
        return res, x

class FeatureNet2D_bot_cross_xyz(nn.Module):
    def __init__(self,  in_ch: int = 32,dim: int = 128):
        super().__init__()
        self.xy=FeatureNet2D_enc_bot(in_ch,dim)
        self.xz=FeatureNet2D_enc_bot(in_ch,dim)
        self.yz=FeatureNet2D_enc_bot(in_ch,dim)
        num_heads = 4
        self.cross=TriplaneCrossAttnLayer(dim=dim, num_heads=num_heads)
        
    def forward(self, xy,xz,yz):
        res_xy,xy=self.xy(xy)
        res_xz,xz=self.xz(xz)
        res_yz,yz=self.yz(yz)
        xy, xz,yz = self.cross(xy, xz,yz)
        
        return (res_xy,xy),(res_xz,xz),(res_yz,yz)

class FeatureNet2D_dec_xyz(nn.Module):
    def __init__(self, in_ch: int = 32,dim: int = 128):
        super().__init__()
        self.xy=FeatureNet2D_dec(in_ch,dim)
        self.xz=FeatureNet2D_dec(in_ch,dim)
        self.yz=FeatureNet2D_dec(in_ch,dim)
        
    def forward(self, xy_t,xz_t,yz_t):
        res_xy,xy=xy_t
        res_xz,xz=xz_t
        res_yz,yz=yz_t
        xy=self.xy(res_xy,xy)
        xz=self.xz(res_xz,xz)
        yz=self.yz(res_yz,yz)
        return xy,xz,yz

class FeatureNet2D_xyz(nn.Module):
    def __init__(self,  in_ch: int = 32,dim: int = 128):
        super().__init__()
        self.enc=FeatureNet2D_bot_cross_xyz(in_ch,dim)
        self.dec=FeatureNet2D_dec_xyz(in_ch,dim)
    def forward(self,xy,xz,yz):
        xy,xz,yz = self.enc(xy,xz,yz)
        xy,xz,yz = self.dec(xy,xz,yz)
        return xy,xz,yz

def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    causal: bool = False,
    softmax_scale=None,
) -> torch.Tensor:

    out = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=causal,      
    )
    return out

class FlashCrossAttention(nn.Module):

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.dropout = dropout
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        B, L_q, C = q.shape
        H = self.num_heads
        q = rearrange(self.q_proj(q), "b l (h d) -> b h l d", h=H)
        k = rearrange(self.k_proj(k), "b l (h d) -> b h l d", h=H)
        v = rearrange(self.v_proj(v), "b l (h d) -> b h l d", h=H)
        out = flash_attn_func(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            causal=False,
            softmax_scale=None,
        )
        out = rearrange(out, "b h l d -> b l (h d)")
        return self.out_proj(out)

class SharedAxisCrossAttnBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        drop_rate: float = 0.,      # attention dropout
        mlp_ratio: float = 4.0,
        drop_path: float = 0.,
    ):
        super().__init__()
        self.ca = FlashCrossAttention(dim, num_heads, drop_rate)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.norm_ffn = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
            nn.Dropout(drop_rate),
        )
        self.dp_attn = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.dp_mlp = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    @staticmethod
    def _prepare_tokens(feat: torch.Tensor, axis: str) -> tuple[torch.Tensor, int]:
        if axis == "x":
            tok = feat.permute(0, 3, 2, 1).contiguous()
        elif axis == "y":
            H_eq_W = feat.shape[2] == feat.shape[3]
            if H_eq_W:
                tok = feat.permute(0, 2, 3, 1).contiguous()
            else:
                if feat.shape[2] < feat.shape[3]:
                    tok = feat.permute(0, 3, 2, 1).contiguous()
                else:
                    tok = feat.permute(0, 2, 3, 1).contiguous()
        else:
            tok = feat.permute(0, 2, 3, 1).contiguous()
        B, S, L, C = tok.shape
        flat = tok.view(B * S, L, C)
        return flat, S

    def forward(self, F_q: torch.Tensor, F_kv: torch.Tensor, axis: str) -> torch.Tensor:
        assert axis in {"x", "y", "z"}, "axis must be one of {'x','y','z'}"
        q_flat, S = self._prepare_tokens(F_q, axis)
        kv_flat, _ = self._prepare_tokens(F_kv, axis)
        q_n = self.norm_q(q_flat)
        kv_n = self.norm_kv(kv_flat)
        attn_out = self.ca(q_n, kv_n, kv_n)
        x = q_flat + self.dp_attn(attn_out)
        x = x + self.dp_mlp(self.mlp(self.norm_ffn(x)))
        B = F_q.shape[0]
        L_q = x.shape[1]
        C = x.shape[2]
        x = x.view(B, S, L_q, C)
        if axis == "x":
            x = x.permute(0, 3, 2, 1).contiguous()
        elif axis == "y":
            if L_q == F_q.shape[3]:
                x = x.permute(0, 3, 1, 2).contiguous()
            else:
                x = x.permute(0, 3, 2, 1).contiguous()
        else:
            x = x.permute(0, 3, 1, 2).contiguous()
        return x



# ==============================================================================#
#  Layer combining all 6 cross-attention directions
# ------------------------------------------------------------------------------#

class TriplaneCrossAttnLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0., mlp_ratio: float = 4.0):
        super().__init__()
        self.xy_from_yz = SharedAxisCrossAttnBlock(dim, num_heads, dropout, mlp_ratio)
        self.xy_from_xz = SharedAxisCrossAttnBlock(dim, num_heads, dropout, mlp_ratio)
        self.yz_from_xy = SharedAxisCrossAttnBlock(dim, num_heads, dropout, mlp_ratio)
        self.yz_from_xz = SharedAxisCrossAttnBlock(dim, num_heads, dropout, mlp_ratio)
        self.xz_from_xy = SharedAxisCrossAttnBlock(dim, num_heads, dropout, mlp_ratio)
        self.xz_from_yz = SharedAxisCrossAttnBlock(dim, num_heads, dropout, mlp_ratio)

    # ------------------------------------------------------------------
    def forward(
        self,
        F_xy: torch.Tensor,  # (B,C,H,W)
        F_xz: torch.Tensor,  # (B,C,D,W)
        F_yz: torch.Tensor,  # (B,C,D,H)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        F_yz = self.yz_from_xy(F_yz, F_xy, axis="y")  # share *height*
        F_yz = self.yz_from_xz(F_yz, F_xz, axis="z")  # share *depth*

        F_xz = self.xz_from_xy(F_xz, F_xy, axis="x")  # share *width*
        F_xz = self.xz_from_yz(F_xz, F_yz, axis="z")  # share *depth*

        F_xy = self.xy_from_xz(F_xy, F_xz, axis="x")  # share *width*
        F_xy = self.xy_from_yz(F_xy, F_yz, axis="y")  # share *height*

        return F_xy, F_xz, F_yz

class Transient_TDown_2(nn.Module):
    def __init__(self, in_channels,out_channels, ts_stride, norm=nn.InstanceNorm3d):
        super(Transient_TDown_2, self).__init__()
        # assert in_channels == 1
        self.ts_stride = ts_stride
        weights = np.zeros((1, in_channels, 3, 3, 3), dtype=np.float32)
        weights[:, :, 1:, 1:, 1:] = 1.0
        tfweights = torch.from_numpy(weights / np.sum(weights))
        # tfweights = torch.from_numpy(weights)
        tfweights.requires_grad = True
        self.weights = nn.Parameter(tfweights)
        
        ##############################################
        self.conv1 = nn.Sequential(
            # begin, no norm
            nn.ReplicationPad3d(1),
            nn.Conv3d(in_channels,
                      2 * out_channels - 1,
                      kernel_size=3,
                      padding=0,
                      stride=self.ts_stride,
                      bias=True),
            ResConv3D(2 * out_channels - 1, inplace=False),
            ResConv3D(2 * out_channels - 1, inplace=False)
            
        )
    def forward(self, x0):
        x0_conv = F.conv3d(x0, self.weights, \
                           bias=None, stride=self.ts_stride, padding=1, dilation=1, groups=1)
        x1 = self.conv1(x0)

        re = torch.cat([x0_conv, x1], dim=1)
        return re

class MsFeat_2(nn.Module):

    def __init__(self, in_channels=1, out_channels=8):
        super().__init__()
        self.in_chans = in_channels
        self.out_chans = out_channels

        self.conv1 = nn.Sequential(nn.Conv3d(self.in_chans, self.out_chans, 3, stride=(2,1,1), padding=1, dilation=1, bias=True), nn.LeakyReLU(negative_slope=0.1, inplace=True))
        init.kaiming_normal_(self.conv1[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.conv1[0].bias, 0.0)

        self.conv2 = nn.Sequential(nn.Conv3d(self.in_chans, self.out_chans, 3, stride=(2,1,1), padding=2, dilation=2, bias=True), nn.LeakyReLU(negative_slope=0.1, inplace=True))
        init.kaiming_normal_(self.conv2[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.conv2[0].bias, 0.0)

        self.conv3 = nn.Sequential(nn.Conv3d(self.out_chans, self.out_chans, 3, padding=1, dilation=1, bias=True), nn.LeakyReLU(negative_slope=0.1, inplace=True))
        init.kaiming_normal_(self.conv3[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.conv3[0].bias, 0.0)

        self.conv4 = nn.Sequential(nn.Conv3d(self.out_chans, self.out_chans, 3, padding=2, dilation=2, bias=True), nn.LeakyReLU(negative_slope=0.1, inplace=True))
        init.kaiming_normal_(self.conv4[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.conv4[0].bias, 0.0)
        
        self.dimext1 = nn.Sequential(
                        nn.ReplicationPad3d(1),
                        nn.Conv3d(self.in_chans, self.out_chans, 3),
                        nn.LeakyReLU(negative_slope=0.2, inplace=False)
        )
        self.dimext2 = nn.Sequential(
                        nn.ReplicationPad3d(1),
                        nn.Conv3d(self.in_chans, self.out_chans, 3),
                        nn.LeakyReLU(negative_slope=0.2, inplace=False)
            
        )
        self.concat1 = nn.Conv3d(self.out_chans*2,self.out_chans,1,1,0)
        self.concat2 = nn.Conv3d(self.out_chans*2,self.out_chans,1,1,0)
    def forward(self, inputs):
        ds = inputs
        max_pooled = F.max_pool3d(ds, kernel_size=(2,1,1), stride=(2,1,1))
        conv1 = self.conv1(ds)
        conv2 = self.conv2(ds)
        conv1 = self.concat1(torch.cat((conv1,self.dimext1(max_pooled)),dim=1))
        conv2 = self.concat2(torch.cat((conv2,self.dimext2(max_pooled)),dim=1))
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv1)
        return torch.cat((conv1, conv2, conv3, conv4), 1)

class tra2vol(nn.Module):
    def __init__(self, tradim, spatial = 128, crop = 256, bin_len=0.01 *2,
                 material=None, fusion_scale=None, fusion_power=None):
        super().__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if material is not None:
            self.material = material
            self.fusion_scale = fusion_scale
            self.fusion_power = fusion_power
        else:
            self.material = nn.Parameter(torch.tensor([2., 2., 2., 2.], device=device))
            self.fusion_scale = nn.Parameter(torch.tensor([0.25, 0.25, 0.37, 0.5], device=device))
            self.fusion_power = nn.Parameter(torch.tensor([2., 1.5, 2., 2.5], device=device))

        self.recon_lct = Elct(
            fixed_shape=[crop, spatial, spatial],
            bin_len=bin_len,
            material=self.material,
            fusion_scale=self.fusion_scale,
            fk_eps=0.01,
            fusion_power=self.fusion_power
        )

    def normalize(self, data_bxcxdxhxw):
        b, c, d, h, w = data_bxcxdxhxw.shape
        data = data_bxcxdxhxw.reshape(b, c, -1)
        data_min = data.min(2, keepdim=True)[0]
        data_zmean = data - data_min
        data_max = data_zmean.max(2, keepdim=True)[0]
        data_norm = data_zmean / (data_max + 1e-15)
        return data_norm.view(b, c, d, h, w)

    def forward(self, x1: torch.Tensor) -> torch.Tensor:
        x1 = self.recon_lct(x1)
        return self.normalize(x1)


# -------------------------------------------------------------

class Transient_TDown_3(nn.Module):
    def __init__(self, in_channels,out_channels, norm=nn.InstanceNorm3d):
        super(Transient_TDown_3, self).__init__()
        weights = np.zeros((in_channels, in_channels, 3, 3, 3), dtype=np.float32) #  up 14 

        weights[:, :, 1, 1, 1] = 1.0
        tfweights = torch.from_numpy(weights / np.sum(weights))
        # tfweights = torch.from_numpy(weights)

        tfweights.requires_grad = True
        self.weights = nn.Parameter(tfweights)
        
        self.conv1 = nn.Sequential(
            # begin, no norm
            nn.ReplicationPad3d(1),
            nn.Conv3d(in_channels,
                      out_channels,
                      kernel_size=3,
                      padding=0,
                      stride=(1,1,1),
                      bias=True),
            ResConv3D(out_channels, inplace=False),
            ResConv3D(out_channels, inplace=False)
            # resConvNeXtBlock3D_local(out_channels),
            # resConvNeXtBlock3D_local(out_channels)
            
        )
    def forward(self, x0):
        x0_conv = F.conv3d(x0, self.weights, 
                           bias=None, stride=(1,1,1), padding=1, dilation=1, groups=1)
        x1 = self.conv1(x0)
        re = x0_conv + x1
        return re

class AxisAttentionHybrid(nn.Module):
    def __init__(self, c_in, c_out=None, reduction=4):
        super().__init__()
        if c_out is None:
            c_out = c_in
        self.c_out = c_out
        hidden = max(c_in // reduction, 1)

        self.att_conv = nn.Sequential(
            nn.Conv3d(c_in * 3, hidden, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(hidden, 3, kernel_size=1)
        )

        self.mix_conv = nn.Sequential(
            nn.Conv3d(c_in, c_out, kernel_size=1),
            ResConv3D(c_out),
            ResConv3D(c_out)
        )

    def forward(self, feat_xy, feat_xz, feat_yz):
        stacked = torch.cat([feat_xy, feat_xz, feat_yz], dim=1)
        weight = F.softmax(self.att_conv(stacked), dim=1)
        fused_axis = (
            feat_xy * weight[:, 0:1] +
            feat_xz * weight[:, 1:2] +
            feat_yz * weight[:, 2:3]
        )
        return self.mix_conv(fused_axis)
