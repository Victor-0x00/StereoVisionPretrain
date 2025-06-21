import torch
import torch.nn as nn
import torch.nn.functional as F
from swin_transformer_v2 import SwinTransformerBlock
from modules import DispAttentionBlock, UpsampleBlock
from ultralytics.nn.modules.block import C3k2
from ultralytics.nn.modules.conv import Conv
from time import time







class Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = nn.Sequential(Conv(3, 64, 3, 2, p=1), Conv(64, 128, 3, 2, p=1), C3k2(128, 256, c3k=True, e=0.25),
                                   Conv(256, 256, 3, 2, p=1), C3k2(256, 512, c3k=True, e=0.25),
                                   Conv(512, 512, 3, 2, p=1), C3k2(512, 512, c3k=True, e=0.5)).requires_grad_(False)

        self.swin_blocks = nn.Sequential(SwinTransformerBlock(512, (30, 40), 8, window_size=10, shift_size=0),
                                         SwinTransformerBlock(512, (30, 40), 8, window_size=10, shift_size=5),
                                         SwinTransformerBlock(512, (30, 40), 8, window_size=10, shift_size=0),
                                         SwinTransformerBlock(512, (30, 40), 8, window_size=10, shift_size=5))
                                        
        if pretrained:
            state_dict = torch.load('models/yolo11m.pt')['model'].state_dict()
            self.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        feature_maps = [x] 
        for m in self.model:
            feature_maps.append(m(feature_maps[-1]))
        #x = self.model(x)
        x = feature_maps[-1]  
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H*W, C)  
        x = self.swin_blocks(x) 
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x, feature_maps[-4] 





class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.x_pos = torch.arange(40, dtype=torch.long)
        self.disp_attention = DispAttentionBlock(8, 512, 40)
        self.swin_blocks = nn.Sequential(SwinTransformerBlock(512, (30, 40), 8, window_size=10, shift_size=0, drop=0.2),
                                          SwinTransformerBlock(512, (30, 40), 8, window_size=10, shift_size=5, drop=0.2),
                                          SwinTransformerBlock(512, (30, 40), 8, window_size=10, shift_size=0, drop=0.2),
                                          SwinTransformerBlock(512, (30, 40), 8, window_size=10, shift_size=5, drop=0.2))
                                          
        self.upsample_blocks = nn.Sequential(UpsampleBlock(512, 256), UpsampleBlock(256, 128), 
                                              UpsampleBlock(128, 64), UpsampleBlock(64, 32))

        self.out_lay = nn.Conv2d(32, 3, 1)

    def forward(self, left, right):
        B, C, H, W = left.shape
        
        out = self.disp_attention(left, right).view(B, H*W, C)
        out = self.swin_blocks(out)
        out = out.view(B, H, W, C).permute(0, 3, 1, 2)
        out = self.upsample_blocks(out) 
        out = self.out_lay(out)
        return out 





class DecoderDisp(nn.Module):
    def __init__(self):
        super().__init__()
        self.x_pos = torch.arange(40, dtype=torch.long)
        self.disp_attention = DispAttentionBlock(8, 512, 40)
        self.swin_blocks = nn.Sequential(SwinTransformerBlock(512, (30, 40), 8, window_size=10, shift_size=0, drop=0.2),
                                          SwinTransformerBlock(512, (30, 40), 8, window_size=10, shift_size=5, drop=0.2),
                                          SwinTransformerBlock(512, (30, 40), 8, window_size=10, shift_size=0, drop=0.2),
                                          SwinTransformerBlock(512, (30, 40), 8, window_size=10, shift_size=5, drop=0.2))
                                          
        self.upsample_blocks = nn.Sequential(UpsampleBlock(512, 256), UpsampleBlock(512, 128), 
                                              UpsampleBlock(128, 64), UpsampleBlock(64, 32))

        self.out_lay = nn.Conv2d(32, 2, 1)

    def forward(self, left, left_high_res, right):
        B, C, H, W = left.shape
        
        out = self.disp_attention(left, right).view(B, H*W, C)
        out = self.swin_blocks(out)
        out = out.view(B, H, W, C).permute(0, 3, 1, 2)
        out = self.upsample_blocks[0](out)
        out = torch.cat([out, left_high_res], dim=1)
        for up in self.upsample_blocks[1:]:
            out = up(out)  
        out = self.out_lay(out)
        return out





class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, left, right, only_left=False): 
        left, _ = self.encoder(left)
        if only_left:
            right = torch.zeros_like(left)
        else:
            right, _ = self.encoder(right) 
        out = self.decoder(left, right)
        return out 





class ModelDisp(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = DecoderDisp()

    def forward(self, left, right): 
        left, left_high_res = self.encoder(left)
        right, _ = self.encoder(right) 
        out = self.decoder(left, left_high_res, right)
        return out





if __name__ == '__main__':
    model = ModelDisp().to('cuda').eval()
    for i in range(10):
        img1 = torch.randn(1, 3, 480, 640, dtype=torch.float32).to('cuda')
        img2 = torch.randn(1, 3, 480, 640, dtype=torch.float32).to('cuda')
        t = time()
        out = model(img1, img2).cpu()
        print(time()-t) 

