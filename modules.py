import torch
import torch.nn as nn
from torch.nn import functional as F 
from time import time
from ultralytics.nn.modules.block import C3k2, SPPF, C2PSA
from ultralytics.nn.modules.conv import Conv, Concat





class RoPE1d(nn.Module):
	def __init__(self, pos_len, emb_size, base=100.0):
		super().__init__()
		
		inv_freq = 1.0/(base**(torch.arange(0, emb_size, 2).float()/emb_size))
		t = torch.arange(pos_len, dtype=inv_freq.dtype)
		freqs = torch.einsum("i,j->ij", t, inv_freq)
		freqs = torch.cat((freqs, freqs), dim=-1)
		self.register_buffer('sin', freqs.sin())
		self.register_buffer('cos', freqs.cos())
		self.emb_size = emb_size

	def forward(self, emb, pos):
		emb_transform = torch.cat([-emb[..., self.emb_size//2:], emb[..., :self.emb_size//2]], dim=-1)
		return emb*self.cos[pos]+emb_transform*self.sin[pos]
		




class RoPE2d(nn.Module):
	def __init__(self, x_pos_len, y_pos_len, emb_size, base=100.0):
		super().__init__()
		self.x_rope = RoPE1d(x_pos_len, emb_size//2, base=base)
		self.y_rope = RoPE1d(y_pos_len, emb_size//2, base=base)

	def forward(self, emb, x_pos, y_pos):
		x_emb, y_emb = emb.chunk(2, dim=-1)
		x_emb_rot, y_emb_rot = self.x_rope(x_emb, x_pos), self.y_rope(y_emb, y_pos)
		return torch.cat([x_emb_rot, y_emb_rot], dim=-1)





class MultiHeadAttention(nn.Module):
	def __init__(self, head_num, emb_size, x_size, y_size, dim_2d):
		super().__init__()
		self.head_num = head_num
		self.emb_dim = emb_size
		self.head_size = emb_size//head_num
		self.scale = self.head_size**-0.5
		self.dim_2d = dim_2d

		self.q = nn.Linear(emb_size, emb_size, bias=False)
		self.k = nn.Linear(emb_size, emb_size, bias=False)
		self.v = nn.Linear(emb_size, emb_size, bias=False)
		self.proj = nn.Linear(emb_size, emb_size)

		if dim_2d:
			self.rope = RoPE2d(x_size, y_size, self.head_size)
		else:
			self.rope = RoPE1d(x_size, self.head_size)

	def forward(self, emb, pos):
		B, L, C = emb.shape

		q = self.q(emb).view(B, L, self.head_num, self.head_size).transpose(1, 2)
		k = self.k(emb).view(B, L, self.head_num, self.head_size).transpose(1, 2)
		v = self.v(emb).view(B, L, self.head_num, self.head_size).transpose(1, 2)

		if self.dim_2d:
			x_pos, y_pos = pos[:, 0], pos[:, 1]
			k = self.rope(k, x_pos, y_pos)
			q = self.rope(q, x_pos, y_pos)
		else:
			k = self.rope(k, pos)
			q = self.rope(q, pos) 

		att = (q@k.transpose(-2, -1))*self.scale
		att = att.softmax(dim=-1) 

		out = (att@v).transpose(1, 2).reshape(B, L, C)
		out = self.proj(out)
		return out 





class TransformerBlock(nn.Module):
	def __init__(self, emb_size, head_num, x_size, y_size, dim_2d=True):
		super().__init__()
		self.attention = MultiHeadAttention(head_num, emb_size, x_size, y_size, dim_2d)
		self.layer_norm1 = nn.LayerNorm(emb_size)
		self.layer_norm2 = nn.LayerNorm(emb_size)
		self.ffn = nn.Sequential(nn.Linear(emb_size, 2*emb_size), nn.GELU(), nn.Dropout(0.1), nn.Linear(2*emb_size, emb_size))

	def forward(self, emb, pos):
		out = self.layer_norm1(self.attention(emb, pos)+emb)
		out = self.layer_norm2(self.ffn(out)+out) 
		return out





class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(1, num_channels, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, unbiased=False, keepdim=True)
        x_norm = (x-mean)/torch.sqrt(var+self.eps)
        if self.affine:
            x_norm = self.gamma*x_norm+self.beta
        
        return x_norm

 



class DispAttentionBlock(nn.Module):
	def __init__(self, head_num, emb_size, W=40):
		super().__init__()
		self.head_size = emb_size//head_num
		self.head_num = head_num
		self.scale = self.head_size**-0.5 
		self.pos = torch.arange(W, dtype=torch.long)
		self.q = nn.Conv2d(emb_size, emb_size, 1, groups=head_num, bias=False)
		self.k = nn.Conv2d(emb_size, emb_size, 1, groups=head_num, bias=False)
		self.v = nn.Conv2d(emb_size, emb_size, 1, groups=head_num, bias=False)
		self.rope = RoPE1d(W, self.head_size)
		self.proj = nn.Linear(emb_size, emb_size)
		self.norm1 = nn.LayerNorm(emb_size)
		self.norm2 = nn.LayerNorm(emb_size)
		self.ffn = nn.Sequential(nn.Linear(emb_size, 4*emb_size), nn.GELU(), nn.Linear(4*emb_size, emb_size))

	def forward(self, left, right):
		B, C, H, W = left.shape
	
		q = self.q(left).permute(0, 2, 3, 1).view(B, H, W, self.head_num, self.head_size).transpose(2, 3)
		k = self.k(right).permute(0, 2, 3, 1).view(B, H, W, self.head_num, self.head_size).transpose(2, 3)  
		v = self.v(right).permute(0, 2, 3, 1).view(B, H, W, self.head_num, self.head_size).transpose(2, 3) 

		q = self.rope(q, self.pos)
		k = self.rope(k, self.pos)

		Att = self.softpick((q@k.transpose(-1, -2))*self.scale)
		out = (Att@v).transpose(2, 3).reshape(B, H, W, C)
		
		out = self.norm1(self.proj(out)+left.permute(0, 2, 3, 1))
		out = self.norm2(self.ffn(out)+out)
		return out

	def softpick(self, x, e=1e-6):
		m = x.max(-1).values
		m = m.unsqueeze(-1)
		exp = torch.exp(x-m)-torch.exp(-m)
		out = F.relu(exp)/(exp.abs().sum(-1, keepdim=True)+e)
		return out 





class DispAttentionBlock2(nn.Module):
	def __init__(self, head_num, emb_size, W):
		super().__init__()
		self.head_size = emb_size//head_num
		self.head_num = head_num
		#self.scale = self.head_size**-0.5 
		self.pos = torch.arange(W, dtype=torch.long)

		self.q = nn.Conv2d(emb_size, emb_size, 1, groups=head_num, bias=False)
		self.k = nn.Conv2d(emb_size, emb_size, 1, groups=head_num, bias=False)
		self.v = nn.Conv2d(emb_size, emb_size, 1, groups=head_num, bias=False)

		self.rope = RoPE1d(W, self.head_size)
		self.proj = nn.Linear(emb_size, emb_size)
		self.norm1 = nn.LayerNorm(emb_size)
		self.norm2 = nn.LayerNorm(emb_size)
		self.ffn = nn.Sequential(nn.Linear(emb_size, 2*emb_size), nn.GELU(), nn.Dropout(0.0), nn.Linear(2*emb_size, emb_size))

	def forward(self, left, right):
		B, C, H, W = left.shape 

		q = self.q(left).permute(0, 2, 3, 1).view(B, H, W, self.head_num, self.head_size).transpose(2, 3).contiguous()
		k = self.k(right).permute(0, 2, 3, 1).view(B, H, W, self.head_num, self.head_size).transpose(2, 3).contiguous()
		v = self.v(right).permute(0, 2, 3, 1).view(B, H, W, self.head_num, self.head_size).transpose(2, 3).contiguous()
		
		q = self.rope(q, self.pos) 
		k = self.rope(k, self.pos)

		out = F.scaled_dot_product_attention(q, k, v, is_causal=False)

		out = self.proj(out.transpose(2, 3).reshape(B, H, W, C).contiguous())
		out = self.norm1(out+left.permute(0, 2, 3, 1).contiguous())
		out = self.norm2(self.ffn(out)+out)

		return out





class DispAttentionBlock3(nn.Module):
	def __init__(self, head_num, emb_size, W):
		super().__init__()
		self.head_num = head_num 
		self.head_size = emb_size//head_num
		self.pos = torch.arange(W, dtype=torch.long)
		self.causal_mask = torch.triu(torch.ones(W, W), diagonal=1).bool()

		self.att = nn.MultiheadAttention(emb_size, head_num, batch_first=True)
		self.rope = RoPE1d(W, self.head_size)
		self.norm1 = nn.LayerNorm(emb_size)
		self.norm2 = nn.LayerNorm(emb_size)
		self.ffn = nn.Sequential(nn.Linear(emb_size, 2*emb_size), nn.GELU(), nn.Dropout(0.1), nn.Linear(2*emb_size, emb_size))

	def forward(self, left, right):
		B, C, H, W = left.shape 

		q = left.permute(0, 2, 3, 1).reshape(B*H, W, self.head_num, self.head_size).transpose(1, 2).contiguous()
		k = right.permute(0, 2, 3, 1).reshape(B*H, W, self.head_num, self.head_size).transpose(1, 2).contiguous()
		v = right.permute(0, 2, 3, 1).reshape(B*H, W, C).contiguous()

		q = self.rope(q, self.pos).transpose(1, 2).reshape(B*H, W, C).contiguous()
		k = self.rope(k, self.pos).transpose(1, 2).reshape(B*H, W, C).contiguous()
		
		out, _ = self.att(q, k, v, attn_mask=self.causal_mask, need_weights=False, is_causal=True)
		out = out.reshape(B, H, W, C).contiguous()
		out = self.norm1(out+left.permute(0, 2, 3, 1).contiguous())
		out = self.norm2(self.ffn(out)+out)
		return out





class UpsampleBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, padding=1), nn.BatchNorm2d(out_size), nn.SiLU())
        self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 1, 1), nn.BatchNorm2d(out_size), nn.SiLU())
        #self.proj = nn.Conv2d(in_size, out_size, 1, 1, bias=False)

    def forward(self, inp):
        up_inp = F.interpolate(inp, scale_factor=2)
        out = self.conv1(up_inp)
        out = self.conv2(out)#+self.proj(up_inp)
        return out





class Refine(nn.Module):
	def __init__(self, size):
		super().__init__()
		self.layer = nn.Sequential(Conv(size, size, 3, 2, p=1), C3k2(size, size, c3k=True, e=0.25),
								SPPF(size, size), C2PSA(size, size), nn.Upsample(scale_factor=2, mode='nearest'),
								nn.Conv2d(size, size, 3, 1, padding=1))

	def forward(self, inp):
		return self.layer(inp)+inp 





class ImgTransformerBlock(nn.Module):
	def __init__(self, size):
		super().__init__()
		self.x_pos = torch.arange(40, dtype=torch.long)
		self.y_pos = torch.arange(30, dtype=torch.long)
		self.horisontal = TransformerBlock(512, 8, 40, None, dim_2d=False)
		self.vertical =   TransformerBlock(512, 8, 30, None, dim_2d=False)
		self.proj = nn.Conv2d(2*size, size, 1, bias=False)

	def forward(self, inp):
		B, C, H, W = inp.shape
		hor = self.horisontal(inp.permute(0, 2, 3, 1).reshape(B*H, W, C), self.x_pos).reshape(B, H, W, C).permute(0, 3, 1, 2)
		ver = self.vertical(inp.permute(0, 3, 2, 1).reshape(B*W, H, C), self.y_pos).reshape(B, W, H, C).permute(0, 3, 2, 1)
		out = torch.cat([hor, ver], axis=1)
		out = self.proj(out)
		return out
		 




class CrossAttention(nn.Module):
	def __init__(self, size):
		super().__init__()
		self.q = nn.Linear(size, size) 
		self.k = nn.Linear(size, size) 
		self.v = nn.Linear(size, size)  
		self.layer = nn.Linear(size, size) 
		self.norm1 = nn.LayerNorm(size)


	def forward(self, inp1, inp2):   
		B, C, H, W = inp1.shape  
		features1 = inp1.permute(0, 2, 3, 1)#.reshape(B, H*W, C) 
		features2 = inp2.permute(0, 2, 3, 1)#.reshape(B, H*W, C)

		source = torch.cat([features1, features2], axis=0) 
		target = torch.cat([features2, features1], axis=0)  

		q = self.q(source) 
		k = self.k(target) 
		v = self.v(target)  

		out = F.scaled_dot_product_attention(q, k, v)
		out = self.norm1(self.layer(out))   
		out = out.permute(0, 3, 1, 2)

		out1, out2 = out.chunk(chunks=2, axis=0) 
		return out1+inp1, out2+inp2





if __name__ == '__main__': 
	attn = DispAttentionBlock(16, 512, 40)
	for i in range(10):
		img1 = torch.randn(1, 512, 30, 40, dtype=torch.float32)#.to('cuda')
		img2 = torch.randn(1, 512, 30, 40, dtype=torch.float32)#.to('cuda')
		#targ = torch.randn(8, 30, 40, 768, dtype=torch.float32).to('cuda')

		torch.cuda.synchronize()
		t = time()
		attn(img1, img2)
		torch.cuda.synchronize()
		print(time()-t)
