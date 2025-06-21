import torch  
import torch.nn as nn 
import dataloader as dtld 
import warnings
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.optim import AdamW 
from model import ModelDisp, UpsampleBlock
from metrics import epe_bad_x
from modules import LaplacianLossBounded2
from torch.optim import lr_scheduler

warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")



bsize = 15

path = '/media/user/Новый том/unrealstereo_10k_clean/'

files_name = dtld.getFilesName(path)

train_files_name, test_files_name = train_test_split(files_name, test_size=0.2, random_state=666) 

train_dataset = dtld.Dataset_(path, train_files_name[:5000]) 
train_dataloader = DataLoader(train_dataset, batch_size=bsize, shuffle=True, num_workers=10)  

test_dataset = dtld.Dataset_(path, test_files_name[:500])  
test_dataloader = DataLoader(test_dataset, batch_size=bsize, num_workers=10)  

device = 'cuda'

model = ModelDisp()

load_model = torch.load('models/model_ol.pt', map_location='cpu')

model.encoder.load_state_dict(load_model.encoder.state_dict())
model.decoder.disp_attention.load_state_dict(load_model.decoder.disp_attention.state_dict())
model.decoder.swin_blocks.load_state_dict(load_model.decoder.swin_blocks.state_dict())

model.encoder.requires_grad_(False)
model.decoder.disp_attention.requires_grad_(False)
model.decoder.swin_blocks.requires_grad_(False)
model.to(device)

initial_lr = 0.001
optim = AdamW(model.parameters(), lr=initial_lr)

warmup_epochs = 5
constant_epochs = 10
decay_epochs = 20
min_lr = 0.0001

warmup_scheduler = lr_scheduler.LinearLR(optim, start_factor=0.02, end_factor=1.0, total_iters=warmup_epochs)
constant_scheduler = lr_scheduler.ConstantLR(optim, factor=1.0, total_iters=constant_epochs)
decay_scheduler = lr_scheduler.LinearLR(optim, start_factor=1.0, end_factor=min_lr/initial_lr, total_iters=decay_epochs)

scheduler = lr_scheduler.SequentialLR(optim, schedulers=[warmup_scheduler, constant_scheduler, decay_scheduler],
                                      milestones=[warmup_epochs, warmup_epochs + constant_epochs])

#loss_fn = torch.nn.MSELoss()
loss_fn = LaplacianLossBounded2()

for ep_num in range(warmup_epochs+constant_epochs+decay_epochs+100):
	if ep_num == 5:
		model.decoder.swin_blocks.requires_grad_(True)
		model.decoder.disp_attention.requires_grad_(True)
		 

	train_loss = 0 
	model.train()  
	for imgL, imgR, dispL, dispR in train_dataloader:
		imgL = imgL.to(device) 
		imgR = imgR.to(device) 
		dispL = dispL.to(device) 
		preds = model(imgL, imgR)
		disp_pred, conf = torch.chunk(preds, chunks=2, dim=1)
		loss = loss_fn(disp_pred, dispL, conf) 
		loss.backward()
		optim.step()
		optim.zero_grad() 
		train_loss += loss.item() 

	epe, bad = epe_bad_x(model, test_dataloader, 3, device) 

	scheduler.step()
	print(ep_num, train_loss, epe, bad)
	torch.save(model, 'models/model_disp_ol_5k.pt')

print('stop')  