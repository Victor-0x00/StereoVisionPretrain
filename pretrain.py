import torch  
import dataloader as dtld 
from dataloader import generateRandomMask
from torch.utils.data import DataLoader
from model import Model 
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from grokfast import gradfilter_ma, gradfilter_ema




#path = '/media/user/Новый том/stereo_imgs/'
path = '/home/stereo_imgs/'

bsize = int(input('batch size = '))
num_ep = int(input('number of epochs = '))
only_left = True if input('only left = ') == 'True' else False 

files_name = dtld.getFilesName(path)
train_files_name, test_files_name = train_test_split(files_name, test_size=0.2, random_state=666) 

train_dataset = dtld.Dataset_(path, train_files_name, disp_map=False) 
train_dataloader = DataLoader(train_dataset, batch_size=bsize, shuffle=True, num_workers=2)  

test_dataset = dtld.Dataset_(path, test_files_name, disp_map=False)  
test_dataloader = DataLoader(test_dataset, batch_size=bsize, num_workers=2)





device = 'cuda'
#model = torch.load('model.pt')
model = Model().to(device) 
model.train()  

optim = AdamW(model.parameters(), lr=0.0001) 
loss_fn = torch.nn.MSELoss() 

val_loss_prev = 1e8
grads = None

for ep in range(1, num_ep+1):
	train_loss = 0 
	val_loss = 0 

	for imgL, imgR in train_dataloader:
		imgL = imgL.to(device) 
		imgR = imgR.to(device)
		mask_resize, _ = generateRandomMask((480, 640), 16, 0.9)
		mask_resize = mask_resize.to(device)
		inv_mask = torch.ones_like(mask_resize)-mask_resize  
		pred = model(imgL*mask_resize, imgR, only_left)  
		loss = loss_fn(pred*inv_mask, imgL*inv_mask) 
		loss.backward()
		grads = gradfilter_ema(model, grads=grads)
		optim.step()
		optim.zero_grad() 
		train_loss += loss.item()

	for imgL, imgR in test_dataloader:
		imgL = imgL.to(device) 
		imgR = imgR.to(device)
		mask_resize, _ = generateRandomMask((480, 640), 16, 0.9)
		mask_resize = mask_resize.to(device)
		inv_mask = torch.ones_like(mask_resize)-mask_resize
		with torch.no_grad():
			pred = model(imgL*mask_resize, imgR, only_left) 
		loss = loss_fn(pred*inv_mask, imgL*inv_mask) 
		val_loss += loss.item()

	print(ep, train_loss, val_loss)  
	with open('models/train_log.txt', 'a') as f:
		f.write(f'{ep} {train_loss} {val_loss} {only_left}\n')
	
	if val_loss <= val_loss_prev: 
		torch.save(model, 'models/best_model.pt')
		val_loss_prev = val_loss 
	else:
		torch.save(model, 'models/model.pt')

  