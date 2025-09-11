from dataloader import DatasetObjects2 
from torch.utils.data import DataLoader
from model import ModelDetection
from detection_loss import loss_fn, disparity_metric
from sklearn.model_selection import train_test_split
from torch.optim import AdamW 
from torch import nn 
import torch.nn.functional as F  
import torch
import os 

    



path = 'path to dataset'
files_name = os.listdir(path+'/images')
files_name = [f.rpartition('.')[0] for f in files_name]
train_files_name, valid_files_name = train_test_split(files_name, test_size=0.2, random_state=666)


train_dataset = DatasetObjects2(path, train_files_name, augmentation=True)
train_dataloader = DataLoader(train_dataset, batch_size=32, num_workers=10, shuffle=True)  

valid_dataset = DatasetObjects2(path, valid_files_name)
valid_dataloader = DataLoader(valid_dataset, batch_size=32, num_workers=10)

model = ModelDetection()
#model = torch.load('models/detection_model.pt')

device = 'cuda'

load_model = torch.load('models/model.pt', map_location='cpu')

model.encoder.load_state_dict(load_model.encoder.state_dict())
model.decoder.disp_attention.load_state_dict(load_model.decoder.disp_attention.state_dict())
model.decoder.swin_blocks.load_state_dict(load_model.decoder.swin_blocks.state_dict())

model.encoder.requires_grad_(False)
model.decoder.disp_attention.requires_grad_(True)
model.decoder.swin_blocks.requires_grad_(True)
model.to(device)

optim = AdamW(model.parameters(), lr=0.0001)
  

for ep in range(100):
	ep_loss = 0 
	ep_obj_loss = 0  
	ep_coords_loss = 0  
	for left, right, objectness, coords in train_dataloader:
		left = left.to(device)
		right = right.to(device)
		objectness = objectness.to(device)
		coords = coords.to(device) 

		pred_objectness, pred_coords = model(left, right)
		loss, obj_loss, coords_loss = loss_fn(pred_objectness, pred_coords, objectness, coords)
		loss.backward()
		optim.step()
		optim.zero_grad()

		ep_loss += loss.item() 
		ep_obj_loss += obj_loss.item()
		ep_coords_loss += coords_loss.item() 

	total_size = 0 
	metric = 0
	for left, right, objectness, coords in valid_dataloader:
		B, _, _, _ = left.shape
		left = left.to(device)
		right = right.to(device)
		objectness = objectness.to(device)
		coords = coords.to(device) 
		with torch.no_grad():
			pred_objectness, pred_coords = model(left, right)
		metric += disparity_metric(pred_coords[:, 4:5, :, :], coords[:, 4:5, :, :], objectness).item()  
		total_size += 1
		

	print(ep, 640*metric/total_size)
	torch.save(model, 'models/detection_model.pt')
