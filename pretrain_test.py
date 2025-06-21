import torch  
import dataloader as dtld 
from dataloader import generateRandomMask, tensor2img
from torch.utils.data import DataLoader
from model import Model 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt  
import numpy as np 
import cv2 





bsize = 1

path = '/media/user/Новый том/stereo_imgs/'
#path = '/media/user/Новый том/unrealstereo_10k_clean/'

files_name = dtld.getFilesName(path)
train_files_name, test_files_name = train_test_split(files_name, test_size=0.2, random_state=666)  

test_dataset = dtld.Dataset_(path, test_files_name, disp_map=False)  
test_dataloader = DataLoader(test_dataset, batch_size=bsize, num_workers=1)



device = 'cuda'
#model = torch.load('models/model_ol.pt', map_location='cpu').to(device).eval()
model = Model()
weights = torch.load('models/model_ol.pt', map_location='cpu').state_dict()
model.load_state_dict(weights) 
model.to(device).eval()


for i, imgs in enumerate(test_dataloader):
	imgL, imgR = imgs
	imgL = imgL.to(device)
	imgR = imgR.to(device)
	mask_resize, _ = generateRandomMask((480, 640), 16, 0.9)
	mask_resize = mask_resize.to(device)

	pred = model(imgL*mask_resize, imgR, True) 

	img_pred = tensor2img(pred[0].cpu())
	imgL_real = tensor2img(imgL[0].cpu())  
	imgR_real = tensor2img(imgR[0].cpu())
	img_mask = tensor2img((imgL*mask_resize)[0].cpu())

	for x in range(80, 640, 80):
		img_pred = cv2.line(img_pred, (x, 0), (x, 480), (0, 0, 255), 1)
		imgL_real = cv2.line(imgL_real, (x, 0), (x, 480), (0, 0, 255), 1)
		imgR_real = cv2.line(imgR_real, (x, 0), (x, 480), (0, 0, 255), 1)

	top = np.concatenate([img_mask, imgR_real], axis=1)
	down = np.concatenate([img_pred, imgL_real], axis=1)
	out = np.concatenate([top, down], axis=0) 


	cv2.imshow('test', out)
	cv2.waitKey(0)
	#cv2.imwrite('images/'+str(i)+'.png', out) 