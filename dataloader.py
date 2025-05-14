from torch.utils.data import Dataset 
from torchvision import transforms 
import cv2 
import numpy as np 
import torch  
import os 





def getFilesName(path):
	dirs = os.listdir(path)
	files_name = []   
	for dir_ in dirs:
		imgs_name = [img.rpartition('.')[0] for img in os.listdir(path+dir_+'/imL')]
		for img_name in imgs_name:
			files_name.append([dir_, img_name])

	return files_name 





class Dataset_(Dataset):
	def __init__(self, path, files_name, disp_map=True):
		super().__init__() 
		self.path = path 
		self.files_name = files_name
		self.disp_map = disp_map
		self.transform = transforms.Compose([transforms.ToTensor(),
											 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

	def __len__(self):
		return len(self.files_name) 

	def __getitem__(self, idx):
		dir_, img_name = self.files_name[idx] 
		imgL = cv2.imread(self.path+dir_+'/imL/'+img_name+'.png')
		imgR = cv2.imread(self.path+dir_+'/imR/'+img_name+'.png')
		if (imgL is None) or (imgR is None):
			print(img_name)
		imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
		imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB)
		if self.disp_map:  
			dispL = cv2.imread(self.path+dir_+'/dispL_occ_pfm/'+img_name+'.pfm', cv2.IMREAD_UNCHANGED)
			dispR = cv2.imread(self.path+dir_+'/dispR_occ_pfm/'+img_name+'.pfm', cv2.IMREAD_UNCHANGED)

		imgL = self.transform(imgL)
		imgR = self.transform(imgR) 

		if self.disp_map:
			dispL = torch.from_numpy(dispL).unsqueeze(0)   
			dispR = torch.from_numpy(dispR).unsqueeze(0)
			return imgL, imgR, dispL, dispR
		else:
			return imgL, imgR





def generateRandomMask(image_size, patch_size, mask_ratio):
    num_patches_h = image_size[0]//patch_size
    num_patches_w = image_size[1]//patch_size
    total_patches = num_patches_h*num_patches_w
    num_masked_patches = int(total_patches*mask_ratio)
    mask = torch.ones(total_patches, dtype=torch.float32)
    mask[:num_masked_patches] = 0.0  
    mask = mask[torch.randperm(total_patches)]
    mask = mask.view(1, 1, num_patches_h, num_patches_w)
    mask_resize = torch.nn.functional.interpolate(mask, scale_factor=patch_size)
    return mask_resize, mask 





def tensor2img(tensor):
	img = tensor.permute(1, 2, 0).numpy()
	img *= np.array([[0.229, 0.224, 0.225]])
	img += np.array([0.485, 0.456, 0.406]) 
	img *= 255.0 
	img[img>255.0]=255.0
	img[img<0.0]=0.0
	img = img.astype('uint8') 
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
	return img 