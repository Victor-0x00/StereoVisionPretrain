from torch.utils.data import Dataset 
from torchvision import transforms 
from PIL import Image 
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





class DatasetObjects(Dataset):
	def __init__(self, path, augmentation=False):
		super().__init__() 
		self.path = path
		self.files_name = os.listdir(path+'/images')
		self.files_name = [f.rpartition('.')[0] for f in self.files_name]
		self.augmentation = augmentation
		if augmentation:
			self.transform = transforms.Compose([
				transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), 
				transforms.RandomGrayscale(p=0.2),
				transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 0.5))], p=0.5),   
				transforms.ToTensor(),
				transforms.Compose([
				transforms.RandomErasing(p=0.7, scale=(0.001, 0.003), ratio=(0.3, 3.3), value='random') for _ in range(15)]),
				transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
		else:
			self.transform = transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

	def __len__(self):
		return len(self.files_name) 

	def __getitem__(self, idx):
		with open(self.path+'/labels/'+self.files_name[idx]+'.txt', 'r') as f:
			labels = f.read()
			labels = labels.split('\n')

		left_labels = []
		right_labels = []  

		for label in labels:
			label = label[1:].split()
			if label != []:
				x, y, w, h = list(map(float, label))
				if x<0.5:
					left_labels.append([x, y, w, h])
				else:
					right_labels.append([x, y])

		left_labels = sorted(left_labels, key = lambda lab: lab[0])
		left_labels = sorted(left_labels, key = lambda lab: lab[1])
		right_labels = sorted(right_labels, key = lambda lab: lab[0])
		right_labels = sorted(right_labels, key = lambda lab: lab[1])

		img = cv2.imread(self.path+'/images/'+self.files_name[idx]+'.jpg')
		objectness = torch.zeros(1, 15, 20, dtype=torch.float32)
		coords = torch.zeros(5, 15, 20, dtype=torch.float32)

		for left, right in zip(left_labels, right_labels):
			xl, yl, w, h = left 
			xr, _ = right
			disp = 2*(xl-(xr-0.5))
			xl *= 40
			yl *= 15  
			xc = int(xl)
			yc = int(yl) 
			objectness[:, yc, xc] = 1.0
			coords[:, yc, xc] = torch.tensor([xl-xc, yl-yc, w, h, disp], dtype=torch.float32) 

		#left, right = torch.chunk(self.transform(img), chunks=2, dim=2)  
		left, right = img[:, :640, :], img[:, 640:, :]
		if self.augmentation:
			left = self.transform(Image.fromarray(left))
			right = self.transform(Image.fromarray(right)) 
		else:
			left = self.transform(left)
			right = self.transform(right)


		return left, right, objectness, coords





class DatasetObjects2(Dataset):
	def __init__(self, path, files_name, augmentation=False):
		super().__init__() 
		self.path = path
		#self.files_name = os.listdir(path+'/images')
		#self.files_name = [f.rpartition('.')[0] for f in self.files_name]
		self.files_name = files_name
		self.augmentation = augmentation
		if augmentation:
			self.transform = transforms.Compose([
				transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), 
				transforms.RandomGrayscale(p=0.2),
				transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 0.5))], p=0.5),   
				transforms.ToTensor(),
				transforms.Compose([
				transforms.RandomErasing(p=0.7, scale=(0.001, 0.003), ratio=(0.3, 3.3), value='random') for _ in range(15)]),
				transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
		else:
			self.transform = transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

	def __len__(self):
		return len(self.files_name) 

	def __getitem__(self, idx):
		with open(self.path+'/labels/'+self.files_name[idx]+'.txt', 'r') as f:
			labels = f.read()
			labels = labels.split('\n')

		left_labels = []
		right_labels = []  

		for label in labels:
			label = label[1:].split()
			if label != []:
				x, y, w, h = list(map(float, label))
				if x<0.5:
					left_labels.append([x, y, w, h])
				else:
					right_labels.append([x, y])

		left_labels = sorted(left_labels, key = lambda lab: lab[0])
		left_labels = sorted(left_labels, key = lambda lab: lab[1])
		right_labels = sorted(right_labels, key = lambda lab: lab[0])
		right_labels = sorted(right_labels, key = lambda lab: lab[1])

		img = cv2.imread(self.path+'/images/'+self.files_name[idx]+'.jpg')
		objectness = torch.zeros(1, 15, 20, dtype=torch.float32)
		coords = torch.zeros(5, 15, 20, dtype=torch.float32)

		for left, right in zip(left_labels, right_labels):
			xl, yl, w, h = left 
			xr, _ = right
			disp = 2*(xl-(xr-0.5))
			xl *= 40 #координаты x левого изображения
			yl *= 15 #координаты y левого изображения 
			i = int(xl) #индекс ячейки по оси x
			j = int(yl) #индекс ячейки по оси y
			#координаты внутри ячейки
			xc = xl-i 
			yc = yl-j 
			
			objectness[:, j, i] = 1.0
			coords[:, j, i] = torch.tensor([xc, yc, w, h, disp], dtype=torch.float32)
			  
			if xc > 0.5:
				ii = i+1  
				if ii < 32: 
					objectness[:, j, ii] = 1.0
					coords[:, j, ii] = torch.tensor([xc-1.0, yc, w, h, disp], dtype=torch.float32)
			elif xc < 0.5:
				ii = i-1
				if ii >= 0: 
					objectness[:, j, ii] = 1.0
					coords[:, j, ii] = torch.tensor([xc+1.0, yc, w, h, disp], dtype=torch.float32)

			if yc > 0.5:
				jj = j+1
				if jj < 32:
					objectness[:, jj, i] = 1.0
					coords[:, jj, i] = torch.tensor([xc, yc-1.0, w, h, disp], dtype=torch.float32)
			elif yc < 0.5:
				jj = j-1 
				if jj >= 0:
					objectness[:, jj, i] = 1.0
					coords[:, jj, i] = torch.tensor([xc, yc+1.0, w, h, disp], dtype=torch.float32)
		
		left, right = img[:, :640, :], img[:, 640:, :]
		if self.augmentation:
			left = self.transform(Image.fromarray(left))
			right = self.transform(Image.fromarray(right)) 
		else:
			left = self.transform(left)
			right = self.transform(right)

		return left, right, objectness, coords





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
	img = tensor.permute(1, 2, 0).detach().numpy()
	img *= np.array([[0.229, 0.224, 0.225]])
	img += np.array([0.485, 0.456, 0.406]) 
	img *= 255.0 
	img[img>255.0]=255.0
	img[img<0.0]=0.0
	img = img.astype('uint8') 
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	return img 