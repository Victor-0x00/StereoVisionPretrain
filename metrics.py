import torch 




def epe_bad_x(model, dataloader, x, device='cuda'):
	epe = 0
	bad = 0 
	N = 0 
	model.eval()
	for imgL, imgR, dispL, _ in dataloader:
		imgL = imgL.to(device)
		imgR = imgR.to(device)  
		dispL = dispL.to(device)
		B, _, H, W = imgL.shape
		with torch.no_grad():
			pred_disp, _ = torch.chunk(model(imgL, imgR), chunks=2, dim=1)
	
		abs_err = (dispL-pred_disp).abs()
		epe_err = abs_err.sum().item()  
		epe_err /= (H*W)  
		epe += epe_err

		bad_err = (abs_err>x).sum().item()
		bad_err /= (H*W)
		bad += bad_err
		
		N += B 

	return epe/N, bad/N  