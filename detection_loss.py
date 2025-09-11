import torch
import torch.nn.functional as F




def objectness_loss(pred_conf, gt_obj, neg_pos_ratio=2, min_neg=10, pos_weight=2.0, neg_weight=1.0, eps=1e-7):
    batch_size, _, height, width = pred_conf.shape
    
    pred_conf = pred_conf.view(batch_size, -1) 
    gt_obj = gt_obj.view(batch_size, -1)        
    
    pos_mask = (gt_obj == 1.0)
    neg_mask = (gt_obj == 0.0)
    
    num_pos_per_batch = pos_mask.sum(dim=1)  
    total_pos = num_pos_per_batch.sum().float()
    
    selected_pos_mask = pos_mask.clone()
    selected_neg_mask = torch.zeros_like(neg_mask)
    
    for i in range(batch_size):
        num_pos = num_pos_per_batch[i]
        total_neg = neg_mask[i].sum()
        
        if total_neg == 0:
            continue
            
        if num_pos > 0:
            num_neg = min(neg_pos_ratio*num_pos, total_neg)
        else:
            num_neg = min(min_neg, total_neg)
        
        if num_neg > 0:
            neg_conf = pred_conf[i][neg_mask[i]]
            noise = torch.rand_like(neg_conf) * 1e-5
            _, top_idx = torch.topk(neg_conf + noise, k=int(num_neg), largest=True)
            neg_indices = torch.where(neg_mask[i])[0][top_idx]
            selected_neg_mask[i, neg_indices] = 1
    
    selected_mask = selected_pos_mask | selected_neg_mask
    selected_pred = pred_conf[selected_mask]
    selected_gt = gt_obj[selected_mask]

    weights = torch.ones_like(selected_pred)
    weights[selected_gt == 1] = pos_weight
    weights[selected_gt == 0] = neg_weight
    
    loss = F.binary_cross_entropy_with_logits(selected_pred, selected_gt, weight=weights, reduction='sum')
    loss = loss/torch.clamp(total_pos, min=1.0)
    
    return loss





def smooth_l1(diff, sigma):
        abs_diff = torch.abs(diff)
        return torch.where(abs_diff<1/sigma**2, 0.5*sigma**2*diff**2, abs_diff-0.5/sigma**2)





def coords_loss_fn(pred_boxes, gt_boxes, gt_obj, sigma=1.0, use_log_for_size=True, 
                   center_weight=1.0, size_weight=1.0, disparity_weight=1.0, use_log_for_disparity=False):
    
    batch_size, _, height, width = pred_boxes.shape
    
    pred_boxes = pred_boxes.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 5)
    gt_boxes = gt_boxes.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 5)
    gt_obj = gt_obj.view(batch_size, -1)
    
    pos_mask = (gt_obj == 1)
    if pos_mask.sum() == 0:
        return torch.tensor(0.0, device=pred_boxes.device)
    
    pred = pred_boxes[pos_mask]  
    gt = gt_boxes[pos_mask]     
    
    pred_xy = torch.sigmoid(pred[:, :2])*2.0-0.5       
    pred_wh = torch.sigmoid(pred[:, 2:4])       
    pred_disp = torch.sigmoid(pred[:, 4])       
    
    gt_xy = gt[:, :2]            
    gt_wh = gt[:, 2:4]          
    gt_disp = gt[:, 4]           
    
    xy_diff = pred_xy - gt_xy
    xy_loss = smooth_l1(xy_diff, sigma)
    xy_loss = xy_loss.sum(dim=1).mean()  
    
    if use_log_for_size:
        wh_diff = torch.log(pred_wh) - torch.log(gt_wh)
    else:
        wh_diff = pred_wh - gt_wh
    
    wh_loss = smooth_l1(wh_diff, sigma)
    wh_loss = wh_loss.sum(dim=1).mean()  
    
    if use_log_for_disparity:
        disp_diff = torch.log(pred_disp + 1e-8) - torch.log(gt_disp + 1e-8)
    else:
        disp_diff = pred_disp-gt_disp
    
    disp_loss = smooth_l1(disp_diff, sigma)
    disp_loss = disp_loss.mean() 
    
    total_loss = (center_weight*xy_loss+size_weight*wh_loss+disparity_weight*disp_loss)
    
    return total_loss
     




def loss_fn(pred_objectness, pred_coords, objectness, coords):
    B, C, H, W = pred_objectness.shape
    obj_loss = objectness_loss(pred_objectness, objectness)
    coords_loss = 2.0*coords_loss_fn(pred_coords[:, 0:5, :, :], coords[:, 0:5, :, :], objectness)

    return obj_loss+coords_loss, obj_loss, coords_loss





def disparity_metric(pred_disp, gt_disp, gt_obj):
    obj_mask = (gt_obj == 1.0)

    pred_disp = torch.sigmoid(pred_disp)
    
    if not obj_mask.any():
        return torch.tensor(0.0, device=pred_disp.device)
    
    l1_loss = torch.abs(pred_disp[obj_mask] - gt_disp[obj_mask])
    return l1_loss.mean()