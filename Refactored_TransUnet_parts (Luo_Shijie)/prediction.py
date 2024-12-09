import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from medpy import metric

def predict(
        model,
        dataset,
        device,
        img_size = 256,
        out_threshold=0.5):
    model.eval()
    test_loader = DataLoader(
        dataset, batch_size=1, drop_last=False, num_workers=1
    )
    #activator = torch.nn.Sigmoid()
    pred_images = np.zeros((len(dataset), img_size, img_size))
    mask_images = np.zeros((len(dataset), img_size, img_size))
    with torch.no_grad():
            for i, batch in tqdm(enumerate(test_loader)):
                x, _ = batch
                x= x.to(device, dtype=torch.float)
                y_pred = model(x).cpu()
                #mask = activator(y_pred) > out_threshold
                mask = y_pred > out_threshold  
                pred_images[i,:,:]=y_pred.numpy()
                mask_images[i,:,:]=mask.numpy()
    return pred_images, mask_images

def calculate_metric_percase(pred, gt):
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()  # Convert Tensor to NumPy if it's a Tensor
    if isinstance(gt, torch.Tensor):
        gt = gt.cpu().numpy()  # Convert Tensor to NumPy if it's a Tensor

    # Cast to boolean
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        #hd95 = metric.binary.hd95(pred, gt)
        hd95 = 0
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0

def multi_class_predict(
        model,
        dataset,
        device,
        num_class,
        img_size = 256,
        out_threshold=0.5):
    model.eval()
    test_loader = DataLoader(
        dataset, batch_size=1, drop_last=False, num_workers=1
    )
    #activator = torch.nn.Sigmoid()
    pred_images = np.zeros((len(dataset), img_size, img_size))
    mask_images = np.zeros((len(dataset), img_size, img_size))
    metric_list = 0.0
    with torch.no_grad():
            for i, batch in tqdm(enumerate(test_loader)):
                x, y_true = batch
                x = x.to(device, dtype=torch.float)
                y_pred = torch.argmax(torch.softmax(model(x), dim=1), dim=1).squeeze(0)
                
                # attaching the prediction
                pred_images[i] = y_pred.cpu().detach().numpy()
                
                #mask
                mask_images[i] = y_true.squeeze(0).squeeze(-1).numpy()
                
                prediction = y_pred.cpu().detach().numpy()
                single_metric_list=[]
                for j in range(1, num_class):
                     single_metric_list.append(calculate_metric_percase(prediction==j, y_true==j))
                metric_list += np.array(single_metric_list)
    metric_list=metric_list/len(dataset)
    for i in range(1, num_class):
         print('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    return pred_images, mask_images