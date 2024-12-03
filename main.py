import torch
import getTransUnet
import loadData as dl
import train
from torchinfo import summary
import os
import time
import numpy as np
import prediction
from evaluation import save_eval_result
from unet import UNet
from pyramidDilate import PyramidDilate
from dilated_net import DilatedNet
from dilated_net_6 import DilatedNet as DilatedNet6
from dilated_net_8 import DilatedNet as DilatedNet8
from dilated_net_12 import DilatedNet as DilatedNet12
from unet import UNet

KCONST = 3
NEGATIVENUM = 39
POSITIVENUM = 19
dateTime = time.strftime("%Y%m%d_%H_%M", time.localtime())
resultDir = os.path.join('../result/TransUnet', dateTime)

negSliceList = []
posSliceList = []
def get_set_slice(KConst, NSize, PSize, Round):
    if negSliceList != []:
        return negSliceList[Round], posSliceList[Round]
    negID, posID = 0, 0
    remainder = [NSize%KConst, PSize%KConst]
    for i in range(KConst):
        foldSize = [NSize//KConst, PSize//KConst]
        if (i < remainder[0]):
            foldSize[0]+=1
        if (i < remainder[1]):
            foldSize[1]+=1
        oldNegID, oldPosID = negID, posID
        negID, posID = negID+foldSize[0], posID+foldSize[1]
        negSliceList.append([oldNegID, negID])
        # Positive cases are placed after negative cases
        # The actual index needs to be added with the previous number of negative cases
        posSliceList.append([oldPosID+NSize, posID+NSize]) 
    return negSliceList[Round], posSliceList[Round]

def train_by_model(model, model_name, image_set, round, base_lr=0.01):
    test_slice = get_set_slice(3, NEGATIVENUM, POSITIVENUM, round)
    train_dataset = dl.SegmentationDataset(image_set, test_slice, flag='train')
    test_dataset = dl.SegmentationDataset(image_set, test_slice, flag='test')
    model = train.train(model, device, train_dataset, num_classes=1, base_lr=base_lr, epochs=20)
    torch.save(model.state_dict(), os.path.join(resultDir,model_name+'.pth'))
    _, _, label_img = image_set.getImageSet(test_slice, 'test')
    fileName_set = image_set.getFileNameSet(test_slice, 'train')
    print("file name train", fileName_set)
    fileName_set = image_set.getFileNameSet(test_slice, 'test')
    print("file name test", fileName_set)

    #print("-------",np.shape(image_set.fileNames), np.shape(fileName_set))
    test_raw_img = image_set.getRawImageSet(test_slice, 'test')
    pred_img, mask_img = prediction.predict(model, test_dataset, device)
    subResDir = os.path.join(resultDir, 'iteration '+str(round), model_name)
    meanPrec, meanReca, meanFmea = save_eval_result(predict_image=pred_img*255.0,
                                                    gt_image=label_img,
                                                    test_filenames=fileName_set,
                                                    resDir=subResDir,
                                                    mask_image=mask_img,
                                                    overlay_on=True,
                                                    ori_image=test_raw_img)
    
os.makedirs(resultDir, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image_set = dl.ImageSet(dataset_dir='../dataset')
for i in range(KCONST):
    def train_md_convnet(block_num, dilation_rates, layer_num, net_name):
        start_time = time.time()
        model = DilatedNet(1, 1, block_num = block_num, dilation_rates = dilation_rates, layer_num = layer_num)
        model = model.to(device=device)
        train_by_model(model, net_name, image_set, i, base_lr=0.01)
        end_time = time.time()
        with open(os.path.join(resultDir, net_name+'.txt'), "a+") as f:
            f.writelines("fold" + str(i) + ": " + str(end_time-start_time)+'\n')

    '''
    train_md_convnet(10, [1,3], 2, "md_block10_layer1+3")
    train_md_convnet(10, [1,2,3,5,7], 5, "md_block10_layer1+2+3+5+7")
    train_md_convnet(10, [1,5], 2, "md_block10_layer1+5")
    train_md_convnet(10, [1,2], 2, "md_block10_layer1+2")
    train_md_convnet(10, [1,2,5], 3, "md_block10_layer1+2+5")
    train_md_convnet(8, [1,2,5,9], 4, "md_block8_layer1+2+5+9")
    train_md_convnet(6, [1,2,5,9], 4, "md_block6_layer1+2+5+9")
    '''
    for j in range(1):
        train_md_convnet(10, [1,2,5,9], 4, "md_block10_layer1+2+5+9_"+str(j))
    '''
    train_md_convnet(12, [1,2,5,9], 4, "md_block12_layer1+2+5+9")
    train_md_convnet(10, [1,2,3,5], 4, "md_block10_layer1+2+3+5")
    train_md_convnet(10, [1,2,3,5,7,11], 6, "md_block10_layer1+2+3+5+7+11")
    '''
    start_time = time.time()
    model = getTransUnet.get_transNet(1, bMask=False)
    model = model.to(device=device)
    model_name = 'transunet'
    train_by_model(model, model_name, image_set, i, base_lr=0.01)
    end_time = time.time()
    with open(os.path.join(resultDir, model_name+'.txt'), "a+") as f:
        f.writelines("fold" + str(i) + ": " + str(end_time-start_time)+'\n')
    start_time = time.time()
    model = UNet(1, 1)
    model = model.to(device=device)
    model_name = 'unet'
    train_by_model(model, model_name, image_set, i, base_lr=0.01)
    end_time = time.time()
    with open(os.path.join(resultDir, model_name+'.txt'), "a+") as f:
        f.writelines("fold" + str(i) + ": " + str(end_time-start_time)+'\n')
    '''
    model = DilatedNet8(1, 1)
    model = model.to(device=device)
    train_by_model(model, 'multiscaledilated8', image_set, i, base_lr=0.01)
    model = DilatedNet(1, 1)
    model = model.to(device=device)
    train_by_model(model, 'multiscaledilated10', image_set, i, base_lr=0.01)
    model = DilatedNet(1, 1)
    model = model.to(device=device)
    train_by_model(model, 'multiscaledilated10_1', image_set, i, base_lr=0.01)
    model = DilatedNet(1, 1)
    model = model.to(device=device)
    train_by_model(model, 'multiscaledilated10_2', image_set, i, base_lr=0.01)
    model = DilatedNet(1, 1)
    model = model.to(device=device)
    train_by_model(model, 'multiscaledilated10_3', image_set, i, base_lr=0.01)
    model = DilatedNet(1, 1)
    model = model.to(device=device)
    train_by_model(model, 'multiscaledilated10_4', image_set, i, base_lr=0.01)
    
    '''
    #model = DilatedNet12(1, 1)
    #model = model.to(device=device)
    #train_by_model(model, 'multiscaledilated12', image_set, i, base_lr=0.01)
