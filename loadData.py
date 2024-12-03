import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
#from Rfund.InputFeature import InputFeature
#from Rfund.Augmentation import Augmentation
import random

def data_loader(dataset_dir, flag='train', batch_size=1):
    dataset = SegmentationDataset(dataset_dir, flag)
    loader = DataLoader(
        dataset, batch_size=batch_size, drop_last=False, num_workers=1
    )
    return loader

def readImages(dirPath):
    imgList = []
    fileNameList = []
    for root, dirs, fs in os.walk(dirPath):
        for fn in sorted(fs):
            _, ext = os.path.splitext(fn)
            if ext not in [".bmp", ".BMP", ".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"]:
                continue
            #fileName = os.path.splitext(os.path.split(fn)[1])[0]
            fileNameList.append(fn)
            filePath = os.path.join(root, fn)
            img = cv2.imread(filePath, cv2.IMREAD_GRAYSCALE)
            imgList.append(img)
    #print(np.array(imgList).shape,len(fileNameList))
    #print(imgList[0])
    return np.array(imgList), fileNameList

class ImageSet:
    def __init__(self, dataset_dir) -> None:
        #self.negativeOrder = list(range(0,39))
        #self.positiveOrder = list(range(39,58))
        #random.shuffle(self.negativeOrder)
        #random.shuffle(self.positiveOrder)
        #self.negativeOrder = [29, 25, 27, 32, 9, 7, 24, 12, 2, 16, 20, 30, 33, 23, 28, 19, 5, 6, 35, 17, 10, 13, 15, 21, 8, 3, 37, 4, 34, 0, 31, 18, 14, 26, 36, 22, 11, 1, 38]
        self.negativeOrder = [19, 18, 21, 28, 23, 6, 12, 3, 34, 13, 32, 37, 0, 31, 27, 16, 10, 22, 30, 4, 26, 25, 35, 14, 11, 17, 8, 7, 36, 29, 1, 38, 20, 24, 15, 5, 9, 2, 33]
        #self.positiveOrder = [48, 44, 41, 57, 53, 46, 42, 49, 56, 40, 47, 52, 55, 51, 39, 54, 45, 43, 50]
        self.positiveOrder = [44, 40, 39, 41, 43, 42]
        
        originalImgDir = os.path.join(dataset_dir, 'original/image')
        featureRootDir = os.path.join(dataset_dir, 'feature')
        labelRootDir = os.path.join(dataset_dir, 'original/label/')
        augDir = os.path.join(dataset_dir, 'augmentation')
        print("reading images...")
        imgList, fileNameList = readImages(originalImgDir)
        labelImgList, _ = readImages(labelRootDir)
        labelImgList = np.expand_dims(labelImgList, axis=-1)
        dataSize = len(imgList)
        self.originalSize = dataSize
        self.rawImages = imgList[..., np.newaxis]
        self.fileNames = fileNameList
        self.labelImgList = labelImgList
        imgSize = np.shape(imgList)[-2:]
        self.imageShape = imgSize
    
    def getImageSet(self, set_slice, flag):
        NEGATIVENUM = 39
        # 39 and 6
        negativeSizeInOneAug = set_slice[0][1]-set_slice[0][0]
        positiveSizeInOneAug = set_slice[1][1]-set_slice[1][0]
        # 45
        currSetSize = negativeSizeInOneAug + positiveSizeInOneAug
        
        negIndexList = list(range(set_slice[0][0], set_slice[0][1]))
        
        posIndexList = list(range(set_slice[1][0], set_slice[1][1]))
        print(negIndexList)
        print(posIndexList)
        
        if flag=='train':
            # becomes 0
            currSetSize = self.originalSize - currSetSize
            negativeSizeInOneAug = NEGATIVENUM - negativeSizeInOneAug
            positiveSizeInOneAug = currSetSize - negativeSizeInOneAug
            currSetSize = currSetSize
            negIndexList = list(range(0,set_slice[0][0]))+list(range(set_slice[0][1], NEGATIVENUM))
            posIndexList = list(range(NEGATIVENUM,set_slice[1][0]))+list(range(set_slice[1][1], self.originalSize))
            print('=========getImageSet=========', set_slice, negativeSizeInOneAug, positiveSizeInOneAug, currSetSize)
        posIndexList = [i-NEGATIVENUM for i in posIndexList]
        imageSet = np.zeros((currSetSize, self.imageShape[0], self.imageShape[1], 1))
        labelSet = np.zeros((currSetSize, self.imageShape[0], self.imageShape[1], 1))
        id = 0
        rawImages = self.rawImages
        labelImages = self.labelImgList
        imageSet[id:id+negativeSizeInOneAug] = np.array([rawImages[self.negativeOrder[i]] for i in negIndexList])
        labelSet[id:id+negativeSizeInOneAug] = np.array([labelImages[self.negativeOrder[i]] for i in negIndexList])

        id+=negativeSizeInOneAug

        imageSet[id:id+positiveSizeInOneAug] = np.array([rawImages[self.positiveOrder[i]] for i in posIndexList])
        labelSet[id:id+positiveSizeInOneAug] = np.array([labelImages[self.positiveOrder[i]] for i in posIndexList])

        id+=positiveSizeInOneAug

        print("===========", flag, np.shape(imageSet), "===========")
        
        return currSetSize, imageSet, labelSet
         
    def getFileNameSet(self, set_slice, flag):
        NEGATIVENUM = 39
        negIndexList = list(range(set_slice[0][0], set_slice[0][1]))
        posIndexList = list(range(set_slice[1][0], set_slice[1][1]))
        if flag=='train':
            negIndexList = list(range(0,set_slice[0][0]))+list(range(set_slice[0][1], NEGATIVENUM))
            posIndexList = list(range(NEGATIVENUM,set_slice[1][0]))+list(range(set_slice[1][1], self.originalSize))
        posIndexList = [i-NEGATIVENUM for i in posIndexList]
        fileIndexList = [self.negativeOrder[i] for i in negIndexList] + [self.positiveOrder[i] for i in posIndexList]
        fileNameList = [self.fileNames[i] for i in fileIndexList]
        return fileNameList    
    
    def getRawImageSet(self, set_slice, flag):
        NEGATIVENUM = 39
        negIndexList = list(range(set_slice[0][0], set_slice[0][1]))
        posIndexList = list(range(set_slice[1][0], set_slice[1][1]))
        if flag=='train':
            negIndexList = list(range(0,set_slice[0][0]))+list(range(set_slice[0][1], NEGATIVENUM))
            posIndexList = list(range(NEGATIVENUM,set_slice[1][0]))+list(range(set_slice[1][1], self.originalSize))
        posIndexList = [i-NEGATIVENUM for i in posIndexList]
        imageIndexList = [self.negativeOrder[i] for i in negIndexList] + [self.positiveOrder[i] for i in posIndexList]
        rawImageList = [self.rawImages[i] for i in imageIndexList]
        return rawImageList
    
class SegmentationDataset(Dataset):
    def __init__(self, image_set: ImageSet, set_slice, flag = 'test') -> None:
        self.set_size, self.imageSet, self.labelImages = image_set.getImageSet(set_slice, flag)
        self.transform = transforms.Compose([
                transforms.ToTensor() # 0-255 -> 0-1, dimensions (H, W, C) -> (C, H, W)
        ])

    def __len__(self):
        return self.set_size
    
    def __getitem__(self, index):
        image = self.imageSet[index].astype(np.uint8) # dimensions (H, W, C)
        label = self.labelImages[index].astype(np.uint8) # dimensions (H, W, C)
        #image_tensor = torch.from_numpy(image.astype(np.float32))
        #label_tensor = torch.from_numpy(label.astype(np.float32))
        image_tensor = self.transform(image)
        label_tensor = label
        #eric
        #for multiple classes this part has been changed as labels should not be normalized
        #label_tensor = torch.tensor(label, dtype=torch.long)
        #label_tensor = label_tensor.squeeze(0)
        #label_tensor = self.transform(label)
        #print(image_tensor)
        return [image_tensor, label_tensor]