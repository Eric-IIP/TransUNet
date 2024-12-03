import os
import torch
import cv2
import numpy as np
import preprocess as prep
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

class ImgageSet:
    def __init__(self, dataset_dir) -> None:
        self.negativeOrder = list(range(0,39))
        self.positiveOrder = list(range(39,58))
        random.shuffle(self.negativeOrder)
        random.shuffle(self.positiveOrder)

        originalImgDir = os.path.join(dataset_dir, 'original/image')
        featureRootDir = os.path.join(dataset_dir, 'feature')
        labelRootDir = os.path.join(dataset_dir, 'original/label/')
        augDir = os.path.join(dataset_dir, 'augmentation')
        print("reading images...")
        imgList, fileNameList = prep.readImages(originalImgDir)
        labelImgList, _ = prep.readImages(labelRootDir)
        dataSize = len(imgList)
        imgSize = np.shape(imgList)[-2:]
        print(np.shape(imgList),np.shape(labelImgList))
        labelImgList = np.expand_dims(labelImgList, axis=-1)
        prep.createAugmentedImages(augDir, featureRootDir, imgList, labelImgList, fileNameList)
        self.rawImages = imgList
        self.fileNames = fileNameList
        self.multiChannelImages = {}
        self.labelImages = {}
        self.originalSize = dataSize
        self.imageShape = imgSize
        for augmentation in Augmentation:
            augmentationVal = augmentation.value
            
            currLabelPath = os.path.join(augDir, 'labels', augmentationVal)
            labelList, fileNameList = prep.readImages(currLabelPath)
            i=0
            multiChannelImages = np.zeros((dataSize,imgSize[0],imgSize[1],len(InputFeature)))
            for inputFeature in InputFeature:
                featureVal = inputFeature.value
                imgPath = os.path.join(featureRootDir, augmentationVal, featureVal)
                #print("imgPath", imgPath)
                imgList, fileNameList = prep.readImages(imgPath)
                #if inputFeature == InputFeature.GRY_ and augmentation == Augmentation.SCALEX:
                #    prep.saveImages(imgList, fileNameList, os.path.join('../result/tempImage/afterread', augmentationVal))
                #    exit()
                prep.stackChannelImages(multiChannelImages, imgList, i)
                i+=1
            self.multiChannelImages[augmentationVal] = multiChannelImages
            self.labelImages[augmentationVal] = labelList[..., np.newaxis]
            print(np.shape(self.multiChannelImages), np.shape(self.labelImages))
        for k in self.multiChannelImages:
            print(k)
    
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
        
    def getImageSet(self, set_slice, flag):
        NEGATIVENUM = 39
        negativeSizeInOneAug = set_slice[0][1]-set_slice[0][0]
        positiveSizeInOneAug = set_slice[1][1]-set_slice[1][0]
        currSetSize = negativeSizeInOneAug + positiveSizeInOneAug
        negIndexList = list(range(set_slice[0][0], set_slice[0][1]))
        posIndexList = list(range(set_slice[1][0], set_slice[1][1]))
        if flag=='train':
            currSetSize = self.originalSize - currSetSize
            negativeSizeInOneAug = NEGATIVENUM - negativeSizeInOneAug
            positiveSizeInOneAug = currSetSize - negativeSizeInOneAug
            currSetSize = currSetSize * len(Augmentation)
            negIndexList = list(range(0,set_slice[0][0]))+list(range(set_slice[0][1], NEGATIVENUM))
            posIndexList = list(range(NEGATIVENUM,set_slice[1][0]))+list(range(set_slice[1][1], self.originalSize))
            print('=========getImageSet=========', set_slice, negativeSizeInOneAug, positiveSizeInOneAug, currSetSize)
        posIndexList = [i-NEGATIVENUM for i in posIndexList]
        '''
        print("===========index list===========")
        print(len(negIndexList), negIndexList, len(posIndexList), posIndexList)
        print("===========actual index list===========")
        print(self.negativeOrder, [self.negativeOrder[i] for i in negIndexList])
        print(self.positiveOrder, [self.positiveOrder[i] for i in posIndexList])
        '''
        imageSet = np.zeros((currSetSize, self.imageShape[0], self.imageShape[1], len(InputFeature)))
        labelSet = np.zeros((currSetSize, self.imageShape[0], self.imageShape[1], 1))
        id = 0
        if flag == 'test':
            multiChannelImages = self.multiChannelImages[Augmentation.ORIGINAL.value]
            labelImages = self.labelImages[Augmentation.ORIGINAL.value]
        
            imageSet[id:id+negativeSizeInOneAug] = np.array([multiChannelImages[self.negativeOrder[i]] for i in negIndexList])
            labelSet[id:id+negativeSizeInOneAug] = np.array([labelImages[self.negativeOrder[i]] for i in negIndexList])
            
            id+=negativeSizeInOneAug
            
            imageSet[id:id+positiveSizeInOneAug] = np.array([multiChannelImages[self.positiveOrder[i]] for i in posIndexList])
            labelSet[id:id+positiveSizeInOneAug] = np.array([labelImages[self.positiveOrder[i]] for i in posIndexList])

            id+=positiveSizeInOneAug
        else:
            for key, multiChannelImages in self.multiChannelImages.items():
                labelImages = self.labelImages[key]
                imageSet[id:id+negativeSizeInOneAug] = np.array([multiChannelImages[self.negativeOrder[i]] for i in negIndexList])
                labelSet[id:id+negativeSizeInOneAug] = np.array([labelImages[self.negativeOrder[i]] for i in negIndexList])
                
                id+=negativeSizeInOneAug
                
                imageSet[id:id+positiveSizeInOneAug] = np.array([multiChannelImages[self.positiveOrder[i]] for i in posIndexList])
                labelSet[id:id+positiveSizeInOneAug] = np.array([labelImages[self.positiveOrder[i]] for i in posIndexList])

                id+=positiveSizeInOneAug

        print("===========", flag, np.shape(imageSet), "===========")
        '''
        fileNameSet = self.getFileNameSet(set_slice, flag)
        saveTestPath = os.path.join("/home/luosj/research/test/result/dataloaderDebug", flag)
        os.makedirs(saveTestPath, exist_ok=True)
        for i in range(imageSet.shape[0]):
            img = imageSet[i]
            cv2.imwrite(os.path.join(saveTestPath, "image"+'_'+fileNameSet[i]), img)
            img = labelSet[i]
            cv2.imwrite(os.path.join(saveTestPath, "label"+'_'+fileNameSet[i]), img)
        '''
        return currSetSize, imageSet, labelSet



class SegmentationDataset(Dataset):
    def __init__(self, image_set: ImgageSet, set_slice, flag = 'test') -> None:
        self.set_size, self.multiChannelImages, self.labelImages = image_set.getImageSet(set_slice, flag)
        self.transform = transforms.Compose([
                transforms.ToTensor() # 0-255 -> 0-1, dimensions (H, W, C) -> (C, H, W)
        ])

    def __len__(self):
        return self.set_size
    
    def __getitem__(self, index):
        image = self.multiChannelImages[index].astype(np.uint8) # dimensions (H, W, C)
        label = self.labelImages[index].astype(np.uint8) # dimensions (H, W, C)
        #image_tensor = torch.from_numpy(image.astype(np.float32))
        #label_tensor = torch.from_numpy(label.astype(np.float32))
        image_tensor = self.transform(image)
        label_tensor = self.transform(label)
        return [image_tensor, label_tensor]
        

# ------- The above is the dataset processing of VF cervical spine images --------- #
# ------- The bellow is the processing of Synapse multi-organ segmentation dataset --------- #

class SynapseDataset(Dataset):
    def __init__(self, base_dir, list_dir, flag):
        self.transform = transforms.Compose([
                transforms.ToTensor() # 0-255 -> 0-1, dimensions (H, W, C) -> (C, H, W)
        ])
        self.flag = flag
        self.dataDir = base_dir
        caseList = open(os.path.join(list_dir, self.flag+'.txt')).readlines()
        caseList = [i.strip('\n') for i in caseList]
        self.sampleList = self.getSliceList(caseList)

    def getSliceList(self, caseList):
        sliceList = []
        for caseName in caseList:
            if caseName=='':
                continue
            caseDir = os.path.join(self.dataDir, caseName)
            if not os.path.exists(caseDir):
                continue
            for _, _, fs in os.walk(caseDir):
                for f in fs:
                    if f.endswith('.npz'):
                        sliceList.append(os.path.join(caseName, f))
        return sliceList

    def __len__(self):
        return len(self.sampleList)

    def __getitem__(self, index):
        dataPath = os.path.join(self.dataDir, self.sampleList[index])
        data = np.load(dataPath)
        image, label = data['image'], data['label']
        image = self.transform(image)
        #label = self.transform(label)
        return [image, label]