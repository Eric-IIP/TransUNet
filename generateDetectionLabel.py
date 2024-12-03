import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

THERSHOLD = 5
def generate_detection_box(binary_segmentation_label):
    # 找到所有非零像素的坐标
    nonzero_indices = np.nonzero(binary_segmentation_label)

    # 计算包围所有坐标的最小矩形框
    min_x = max(0, np.min(nonzero_indices[1])-THERSHOLD)
    max_x = min(256, np.max(nonzero_indices[1])+THERSHOLD)
    min_y = max(0, np.min(nonzero_indices[0])-THERSHOLD)
    max_y = min(256, np.max(nonzero_indices[0])+THERSHOLD)

    detection_box = (min_x, min_y, max_x, max_y)

    return detection_box

if __name__ == '__main__':
    dirPath = '/home/luosj/research/test/dataset/original/label'
    resPath = os.path.join(dirPath, 'box')
    outputPath = os.path.join(resPath, 'image')
    os.makedirs(resPath, exist_ok=True)
    os.makedirs(outputPath, exist_ok=True)
    boxFile = open(os.path.join(resPath, 'boxInfo.txt'),'w')
    files = [f for f in os.listdir(dirPath) if os.path.isfile(os.path.join(dirPath, f))]
    for fn in files:
        _, ext = os.path.splitext(fn)
        if ext not in [".bmp", ".BMP", ".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"]:
            continue
        
        fileName = fn
        filePath = os.path.join(dirPath, fn)
        img = cv2.imread(filePath)
        box = generate_detection_box(img)
        tmp = ' '.join([str(box[i]) for i in range(len(box))])
        boxFile.write(fileName+' '+tmp+'\n')
        x_min, y_min, x_max, y_max = map(int, box)
        boxImg = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255,0,0), 2)
        cv2.imwrite(os.path.join(outputPath, fileName), boxImg)
    boxFile.close()

