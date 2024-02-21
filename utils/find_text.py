# -*- coding:utf-8 -*- 
# Author: 陈文华(Steven)
# Website: https://wenhua-chen.github.io/
# Github: https://github.com/wenhua-chen
# Date: 2022-09-21 18:29:23
# LastEditTime: 2022-10-07 18:06:19
# Description: 找到文字区域

import cv2
import numpy as np
from glob import glob
import os
import torch
from utils.metrics import bbox_ioa

def preprocess(gray, save=False):
    # 膨胀和腐蚀操作的核函数
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 9))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 6))
 
    # 二值化
    sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize = 3) # Sobel算子，x方向求梯度
    ret, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)

    # 膨胀一次，让轮廓突出
    dilation = cv2.dilate(binary, element2, iterations = 1)
    # 腐蚀一次，去掉细节，如表格线等。注意这里去掉的是竖直的线
    # erosion = cv2.erode(dilation, element1, iterations = 1)
    # 再次膨胀，让轮廓明显一些
    # dilation2 = cv2.dilate(erosion, element2, iterations = 3)
 
    # 存储中间图片 
    if save:
        tmp_dir = f'{os.path.dirname(output_dir)}/tmp_dir'
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        cv2.imwrite(f"{tmp_dir}/{base_name[:-4]}_binary.png", binary)
        cv2.imwrite(f"{tmp_dir}/{base_name[:-4]}_dilation.png", dilation)
        # cv2.imwrite(f"{tmp_dir}/{base_name[:-4]}_erosion.png", erosion)
        # cv2.imwrite(f"{tmp_dir}/{base_name[:-4]}_dilation2.png", dilation2)
    return dilation
 
def findTextBbox(img):
    bboxes = []
    # 找轮廓
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 整理bbox, 初步过滤
    for i in range(len(contours)):
        # 找到最小的矩形，该矩形可能有方向
        cnt = contours[i]
        rect = cv2.minAreaRect(cnt)
        box = np.int0(cv2.boxPoints(rect))
        # 筛选那些太细的矩形，留下扁的
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])
        if(height > width * 1.5):
            continue
        xs = sorted([x for x,_ in box])
        ys = sorted([y for _,y in box])
        xyxy = [xs[0], ys[0], xs[-1], ys[-1]]
        bboxes.append(xyxy)
    # 重合过滤
    bboxes = torch.from_numpy(np.array(bboxes))
    to_delete = []
    for i, bbox in enumerate(bboxes):
        iou = bbox_ioa(bbox, bboxes)>0.9
        indexes = iou.nonzero()
        if len(indexes)>1:
            to_delete.extend(indexes[indexes!=i].tolist())
    indexes = [i for i in range(len(bboxes)) if i not in set(to_delete)]
    bboxes = np.array(bboxes[indexes])
    return bboxes

def find_text(img):
    # 灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 预处理
    dilation = preprocess(gray)
    # 查找文字区域
    bboxes = findTextBbox(dilation)
    return bboxes
 
def detect(img):
    # 灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 预处理
    dilation = preprocess(gray)
    # 查找文字区域
    bboxes = findTextBbox(dilation)
    # 画框
    for bbox in bboxes:
        cv2.rectangle(img, bbox[:2], bbox[2:], (0, 255, 0), 2)
    return img
 
if __name__ == '__main__':
    
    img_dir = 'output/cover'
    output_dir = 'output/text_area'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for img_path in glob(f'{img_dir}/*'):
        base_name = os.path.basename(img_path)
        output_path = f'{output_dir}/{base_name}'

        img = cv2.imread(img_path)
        img = detect(img)

        cv2.imwrite(output_path, img)
