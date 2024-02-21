# -*- coding:utf-8 -*- 
# Author: 陈文华(Steven)
# Website: https://wenhua-chen.github.io/
# Github: https://github.com/wenhua-chen
# Date: 2022-09-23 10:58:06
# LastEditTime: 2023-08-25 10:31:08
# Description: 推理

import os
import json
import torch
import cv2
import easyocr
import argparse
import numpy as np
from tqdm import tqdm
from PIL import ImageFont, ImageDraw, Image
from collections import defaultdict
from models.experimental import attempt_load
from utils.datasets import LoadImagesNobg, LoadImagesInMem
from utils.find_text import find_text
from utils.torch_utils import select_device
from utils.general import non_max_suppression, rescale_coords, \
scale_coords, cover_img, crop_img, crop_text, resize_xyxy, erase_img, rearrage_text_xyxy

class Predictor:
    def __init__(self, device):
        # 参数设置
        self.imgsz = [640,640]
        self.device = select_device(device)
        use_gpu = self.device.type != 'cpu'
        self.chinese_font = ImageFont.truetype("utils/字体.ttc",25)

        # 模型加载
        self.block_model = attempt_load('models/exp_block_epoch300.pt', map_location=self.device)
        self.img_model = attempt_load('models/exp_img_epoch300.pt', map_location=self.device)
        self.ocr_model = easyocr.Reader(['ch_sim','en'], gpu=use_gpu)
        if use_gpu:   # run once
            with torch.no_grad():
                self.block_model(torch.zeros(1, 3, *self.imgsz).to(self.device).type_as(next(self.block_model.parameters())))
                self.img_model(torch.zeros(1, 3, *self.imgsz).to(self.device).type_as(next(self.img_model.parameters())))

        # 用于保存结果
        self.result = {}

    @torch.no_grad()
    def __pred_dets(self, model, img, img0): # 模型推理
        # 预处理
        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # uint8 to fp16/32
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        # infer
        pred = model(img, augment=False, visualize=False)[0]
        # 后处理
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, max_det=100)
        dets = pred[0] # 输入只有一张图片
        dets = dets[torch.sort(dets, dim=0)[1][:,1]] # 从上到下顺序排列
        dets = scale_coords(img.shape[2:], dets[:, :4], img0.shape).round()
        dets = np.array(dets.cpu(),dtype=np.int32)
        return dets
    
    # 预测模块位置
    def __pred_blocks(self):
        print('[图片定位]')
        # 准备结果
        base_names, fullimg0s = [], [] # 图片名称, 图片内容
        blocks, blocks_xyxy = [], [] # blocks内容, blocks坐标

        # 推理
        dataset = LoadImagesNobg(self.img_dir, img_size=self.imgsz, stride=32, auto=True)
        for path, fullimg, fullimg0, _ in tqdm(dataset):
            base_names.append(os.path.splitext(os.path.basename(path))[0])
            fullimg0s.append(fullimg0)
            dets = self.__pred_dets(self.block_model, fullimg, fullimg0)
            blocks_img, blocks_xyxy_img = [], []
            fullimg0_w = fullimg0.shape[1]
            # if dets[0][1] < fullimg0_w/6: # 过滤头部误识别的模块
            #     dets = dets[1:]
            for xyxy in dets:
                xyxy_w_ratio, xyxy_h_ratio = (xyxy[2]-xyxy[0])/fullimg0_w, (xyxy[3]-xyxy[1])/fullimg0_w
                if (xyxy_w_ratio<0.8) or (xyxy_h_ratio<0.4): # 过滤比例不对的block
                    continue
                xyxy[0], xyxy[2] = 0, fullimg0_w # block拓展至原图宽度
                block = crop_img(xyxy, fullimg0, gain_wh=(1,1))
                blocks_img.append(block)
                blocks_xyxy_img.append(xyxy)
            # print(fullimg0.shape)
            # print(len(blocks_img))
            blocks.append(blocks_img)
            blocks_xyxy.append(blocks_xyxy_img)
        
        # 保存结果
        self.result['base_names'] = base_names
        self.result['fullimg0s'] = fullimg0s
        self.result['blocks'] = blocks
        self.result['blocks_xyxy'] = blocks_xyxy
    
    # 裁剪图片区域
    def __crop_areas(self):
        print('[图片裁剪]')
        # 准备结果
        covered_blocks, areas, areas_xyxy = [], [], []

        # 推理
        for blocks_img in tqdm(self.result['blocks']):
            blocks_img = LoadImagesInMem(blocks_img, img_size=self.imgsz, stride=32, auto=True)
            covered_blocks_img, areas_img, areas_xyxy_img = [], [], []
            for block, block0 in blocks_img:
                dets = self.__pred_dets(self.img_model, block, block0)
                block_w = block0.shape[1]
                default_xyxy = np.array([block_w*0.05, block_w*0.03, block_w*0.3, block_w*0.3], np.int32) # 默认xyxy
                if len(dets) == 0: # 如果没有找到area, 用默认值
                    xyxy = default_xyxy
                else:
                    xyxy = dets[0]
                    xyxy_w, xyxy_h = xyxy[2]-xyxy[0], xyxy[3]-xyxy[1]
                    xyxy_ratio = min(xyxy_w,xyxy_h)/max(xyxy_w,xyxy_h)
                    xyxy_w_ratio = xyxy_w/block_w
                    # 如果找到的xyxy比例不对, 也用默认值
                    if xyxy_ratio<0.9 or xyxy_w_ratio<0.2 or 0.25<xyxy_w_ratio:
                        xyxy= default_xyxy
                # xyxy = resize_xyxy(xyxy, block0, gain_wh=(1.1,1.1))
                area = crop_img(xyxy, block0, gain_wh=(1,1))
                covered_block = cover_img(xyxy, block0, gain_wh=(1.1,1.1))
                covered_block = erase_img(covered_block)
                covered_blocks_img.append(covered_block)
                areas_img.append(area)
                areas_xyxy_img.append(xyxy)
            covered_blocks.append(covered_blocks_img)
            areas.append(areas_img)
            areas_xyxy.append(areas_xyxy_img)
        
        # 保存结果
        self.result['covered_blocks'] = covered_blocks
        self.result['areas'] = areas
        self.result['areas_xyxy'] = areas_xyxy
    
    # 预测文字位置
    def __find_text(self):
        print('[文字定位]')
        # 准备结果
        texts_xyxy = []
        texts_crop = []
        texts_shape = []
        
        # 推理
        for i, covered_blocks_img in enumerate(tqdm(self.result['covered_blocks'])):
        # for i, covered_blocks_img in enumerate(self.result['covered_blocks']):
            texts_xyxy_img = []
            texts_crop_img = []
            texts_shape_img = []
            
            filter_j = []
            # for j, covered_block in enumerate(tqdm(covered_blocks_img)):
            for j, covered_block in enumerate(covered_blocks_img):
                area_xyxy = self.result['areas_xyxy'][i][j] # 相对block的坐标
                try:
                    xyxy_lst = find_text(covered_block)
                    xyxy_lst, lst_shape = rearrage_text_xyxy(xyxy_lst, area_xyxy)
                except: # 过滤异常
                    filter_j.append(j)
                    continue
                # xyxy_lst = resize_xyxy(xyxy_lst, covered_block, gain_wh=(1,1.2))
                # 根据是否有[0,0,0,0]来过滤误检测的模块
                if (j==0) and (sum(xyxy_lst[lst_shape[0]]) != 0):
                    filter_j.append(j)
                    continue
                texts_crop_img.append(crop_text(xyxy_lst, covered_block, gain_wh=(1,1.2)))
                texts_xyxy_img.append(xyxy_lst)
                texts_shape_img.append(lst_shape)
            texts_xyxy.append(texts_xyxy_img)
            texts_crop.append(texts_crop_img)
            texts_shape.append(texts_shape_img)
            
            # 进行过滤
            self.result['blocks'][i] = [self.result['blocks'][i][k] for k in range(len(covered_blocks_img)) if k not in filter_j]
            self.result['blocks_xyxy'][i] = [self.result['blocks_xyxy'][i][k] for k in range(len(covered_blocks_img)) if k not in filter_j]
            self.result['covered_blocks'][i] = [self.result['covered_blocks'][i][k] for k in range(len(covered_blocks_img)) if k not in filter_j]
            self.result['areas'][i] = [self.result['areas'][i][k] for k in range(len(covered_blocks_img)) if k not in filter_j]
            self.result['areas_xyxy'][i] = [self.result['areas_xyxy'][i][k] for k in range(len(covered_blocks_img)) if k not in filter_j]

        # 保存结果
        self.result['texts_xyxy'] = texts_xyxy
        self.result['texts_crop'] = texts_crop
        self.result['texts_shape'] = texts_shape

    # ocr文字识别
    def __ocr(self):
        print('[文字识别]')
        # 准备结果
        title_ocr, table_ocr = [], []
        json_ocr = []

        # 推理
        for i, texts_crop_img in enumerate(tqdm(self.result['texts_crop'])):
        # for i, texts_crop_img in enumerate(self.result['texts_crop']):
            title_lst_ocr_img, table_array_ocr_img = [], []
            json_lst_ocr_img = []
            # for j, text_crop_lst in enumerate(tqdm(texts_crop_img)):
            for j, text_crop_lst in enumerate(texts_crop_img):
                # 结构拆解
                lst_shape = self.result['texts_shape'][i][j]
                title_lst = text_crop_lst[:lst_shape[0]]
                table_array = np.array(text_crop_lst[lst_shape[0]:], dtype=object).reshape(*lst_shape[1:3])
                # 平均行高
                crop_h = [crop.shape[0] for crop in title_lst]
                mid_crop_h = sorted(crop_h)[len(crop_h)//2] # 中位数
                
                # ocr
                json_dict = defaultdict(str)
                title_lst_ocr = []
                for k, crop in enumerate(title_lst):
                    text = self.ocr_model.recognize(crop, detail=0)[0].replace(' ','').replace('~','').replace('。','') # 误识别, 修改结果
                    if k == 0:
                        text = ''.join([c for c in text if not '\u4e00'<=c<='\u9fa5']) # 过滤所有中文字符
                        json_dict['productCode'] = text
                    elif k == 1:
                        text = text.replace('6U', 'GU') # GU
                        text = text.replace('6u', 'GU')
                        json_dict['productName'] = text
                    elif k == 2:
                        if '元' in text: # "x元 * x件"
                            price, cnt = text.split('元')
                            text = f'{price}元 x {cnt[1:]}'
                            json_dict['price'] = price
                            json_dict['retailPrice'] = price
                    elif k == 3:
                        text = f'¥{text[1:]}' # "¥xxx"
                    title_lst_ocr.append(text)
                sku_lst = []
                table_array_ocr = np.empty_like(table_array, dtype=object)
                for x in range(lst_shape[1]):
                    for y in range(lst_shape[2]):
                        crop = table_array[x][y]
                        crop_h = crop.shape[0]
                        if crop_h == 0: # 如果crop高度为0, 表示为空
                            table_array_ocr[x][y] = ''
                        elif crop_h < mid_crop_h*0.5: # 如果crop高度比平均行高*0.5还小, 表示为'-'
                            table_array_ocr[x][y] = '-'
                        else:
                            text = self.ocr_model.recognize(crop, detail=0)[0].replace('~','').replace('。','') # 误识别, 修改结果
                            if x == 0: # table第一行修改
                                if text == '8' or text == '5':
                                    text = 'S'
                                elif text == 'N':
                                    text = 'M'
                                elif text == '匕':
                                    text = 'L'
                            elif y > 0:
                                text = text.replace('}','1') # 数字误识别, 修改结果
                            table_array_ocr[x][y] = text
                        if (x>0) and (y>0): # 每个数字是一个元素, 添加到sku_lst中
                            sku_lst.append({
                                'colorName': table_array_ocr[x][0],
                                'sizeName': table_array_ocr[0][y],
                                'count': table_array_ocr[x][y]
                            })
                json_dict['skuList'] = sku_lst
                # 保存
                title_lst_ocr_img.append(title_lst_ocr)
                table_array_ocr_img.append(table_array_ocr)
                json_lst_ocr_img.append(json_dict)
            title_ocr.append(title_lst_ocr_img)
            table_ocr.append(table_array_ocr_img)
            json_ocr.append(json_lst_ocr_img)
        
        # 保存结果
        self.result['title_ocr'] = title_ocr
        self.result['table_ocr'] = table_ocr
        self.result['json_ocr'] = json_ocr

    # 保存结果
    def __save_result(self, save_json=True, save_img=True, save_annotation=True, \
                        save_block=False, save_covered_block=False):
        print('[保存结果]')
        # 保存block
        if save_block:
            block_dir = f'{self.output_dir}/blocks'
            if not os.path.exists(block_dir):
                os.makedirs(block_dir)
            for i, blocks_img in enumerate(self.result['blocks']):
                base_name = self.result['base_names'][i]
                for j, block in enumerate(blocks_img):
                    output_path = f'{block_dir}/{base_name}_{j}.jpg'
                    cv2.imwrite(output_path, block)
                    self.result['json_ocr'][i][j]['blockUrl'] = output_path
        
        # 保存covered_block
        if save_covered_block:
            covered_block_dir = f'{self.output_dir}/covered_blocks'
            if not os.path.exists(covered_block_dir):
                os.makedirs(covered_block_dir)
            for i, covered_blocks_img in enumerate(self.result['covered_blocks']):
                base_name = self.result['base_names'][i]
                for j, covered_block in enumerate(covered_blocks_img):
                    output_path = f'{covered_block_dir}/{base_name}_{j}.jpg'
                    cv2.imwrite(output_path, covered_block)
                    self.result['json_ocr'][i][j]['coveredUrl'] = output_path
        
        # 保存img
        if save_img:
            area_dir = f'{self.output_dir}/imgs'
            if not os.path.exists(area_dir):
                os.makedirs(area_dir)
            for i, areas_img in enumerate(self.result['areas']):
                base_name = self.result['base_names'][i]
                for j, area in enumerate(areas_img):
                    output_path = f'{area_dir}/{base_name}_{j}.jpg'
                    cv2.imwrite(output_path, area)
                    self.result['json_ocr'][i][j]['imgUrl'] = output_path
        
        # 保存annotation
        if save_annotation:
            annotation_dir = f'{self.output_dir}/annotations'
            if not os.path.exists(annotation_dir):
                os.makedirs(annotation_dir)
            for i, fullimg0 in enumerate(self.result['fullimg0s']):
                base_name = self.result['base_names'][i]
                output_path = f'{annotation_dir}/{base_name}.jpg'
                # 画检测框
                for j, block_xyxy in enumerate(self.result['blocks_xyxy'][i]):
                    # 画block_xyxy
                    cv2.rectangle(fullimg0, block_xyxy[:2], block_xyxy[2:], (255, 0, 0), 2)
                    # 画area_xyxy
                    area_xyxy = self.result['areas_xyxy'][i][j] # 相对block的坐标, 需要转换
                    block_xyxy = self.result['blocks_xyxy'][i][j]
                    area_xyxy = rescale_coords(block_xyxy, area_xyxy, fullimg0.shape)
                    cv2.rectangle(fullimg0, area_xyxy[:2], area_xyxy[2:], (0, 255, 0), 2)
                    
                    # 画text_xyxy和ocr结果
                    text_xyxy_lst = self.result['texts_xyxy'][i][j]
                    title_lst_ocr = self.result['title_ocr'][i][j]
                    table_array_ocr = self.result['table_ocr'][i][j]
                    len_lst = len(title_lst_ocr)
                    array_h, array_w = table_array_ocr.shape
                    # title_xyxy
                    for k in range(len_lst):
                        text_xyxy = rescale_coords(block_xyxy, text_xyxy_lst[k], fullimg0.shape)
                        cv2.rectangle(fullimg0, text_xyxy[:2], text_xyxy[2:], (0, 0, 255), 2)
                    # table_xyxy
                    for x in range(array_h):
                        for y in range(array_w):
                            index = len_lst + x*array_w + y
                            text_xyxy = rescale_coords(block_xyxy, text_xyxy_lst[index], fullimg0.shape)
                            cv2.rectangle(fullimg0, text_xyxy[:2], text_xyxy[2:], (0, 0, 255), 2)
                    # title_ocr
                    img_pil = Image.fromarray(fullimg0)
                    draw = ImageDraw.Draw(img_pil)
                    for k in range(len_lst):
                        text_xyxy = rescale_coords(block_xyxy, text_xyxy_lst[k], fullimg0.shape)
                        write_x = text_xyxy[0]
                        write_y = text_xyxy[3]
                        draw.text((write_x,write_y), title_lst_ocr[k], font=self.chinese_font, fill=(255,0,0))
                    # table_ocr
                    for x in range(array_h):
                        for y in range(array_w):
                            index = len_lst + x*array_w + y
                            text_xyxy = rescale_coords(block_xyxy, text_xyxy_lst[index], fullimg0.shape)
                            write_x = text_xyxy[0]
                            write_y = text_xyxy[3]+2
                            draw.text((write_x,write_y), table_array_ocr[x][y], font=self.chinese_font, fill=(255,0,0))
                    fullimg0 = np.array(img_pil)
                    self.result['json_ocr'][i][j]['annotationUrl'] = output_path
                cv2.imwrite(output_path, fullimg0)

        # 保存json结果
        if save_json:
            json_dir = f'{self.output_dir}/jsons'
            if not os.path.exists(json_dir):
                os.makedirs(json_dir)
            for i, json_lst_ocr_img in enumerate(self.result['json_ocr']):
                base_name = self.result['base_names'][i]
                output_path = f'{json_dir}/{base_name}.json'
                with open(output_path, 'w') as f:
                    json.dump(json_lst_ocr_img, f, ensure_ascii=False)
                # 打印结果到控制台
                if len(json_lst_ocr_img) != 0:
                    print_json = {
                        "code": 0,
                        "msg": "识别成功",
                        "data": json_lst_ocr_img
                        # "data": json.dumps(json_lst_ocr_img, ensure_ascii=False)
                    }
                else:
                    print_json = {
                        "code": 1,
                        "msg": "没有找到block区域",
                        "data": ""
                    }
                print(json.dumps(print_json, ensure_ascii=False))
    
    # 分阶段处理, 每次输入最大图片数量不超过1000张
    def pred(self, img_dir, output_dir):
        self.img_dir = img_dir
        self.output_dir = output_dir
        
        # 预测模块位置
        self.__pred_blocks()

        # 裁剪图片区域
        self.__crop_areas()
        
        # 预测文字位置
        self.__find_text()

        # ocr文字识别
        self.__ocr()

        # 保存结果
        self.__save_result()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, default='data/测试图片', help='输入图片路径')
    parser.add_argument('--output_dir', type=str, default='data/测试结果', help='结果保存路径')
    parser.add_argument('--device', default='cpu', help='0表示使用显卡, 默认使用cpu')
    opt = parser.parse_args()
    return opt



if __name__ == '__main__':
    opt = parse_opt()
    predictor = Predictor(opt.device)
    predictor.pred(opt.img_dir, opt.output_dir)
