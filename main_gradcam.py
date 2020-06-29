import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models
import copy
import os
import sys
from tqdm import tqdm
from torchvision import transforms
from collections import defaultdict
from torch.utils.data import dataloader
from prefetch_generator import BackgroundGenerator
from importlib import import_module
from PIL import Image, ImageFilter
from torchvision.datasets.folder import default_loader
from torchvision import transforms

import model
import loss
from option import args



class DataLoaderX(dataloader.DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class Resize(object):
    """ Resize with cv2
    """
    def __init__(self, size, interpolation=1):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        img = np.array(img)
        img = cv2.resize(img, (self.size[1], self.size[0]), self.interpolation)
        img = Image.fromarray(img)
        return img

def loading(path, model, nGPU):
    state_dict = model.state_dict()
    pretrained = torch.load('./weights/{}/model/model_best.pt'.format(path))
    print('pretrained model loaded~!')
    toupdate = dict()
    total = len(pretrained.keys())
    loaded = 0
    if nGPU < 2:
        for k,v in pretrained.items():
            if 'model.'+k in state_dict.keys():
                toupdate['model.'+k] = v
                loaded += 1
            elif k in state_dict.keys():
                toupdate[k] = v
                loaded += 1
    else:
        for k,v in pretrained.items():
            if 'model.module.'+k in state_dict.keys():
                toupdate['model.module.'+k] = v
                loaded += 1
            elif k in state_dict.keys():
                toupdate[k] = v
                loaded += 1
    print('total params: ', total, ', loaded params: ', loaded)
    state_dict.update(toupdate)
    model.load_state_dict(state_dict)

def cam_maker(grad_block, fmap_block, inputs_source, heigh, width):
    assert grad_block != None, '[ERROR] grad_block is None, please set right layer.'
    # gradcam ++
    gradient = grad_block[-1].squeeze_(0).cpu().numpy()  # [C,H,W]
    gradient = np.maximum(gradient, 0.)  # ReLU
    indicate = np.where(gradient > 0, 1, 0.)  # 示性函数
    norm_factor = np.sum(gradient, axis=(1, 2))  # [C]归一化
    for i in range(len(norm_factor)):
        norm_factor[i] = 1. / norm_factor[i] if norm_factor[i] > 0. else 0.  # 避免除零
    norm_factor[norm_factor > 100000] = 100000 # 避免cam在局部区域过重，同时防止nan出现
    alpha = indicate * norm_factor[:, np.newaxis, np.newaxis]  # [C,H,W]
    weight = np.sum(gradient * alpha, axis=(1, 2))  # [C]  alpha*ReLU(gradient)

    # gradcam
    # weight = np.mean(gradient, axis=(1, 2))

    feature = fmap_block[-1].cpu().data.numpy()[0, :]  # [C,H,W]

    cam = feature * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
    cam = np.sum(cam, axis=0)  # [H,W]
    # cam = np.maximum(cam, 0)  # ReLU
    cam = cv2.resize(cam, (width, heigh))
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)

    # to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    inputs_source = np.float32(inputs_source) / 255
    cam = heatmap * 0.75 + inputs_source
    cam = cam / np.max(cam)
    cam = np.hstack((cam, heatmap))
    return cam

class GradCam:
    def __init__(self, args, model, dataloaderX, loss):
        self.args = args
        self.model = model
        self.loss = loss
        self.device = torch.device('cuda')
        self.dataloaderX = dataloaderX

    def forward(self, input):
        return self.model(input)

    def __call__(self):
        pre = './gradcam_imgs/{}/'.format(self.args.save)
        if not os.path.exists(pre):
            os.mkdir(pre)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        def p1_farward_hook(module, input, output):
            p1_fmap_block.append(output)
        def p1_backward_hook(module, grad_in, grad_out):
            p1_grad_block.append(grad_out[0].detach())

        self.model.eval()
        self.loss.step()

        for key in tqdm(self.dataloaderX.keys()):
            pairs_cam = []
            pairs_feature = []
            for dataX in self.dataloaderX[key]:
                inputs = dataX[0]
                labels = dataX[1]
                p1_fmap_block = []
                p1_grad_block = []

                # forward and backward
                self.model.get_model()._modules.get(self.args.layer_name).register_forward_hook(p1_farward_hook)
                self.model.get_model()._modules.get(self.args.layer_name).register_backward_hook(p1_backward_hook)


                #  input processing
                inputs_source = inputs.squeeze().numpy().transpose(1,2,0)
                inputs_source = cv2.cvtColor(np.uint8(255 * (inputs_source * std + mean)), cv2.COLOR_RGB2BGR)

                # forward
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.model.zero_grad()
                outputs = self.model(inputs)

                # extract feature
                inputs_ = copy.deepcopy(inputs)
                # outputs_ = copy.deepcopy(outputs)
                if self.args.similarly:
                    feature = extract_feature_without_fliphor(inputs_, outputs)
                    pairs_feature.append(feature)

                if '1*CrossEntropy' in self.args.gradcam_loss.split('+'):
                    # one hot process
                    for num in range(len(outputs[-1])):
                        index = np.argmax(outputs[-1][num].cpu().data.numpy())
                        # print(index) # every time test, the predicted result is differet, and it maybe mean different  kind have some same responding
                        one_hot = np.zeros((1, outputs[-1][num].size()[-1]), dtype=np.float32)
                        one_hot[0][index] = 1
                        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
                        # one_hot = torch.sum(one_hot.cuda() * outputs[4][num])
                        one_hot = one_hot.cuda() * outputs[-1][num]
                        outputs[-1][num] = one_hot

                # one_hot.backward(retain_graph=True), or can do ce only
                loss = self.loss(outputs, labels)
                loss.backward(retain_graph=True)

                # get weight
                p1_grad_block = copy.deepcopy(p1_grad_block)

                # get gradcam++ imgs
                p1_cam = cam_maker(p1_grad_block, p1_fmap_block, inputs_source, self.args.height, self.args.width)
                inputs_source = np.float32(inputs_source) / 255
                cam = np.hstack((inputs_source, p1_cam))
                pairs_cam.append(cam)

            # calculation of similarity
            if self.args.similarly:
                features = torch.cat(pairs_feature, 0)
                dist = torch.mm(features, features.t())
                # imgs processing
                for j, cam_img in enumerate(pairs_cam):
                    cam_img = [cv2.putText(cam_img, str(np.array(dist[j].cpu())), (0, self.args.height-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)]

            pair_cam = np.vstack(pairs_cam)
            path = key + '.jpg'
            cv2.imwrite(os.path.join(pre, path), np.uint8(255 * pair_cam))

def extract_feature_without_fliphor(inputs, outputs2):
    features = torch.FloatTensor()
    f2 = outputs2[0].data.cpu()
    ff = torch.FloatTensor(inputs.size(0), f2.size(1)).zero_()
    ff = ff + f2

    fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
    ff = ff.div(fnorm.expand_as(ff))

    features = torch.cat((features, ff), 0)
    return features

def img_loader(args):
    test_transform = transforms.Compose([
        Resize((args.height, args.width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    dataloaderX = defaultdict(list)

    # get path
    data_path = args.datadir
    imgs_path = []
    for i in os.walk(data_path):
        for j in i[2]:
            if os.path.splitext(j)[-1] == '.jpg' or os.path.splitext(j)[-1] == '.png':
                imgs_path.append(os.path.join(i[0], j))

    # img process
    imgs = []
    for i in imgs_path:
        img = default_loader(i)
        imgs.append(test_transform(img).unsqueeze_(0))

    # dataloaderX
    for num, path in enumerate(imgs_path):
        # the element is : {class1: [img ,label], class2: [img ,label]}
        dataloaderX[path.split('/')[-2]].append([imgs[num],torch.tensor([int(path.split('/')[-1].split('_')[0])])])
    print('[INFO]Total {} pairs of img...'.format(len(dataloaderX)))

    return dataloaderX

if __name__=='__main__':
    print('[INFO]model:{}, process img:{}, save as:{}'.format(args.model, args.cam_num, args.save))
    model = model.gradcam_Model(args)
    loss = loss.gradcam_Loss(args)
    if args.load_sel:
        loading(args.load_sel, model, args.nGPU)

    # dataloader
    test_transform = transforms.Compose([
        Resize((args.height, args.width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # dataloaderX
    img_path = args.datadir
    dataloaderX = img_loader(args)

    # get cam and similarity
    grad_cam = GradCam(args = args ,model=model, dataloaderX=dataloaderX, loss=loss)
    mask = grad_cam()
