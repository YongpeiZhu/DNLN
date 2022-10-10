import os
import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import argparse
import skimage.data
import skimage.io
import skimage.transform
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
from PIL import Image
import cv2
import model
from option import args
from importlib import import_module
class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers
 
    def forward(self, x):
        outputs = {}
        head =self.submodule(x)
        outputs["head"] = head
        # self.head
        # outputs["tail"] = tail
        # for name, module in self.submodule._modules.items():
        #     if "fc" in name: 
        #         x = x.view(x.size(0), -1)
            
        #     x = module(x)
        #     print(name)
        #     if self.extracted_layers is None or name in self.extracted_layers and 'fc' not in name:
        #         outputs[name] = x

        return outputs


def get_picture(pic_name, transform):
    img = skimage.io.imread(pic_name)
    img = skimage.transform.resize(img, (256, 256))
    img = np.asarray(img, dtype=np.float32)
    return transform(img)

def make_dirs(path):
    if os.path.exists(path) is False:
        os.makedirs(path)


def get_feature(img, net):
    # pic_dir = './benchmark/Set5/LR_bicubic/X4/headx4.png'
    # transform = transforms.ToTensor()
    # img = get_picture(pic_dir, transform)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # # 插入维度
    # img = img.unsqueeze(0)

    # img = img.to(device)

    
    # # net = models.resnet101().to(device)
    # module = import_module('model.' + args.model.lower())
    # net = module.make_model(args).to(device="cuda:1")
    # net.load_state_dict(torch.load('../experiment/models/model_x2.pt', **{}), strict=False)
    # exact_list = None
    dst = './feautures'
    therd_size = 256
    exact_list = ["head"]

    myexactor = FeatureExtractor(net, exact_list)
    outs = myexactor(img)
    for k, v in outs.items():
        features = v[0]
        iter_range = features.shape[0]
        for i in range(iter_range):
            #plt.imshow(x[0].data.numpy()[0,i,:,:],cmap='jet')
            if 'fc' in k:
                continue

            feature = features.data.numpy()
            feature_img = feature[i,:,:]
            feature_img = np.asarray(feature_img * 255, dtype=np.uint8)
            
            dst_path = os.path.join(dst, k)
            
            make_dirs(dst_path)
            feature_img = cv2.applyColorMap(feature_img, cv2.COLORMAP_JET)
            if feature_img.shape[0] < therd_size:
                tmp_file = os.path.join(dst_path, str(i) + '_' + str(therd_size) + '.png')
                tmp_img = feature_img.copy()
                tmp_img = cv2.resize(tmp_img, (therd_size,therd_size), interpolation =  cv2.INTER_NEAREST)
                cv2.imwrite(tmp_file, tmp_img)
            
            dst_file = os.path.join(dst_path, str(i) + '.png')
            cv2.imwrite(dst_file, feature_img)

# if __name__ == '__main__':
#     get_feature()

