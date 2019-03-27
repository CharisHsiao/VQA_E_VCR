"""
ok so I lied. it's not a detector, it's the resnet backbone
"""

import torch
import torch.nn as nn
import torch.nn.parallel
from torchvision.models import resnet

# from utils.pytorch_misc import Flattener
# from torchvision.layers import ROIAlign
# import torch.utils.model_zoo as model_zoo
from config import USE_PLACE365_PRETRAINED

# from utils.pytorch_misc import pad_sequence


from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image

from functools import partial
import pickle



def _load_resnet(pretrained=True):
    # huge thx to https://github.com/ruotianluo/pytorch-faster-rcnn/blob/master/lib/nets/resnet_v1.py
    backbone = resnet.resnet50(pretrained=False,)
    if pretrained:
        backbone.load_state_dict(model_zoo.load_url(
            'https://s3.us-west-2.amazonaws.com/ai2-rowanz/resnet50-e13db6895d81.th'))
    for i in range(2, 4):
        getattr(backbone, 'layer%d' % i)[0].conv1.stride = (2, 2)
        getattr(backbone, 'layer%d' % i)[0].conv2.stride = (1, 1)
    return backbone


def _load_resnet_places365(arch='resnet50',pretrained=True):
    # load the pre-trained weights
    model_file = '%s_places365.pth.tar' % arch   
    if not os.access(model_file, os.W_OK):
        # print('not found,downloading')
        weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
        os.system('wget ' + weight_url)


    
    backbone = models.__dict__[arch](num_classes=365) # model: # torchvision.models.resnet.resnet50
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    backbone.load_state_dict(state_dict)
    # model.eval()
    
    
    # don't know the meaning of the code followed.
#     for i in range(2, 4):
#         getattr(backbone, 'layer%d' % i)[0].conv1.stride = (2, 2)
#         getattr(backbone, 'layer%d' % i)[0].conv2.stride = (1, 1)
#     # use stride 1 for the last conv4 layer (same as tf-faster-rcnn)
#     backbone.layer4[0].conv2.stride = (1, 1)
#     backbone.layer4[0].downsample[0].stride = (1, 1)

    # # Make batchnorm more sensible
    # for submodule in backbone.modules():
    #     if isinstance(submodule, torch.nn.BatchNorm2d):
    #         submodule.momentum = 0.01

    return backbone


class SimpleExtractor(nn.Module):
    def __init__(self,pretrained=True, num_classes=365, arch='resnet50'):
        """
        :param average_pool: whether or not to average pool the representations
        :param pretrained: Whether we need to load from scratch
        :param semantic: Whether or not we want to introduce the mask and the class label early on (default Yes)
        """
        super(SimpleExtractor, self).__init__()
        # huge thx to https://github.com/ruotianluo/pytorch-faster-rcnn/blob/master/lib/nets/resnet_v1.py
        self.num_classes = num_classes
        backbone = _load_resnet_places365(arch,pretrained=pretrained) if USE_PLACE365_PRETRAINED else _load_resnet(
            pretrained=pretrained) 
        
        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4
        )
        
        # self.whole_backbone = backbone
        # load the class label
        file_name = 'categories_places365.txt'
        if not os.access(file_name, os.W_OK):
            synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
            os.system('wget ' + synset_url)
        self.classes = list()
        with open(file_name) as class_file:
            for line in class_file:
                self.classes.append(line.strip().split(' ')[0][3:])
        self.classes = tuple(self.classes)
        
        

    def forward(self,images: torch.Tensor):
        """
        :param images: [batch_size, 3, im_height, im_width]
        :return: img_feats:[batch_size, 2048, im_height // 32, im_width // 32]
        :return: 
        """
        # [batch_size, 2048, im_height // 7, im_width // 7]
        
        images = F.interpolate(images, size=(224,224), scale_factor=None, mode='bilinear', align_corners=None)
        img_feats = self.backbone(images)
        # the prediction of scene classification is absent here 
        # self.whole_backbone
        # logit = self.whole_backbone(images)  # torch.Size([batch_size,365])
        # h_x = F.softmax(logit, 1).data.squeeze()  #  torch.Size([batch_size,365])
        # probs, idx = h_x.sort(0, True)
        # for i in range(probs.shape[0]):
        #   for j in range(0, 5):
        #        print('{:.3f} -> {}'.format(probs[i][j], self.classes[idx[i][j]]))
        
        
        # return img_feats,probs
        return img_feats
