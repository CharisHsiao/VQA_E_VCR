"""
ok so I lied. it's not a detector, it's the resnet backbone
"""

import torch
import torch.nn as nn
import torch.nn.parallel
# from torchvision.models import resnet

# from utils.pytorch_misc import Flattener
# from torchvision.layers import ROIAlign
# import torch.utils.model_zoo as model_zoo
from config import USE_IMAGENET_PRETRAINED
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


def _load_resnet_places365(pretrained=True,arch='resnet50'):
    # load the pre-trained weights
    model_file = '%s_places365.pth.tar' % arch   
    if not os.access(model_path, os.W_OK):
        weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
        os.system('wget ' + weight_url+ ' ' + model_folder)

    
    backbone = models.__dict__[arch](self.num_classes) # model: # torchvision.models.resnet.resnet50
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    backbone.load_state_dict(state_dict)
    # model.eval()
    
    
    # don't know the meaning of the code followed.
    for i in range(2, 4):
        getattr(backbone, 'layer%d' % i)[0].conv1.stride = (2, 2)
        getattr(backbone, 'layer%d' % i)[0].conv2.stride = (1, 1)
    # use stride 1 for the last conv4 layer (same as tf-faster-rcnn)
    backbone.layer4[0].conv2.stride = (1, 1)
    backbone.layer4[0].downsample[0].stride = (1, 1)

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
        super(SimpleDetector, self).__init__()
        # huge thx to https://github.com/ruotianluo/pytorch-faster-rcnn/blob/master/lib/nets/resnet_v1.py
        self.num_classes = num_classes
        model_file = '%s_%s.pth.tar' % (arch,dataset)
        backbone = _load_resnet_places365(pretrained=pretrained,arch) if USE_PLACE365_PRETRAINED else _load_resnet(
            pretrained=pretrained) 

        
        # load the class label
        file_name = 'categories_places365.txt'
        if not os.access(file_name, os.W_OK):
            synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
            os.system('wget ' + synset_url)
        classes = list()
        with open(file_name) as class_file:
            for line in class_file:
                classes.append(line.strip().split(' ')[0][3:])
        classes = tuple(classes)
        # you do it later
        please switch to other part
        multi attention
        ？？？？  not this one
        
        

    def forward(self,
                images: torch.Tensor
                ):
        """
        :param images: [batch_size, 3, im_height, im_width]
        :return: images [batch_size,7,7,dim]
        """
        return r
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        # [batch_size, 2048, im_height // 32, im_width // 32
        img_feats = self.backbone(images)
        box_inds = box_mask.nonzero()
        assert box_inds.shape[0] > 0
        rois = torch.cat((
            box_inds[:, 0, None].type(boxes.dtype),
            boxes[box_inds[:, 0], box_inds[:, 1]],
        ), 1)

        # Object class and segmentation representations
        roi_align_res = self.roi_align(img_feats, rois)
        if self.mask_upsample is not None:
            assert segms is not None
            segms_indexed = segms[box_inds[:, 0], None, box_inds[:, 1]] - 0.5
            roi_align_res[:, :self.mask_dims] += self.mask_upsample(segms_indexed)


        post_roialign = self.after_roi_align(roi_align_res)

        # Add some regularization, encouraging the model to keep giving decent enough predictions
        obj_logits = self.regularizing_predictor(post_roialign)
        obj_labels = classes[box_inds[:, 0], box_inds[:, 1]]
        cnn_regularization = F.cross_entropy(obj_logits, obj_labels, size_average=True)[None]

        feats_to_downsample = post_roialign if self.object_embed is None else torch.cat((post_roialign, self.object_embed(obj_labels)), -1)
        roi_aligned_feats = self.obj_downsample(feats_to_downsample)

        # Reshape into a padded sequence - this is expensive and annoying but easier to implement and debug...
        obj_reps = pad_sequence(roi_aligned_feats, box_mask.sum(1).tolist())
        return {
            'obj_reps_raw': post_roialign,
            'obj_reps': obj_reps,
            'obj_logits': obj_logits,
            'obj_labels': obj_labels,
            'cnn_regularization_loss': cnn_regularization
        }
