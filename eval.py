import os
import pprint
from collections import OrderedDict, defaultdict
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import time
from torch import nn, optim
from batch_engine import valid_trainer
from config import argument_parser
from dataset.AttrDataset import MultiModalAttrDataset, get_transform
from loss.CE_loss import *
from models.base_block import *
from tools.function import get_pedestrian_metrics, get_signle_metrics
from tools.utils import time_str, save_ckpt, ReDirectSTD, set_seed, select_gpus

import torch.nn.functional as F

from CLIP.clip import clip
from CLIP.clip.model import *
from tensorboardX import SummaryWriter

import torch.distributed as dist
import torch.nn.parallel
from torch.utils.data.distributed import DistributedSampler
from peft import LoraConfig, get_peft_model,AdaLoraConfig
def main(args):
    ViT_model, ViT_preprocess = clip.load("ViT-B/16", device=device,download_root='/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c0/DATA/yanghaoxiang/VBT/model') 
    
    Event_ViT_model,Event_ViT_preprocess = clip.load("ViT-B/16", device=device,download_root='/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c0/DATA/yanghaoxiang/VBT/model') 
  
    frame_ViT_model,frame_ViT_preprocess = clip.load("ViT-B/16", device=device,download_root='/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c0/DATA/yanghaoxiang/VBT/model') 


    lora_config_ViT_model = LoraConfig(
            r=4,
            lora_alpha=8,
            # target_modules=["out_proj","c_fc","c_proj"], 
            target_modules=["c_fc","c_proj"], 
            #在这里怎么确定需要加的模块？
            lora_dropout=0.01,
            task_type="TOKEN_CLS",
            bias="none" 
    )
    
    ViT_model = get_peft_model(ViT_model,lora_config_ViT_model)
    Event_ViT_model =  get_peft_model(Event_ViT_model,lora_config_ViT_model)
    frame_ViT_model =  get_peft_model(frame_ViT_model,lora_config_ViT_model)
    
    

    ViT_model = ViT_model.float()
    Event_ViT_model = Event_ViT_model.float()
    frame_ViT_model =  frame_ViT_model.float()

    ViT_model = ViT_model.to(args.local_rank)
    Event_ViT_model = Event_ViT_model.to(args.local_rank)
    frame_ViT_model =  frame_ViT_model.to(args.local_rank)

    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)

    valid_tsfm = get_transform(args)[1]

    valid_set = MultiModalAttrDataset(args=args, split=args.valid_split, transform=valid_tsfm)

    valid_sampler = DistributedSampler(valid_set)
    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        sampler=valid_sampler
    )

    model = TransformerClassifier(valid_set.attr_num, attr_words=valid_set.attributes)
    model = torch.nn.parallel.DistributedDataParallel(model.cuda(), find_unused_parameters=True)
    
    checkpoint = torch.load('/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c0/DATA/yanghaoxiang/VBT/poker_frame8_checkpoint/best_checkpoint.pth',map_location='cuda:0')
   
    #可视化
    #checkpoint = torch.load('/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/lidong/VTF_PAR-main/Visualization/VTF/best_checkpoint.pth',map_location='cuda:0')
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    ViT_model.load_state_dict(checkpoint['ViT_model_state_dict'], strict=False)
    Event_ViT_model.load_state_dict(checkpoint['Event_ViT_model_state_dict'],strict=False)
    frame_ViT_model.load_state_dict(checkpoint['frame_ViT_model_state_dict'],strict=False)


    
    criterion = nn.CrossEntropyLoss()
    
    valid_loss, valid_gt, valid_probs = valid_trainer(
        epoch=1,
        model=model,
        ViT_model=ViT_model,
        Event_ViT_model=Event_ViT_model,
        frame_ViT_model=frame_ViT_model,
        valid_loader=valid_loader,
        criterion=criterion,
    )
   
    if args.dataset == 'MARS':
      

        valid_preds = valid_probs.argmax(axis=1)
        valid_gt = valid_gt.argmax(axis=1)
        valid_correct_predictions = (valid_preds == valid_gt).sum()
        valid_accuracy = valid_correct_predictions / len(valid_gt)

        
        
        #######
        print('===>>valid_accuracy = ', valid_accuracy)

    print('===>>Testing Complete...')

        
if __name__ == '__main__':
    parser = argument_parser()
    parser.add_argument('--local_rank', type=int, help='Local rank for distributed training')
    args = parser.parse_args()
    main(args)
    

