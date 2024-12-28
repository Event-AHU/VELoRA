import os
import pprint
from collections import OrderedDict, defaultdict
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import time
from torch import nn, optim
from batch_engine import valid_trainer, batch_trainer
from config import argument_parser
from dataset.AttrDataset import MultiModalAttrDataset, get_transform
from loss.CE_loss import *
from models.base_block import *
from tools.function import get_pedestrian_metrics, get_signle_metrics
from tools.utils import time_str, save_ckpt, ReDirectSTD, set_seed, select_gpus
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler, make_scheduler
from peft import LoraConfig, get_peft_model,AdaLoraConfig
from thop import profile
import torch.nn.functional as F
import logging
from CLIP.clip import clip
from CLIP.clip.model import *
from tensorboardX import SummaryWriter
#from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.nn.parallel
from torch.utils.data.distributed import DistributedSampler
from model import ours_model_pretrain
from ptflops import get_model_complexity_info




class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)




def main(args):

    # IMAGE
    ViT_model, ViT_preprocess = clip.load("ViT-B/16", device=device,download_root='/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c0/DATA/yanghaoxiang/VBT/model') 
  
    #模型参数不共享
    Event_ViT_model,Event_ViT_preprocess = clip.load("ViT-B/16", device=device,download_root='/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c0/DATA/yanghaoxiang/VBT/model') 
  
    frame_ViT_model,frame_ViT_preprocess = clip.load("ViT-B/16", device=device,download_root='/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c0/DATA/yanghaoxiang/VBT/model') 
    
    # 重构和融合之间不共享模型参数  
    # ETR_CON, ETR_CON_preprocess = clip.load("ViT-B/16", device=device,download_root='/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c0/DATA/yanghaoxiang/VBT/model') 
    
    # ETR_CON_Transform = ETR_CON.visual.transformer.resblocks[-1:]

    # RTE_CON, RTE_CON_preprocess = clip.load("ViT-B/16", device=device,download_root='/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c0/DATA/yanghaoxiang/VBT/model') 
    
    # RTE_CON_Transform = RTE_CON.visual.transformer.resblocks[-1:]
    #最后一层暂时还是使用LORA
    # lora_config_ViT_model = LoraConfig(
    #         r=4,
    #         lora_alpha=8,
    #         target_modules=["out_proj","c_fc","c_proj"], 
    #         # target_modules=["c_fc","c_proj"], 
    #         #在这里怎么确定需要加的模块？
    #         lora_dropout=0.01,
    #         task_type="TOKEN_CLS",
    #         bias="none" 
    # )
    

    lora_config_ViT_model = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["c_fc","c_proj"], 
            lora_dropout=0.01,
            task_type="TOKEN_CLS",
            bias="none" 
    )
    

  
    

    
    # ViT_model = get_peft_model(ViT_model,lora_config_ViT_model)
    # ViT_model = get_peft_model(ViT_model,lora_config_ViT_model)
    
    # for name, param in ViT_model.named_parameters():
    #     if "lora_A" in name:  # 参数中包含lora_A被冻结
    #         param.requires_grad = False


    # Event_ViT_model =  get_peft_model(Event_ViT_model,lora_config_ViT_model)
    # frame_ViT_model =  get_peft_model(frame_ViT_model,lora_config_ViT_model)
    
   


    trainable_params = 0
    all_param = 0
    for _, param in ViT_model.named_parameters():
         all_param += param.numel()
         if param.requires_grad:
             trainable_params += param.numel()
                
    print(
        f" VIT trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
        )  
        
    
    ViT_model = ViT_model.float()
    Event_ViT_model = Event_ViT_model.float()
    frame_ViT_model = frame_ViT_model.float()

    # RTE_CON_Transform = RTE_CON_Transform.float()
    # ETR_CON_Transform = ETR_CON_Transform.float()
    
    ViT_model = ViT_model.to(args.local_rank)
    Event_ViT_model = Event_ViT_model.to(args.local_rank)
    frame_ViT_model = frame_ViT_model.to(args.local_rank)

    # RTE_CON_Transform = RTE_CON_Transform.to(args.local_rank)
    # ETR_CON_Transform = ETR_CON_Transform.to(args.local_rank)
    # # 初始化进程组
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)  # 设置当前设备
    log_dir = os.path.join('logs', "/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c0/DATA/yanghaoxiang/VBT")
   
    tb_writer = SummaryWriter('/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c0/DATA/yanghaoxaing/VBT/CaptionCLIP-ViT-B/tensorboardX/exp')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    
    stdout_file = os.path.join(log_dir, f'stdout_{time_str()}.txt')

    if args.redirector:
        print('redirector stdout')
        ReDirectSTD(stdout_file, 'stdout', False)

    pprint.pprint(OrderedDict(args.__dict__))

    print('-' * 60)
    select_gpus(args.gpus)

    print(f'train set: {args.dataset} {args.train_split}, test set: {args.valid_split}')

    
    train_tsfm, valid_tsfm = get_transform(args)

    train_set = MultiModalAttrDataset(args=args, split=args.train_split, transform=train_tsfm)

    # 使用DistributedSampler
    train_sampler = DistributedSampler(train_set)
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.batchsize,
        shuffle=(train_sampler is None),
        num_workers=8,
        pin_memory=True,
        sampler=train_sampler
    )

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
    
    labels = train_set.label
    sample_weight = labels.mean(0)
    
    model = TransformerClassifier(train_set.attr_num, attr_words=train_set.attributes)
    # 使用DistributedDataParallel包装模型
    # model = torch.nn.parallel.DistributedDataParallel(model.cuda())
    model = torch.nn.parallel.DistributedDataParallel(model.cuda(), find_unused_parameters=True)
    
    criterion = nn.CrossEntropyLoss()

    lr = args.lr
    epoch_num = args.epoch
    start_epoch = 1
    #optimizer = optim.Adam(model.parameters(), lr=lr)
    params_to_update = list(model.parameters()) + list(ViT_model.parameters())+list(Event_ViT_model.parameters())+list(frame_ViT_model.parameters())
    # params_to_update = list(model.parameters()) + list(ViT_model.parameters())
    optimizer = optim.AdamW(params_to_update, lr=lr, weight_decay=1e-4)
    scheduler = create_scheduler(optimizer, num_epochs=epoch_num, lr=lr)

    trainer(args=args,
            epoch=epoch_num,
            model=model,
            ViT_model=ViT_model,
            Event_ViT_model = Event_ViT_model,
            frame_ViT_model = frame_ViT_model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            path=log_dir,
            tb_writer=tb_writer,
            start_epoch=start_epoch,
            save_interval=args.save_interval)


def trainer(args, epoch, model, ViT_model, Event_ViT_model , frame_ViT_model,train_loader, valid_loader, criterion, optimizer, scheduler, path,
            tb_writer, start_epoch, save_interval):
    max_ma, max_acc, max_f1 = 0, 0, 0
    start = time.time()
    best_accuracy = 0
    for i in range(start_epoch, epoch + 1):
        scheduler.step(1)#固定lr
        train_loss, train_gt, train_probs = batch_trainer(
            epoch=i,
            model=model,
            ViT_model=ViT_model,
            Event_ViT_model=Event_ViT_model,
            frame_ViT_model = frame_ViT_model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer
        )

        valid_loss, valid_gt, valid_probs = valid_trainer(
            epoch=epoch,
            model=model,
            ViT_model=ViT_model,
            Event_ViT_model=Event_ViT_model,
            frame_ViT_model = frame_ViT_model,
            valid_loader=valid_loader,
            criterion=criterion,
        )

        if args.dataset == 'MARS':
            train_preds = train_probs.argmax(axis=1)
            train_gt = train_gt.argmax(axis=1)
            correct_predictions = (train_preds == train_gt).sum()
            train_accuracy = correct_predictions / len(train_gt)
            print('===>>train_accuracy = ', train_accuracy)

            valid_preds = valid_probs.argmax(axis=1)
            valid_gt = valid_gt.argmax(axis=1)
            valid_correct_predictions = (valid_preds == valid_gt).sum()
            valid_accuracy = valid_correct_predictions / len(valid_gt)
            print('===>>valid_accuracy = ', valid_accuracy)

        # Save checkpoint
        if i % save_interval == 0:
            ckpt_path = os.path.join("/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c0/DATA/yanghaoxiang/VBT/poker_FA_checkpoint", f'checkpoint_{i}.pth')
            save_checkpoint(model,ViT_model,Event_ViT_model, frame_ViT_model,optimizer, epoch=i, path=ckpt_path)
        
        # 如果当前精度高于最高精度值，则保存当前checkpoint
        if valid_accuracy > best_accuracy:
            best_accuracy = valid_accuracy
            best_ckpt_path = os.path.join("/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c0/DATA/yanghaoxiang/VBT/poker_FA_checkpoint", 'best_checkpoint.pth')
            save_checkpoint(model, ViT_model,Event_ViT_model, frame_ViT_model,optimizer, epoch=i, path=best_ckpt_path)

    end = time.time()
    elapsed = end - start
    print('===>>Elapsed Time: [%.2f h %.2f m %.2f s]' % (elapsed // 3600, (elapsed % 3600) // 60, (elapsed % 60)))

    print('===>>Training Complete...')

    
def save_checkpoint(model,ViT_model, Event_ViT_model,frame_ViT_model,optimizer, epoch, path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'ViT_model_state_dict': ViT_model.state_dict(),
        'Event_ViT_model_state_dict':Event_ViT_model.state_dict(),
        'frame_ViT_model_state_dict':frame_ViT_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, path)
    

if __name__ == '__main__':
    parser = argument_parser()
    parser.add_argument('--save_interval', type=int, default=10, help='Interval for saving checkpoints')
    parser.add_argument('--local_rank', type=int, help='Local rank for distributed training')
    args = parser.parse_args()
    main(args)