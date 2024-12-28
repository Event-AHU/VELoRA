# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import time
import json
import random
import argparse
import datetime
import numpy as np
#import matplotlib.pyplot as plt

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from config import get_config
from build import build_model, build_mtl_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, load_pretrained, save_checkpoint, NativeScalerWithGradNormCount, auto_resume_helper

from mtl_loss_schemes import MultiTaskLoss, get_loss
from evaluation.evaluate_utils import PerformanceMeter, get_output
from ptflops import get_model_complexity_info
from models.lora import mark_only_lora_as_trainable

try:
    import wandb
    wandb_available = True
except ImportError:
    print("Warning: wandb library not found. Logging is disabled.")
    wandb_available = False

from swin_transformer_mtlora import SwinTransformerMTLoRA
from swin_transformer import SwinTransformer
from swin_mtl import MultiTaskSwin


def parse_option():
    parser = argparse.ArgumentParser(
        'Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True,
                        metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int,
                        help="batch size for single GPU")
    parser.add_argument('--ckpt-freq', type=int, default=5,
                        help="checkpoint saving frequency")
    parser.add_argument('--eval-freq', type=int, default=5,
                        help="model evaluation frequency")
    parser.add_argument('--epochs', type=int, default=300,
                        help="number of epochs to train")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true',
                        help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int,
                        help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true',
                        help='Disable pytorch amp')
    parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used (deprecated!)')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--name', type=str, help='override model name')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true',
                        help='Test throughput only')
    # distributed training
    parser.add_argument("--local_rank", type=int, default=0,
                        help='local rank for DistributedDataParallel')
    parser.add_argument("--local-rank", type=int, default=0,
                        help='local rank for DistributedDataParallel')

    # for acceleration
    parser.add_argument('--fused_window_process', action='store_true',
                        help='Fused window shift & window partition, similar for reversed part.')
    parser.add_argument('--fused_layernorm',
                        action='store_true', help='Use fused layernorm.')
    # overwrite optimizer in config (*.yaml) if specified, e.g., fused_adam/fused_lamb
    parser.add_argument('--optim', type=str,
                        help='overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.')

    # MTL Config
    parser.add_argument('--tasks', type=str, default='depth',
                        help='List of tasks to run in MTL setup.')
    parser.add_argument(
        '--nyud', type=str, help='specify the path to load NYUD, replaces --data-path')
    parser.add_argument(
        '--pascal', type=str, help='specify the path to load PASCAL, replaces --data-path and --nyud')
    parser.add_argument('--eval-training-freq', type=int,
                        help='calculate performance score on the training dataset')
    parser.add_argument('--resume-backbone',
                        help='resume checkpoint into the backbone')
    parser.add_argument('--freeze-backbone',
                        action='store_true', help='Freeze encoder layers.')

    parser.add_argument('--skip_initial_validation', action='store_true',
                        help='Skip running validation at the start')
    parser.add_argument('--decoder_map', type=str,
                        help='Path to JSON file containing the type of decoder heads')
    parser.add_argument('--skip_decoder', action='store_true',
                        help='Skip loading decoder head weights')
    parser.add_argument('--disable_wandb', action='store_true',
                        help='Disable wandb logging.')
    parser.add_argument('--run_name', type=str,
                        help='wandb run name')
    parser.add_argument('--no_eval_50', action='store_false',
                        help='Disable the iniital eval at 50 epochs.')
    args = parser.parse_args()

    config = get_config(args)

    return args, config



def build_model(config, is_pretrain=False):
    model_type = config.MODEL.TYPE

    # accelerate layernorm
    if config.FUSED_LAYERNORM:
        try:
            import apex as amp
            layernorm = amp.normalization.FusedLayerNorm
        except:
            layernorm = None
            print("To use FusedLayerNorm, please install apex.")
    else:
        import torch.nn as nn
        layernorm = nn.LayerNorm

    if model_type == 'swin':
        if config.MODEL.MTLORA.ENABLED:
            model = SwinTransformerMTLoRA(img_size=config.DATA.IMG_SIZE,
                                          patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                          in_chans=config.MODEL.SWIN.IN_CHANS,
                                          num_classes=config.MODEL.NUM_CLASSES,
                                          embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                          depths=config.MODEL.SWIN.DEPTHS,
                                          num_heads=config.MODEL.SWIN.NUM_HEADS,
                                          window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                          mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                          qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                          qk_scale=config.MODEL.SWIN.QK_SCALE,
                                          drop_rate=config.MODEL.DROP_RATE,
                                          drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                          ape=config.MODEL.SWIN.APE,
                                          norm_layer=layernorm,
                                          patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                          use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                          fused_window_process=config.FUSED_WINDOW_PROCESS,
                                          tasks=config.TASKS,
                                          mtlora=config.MODEL.MTLORA)
        else:
            model = SwinTransformer(img_size=config.DATA.IMG_SIZE,
                                    patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                    in_chans=config.MODEL.SWIN.IN_CHANS,
                                    num_classes=config.MODEL.NUM_CLASSES,
                                    embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                    depths=config.MODEL.SWIN.DEPTHS,
                                    num_heads=config.MODEL.SWIN.NUM_HEADS,
                                    window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                    mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                    qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                    qk_scale=config.MODEL.SWIN.QK_SCALE,
                                    drop_rate=config.MODEL.DROP_RATE,
                                    drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                    ape=config.MODEL.SWIN.APE,
                                    norm_layer=layernorm,
                                    patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                    use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                    fused_window_process=config.FUSED_WINDOW_PROCESS)
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model


def build_mtl_model(backbone, config):
    model = MultiTaskSwin(backbone, config)
    return model

def main(config):
    # dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(
    #     config)

   
    # teacher = Nonet()
    model = build_model(config)
    
    model = build_mtl_model(model, config)

    # n_parameters = sum(p.numel() for p in model.parameters())
   
    # model.cuda()
if __name__ == '__main__':
    args, config = parse_option()
    
    if config.AMP_OPT_LEVEL:
        print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    
    
    model = build_model(config)
    model = build_mtl_model(model, config)
    x = torch.randn(4,196,768)
    x=model(x)