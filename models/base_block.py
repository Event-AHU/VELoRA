import torch.nn as nn
import torch
import torch.nn.functional as F
from CLIP.clip import clip
from models.vit import *
from CLIP.CoOp import *
import pdb 
from model import ours_model_pretrain
from model.hub_model import OursModelFineTune
from peft import LoraConfig, get_peft_model,AdaLoraConfig
device = "cuda" if torch.cuda.is_available() else "cpu"
from sklearn.neighbors import NearestNeighbors
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from .configml import get_config
import argparse
import os
from .swin_transformer_mtlora import SwinTransformerMTLoRA
from .swin_transformer import SwinTransformer
from .swin_mtl import MultiTaskSwin
from .build import build_model, build_mtl_model
from ptflops import get_model_complexity_info
from thop import profile
import cv2
import matplotlib.pyplot as plt
from ptflops import get_model_complexity_info
# from lora import *

# from lora.model import add_lora_by_layer_names
# from lora.model import LoRAFAParametrization

from PEFT import LoraConfig as loraConfig
from PEFT import get_peft_model as get_model

# from modeling.fusion_part import CRM

class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # 计算查询、键和值
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # 计算注意力分数
        attention_scores = torch.matmul(Q, K.transpose(-2, -1))
        attention_scores = attention_scores / torch.sqrt(torch.tensor(self.input_dim).float())  # 缩放注意力分数
        
        # 应用 softmax 函数获取注意力权重
        attention_weights = self.softmax(attention_scores)
        
        # 使用注意力权重对值进行加权求和
        attended_values = torch.matmul(attention_weights, V)
        
        return attended_values
    

class CrossAttention(nn.Module):
    def __init__(self, in_dim, out_dim,in_q_dim,hid_q_dim):
        super(CrossAttention, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.in_q_dim = in_q_dim #新增
        self.hid_q_dim = hid_q_dim #新增
        # 定义查询、键、值三个线性变换
        self.query = nn.Linear(in_q_dim, hid_q_dim, bias=False) #变化
        self.key = nn.Linear(in_dim, out_dim, bias=False)
        self.value = nn.Linear(in_dim, out_dim, bias=False)
        
    def forward(self, x, y):
        # 对输入进行维度变换，为了方便后面计算注意力分数

        batch_size = x.shape[0]   # batch size
        num_queries = x.shape[1]  # 查询矩阵中的元素个数
        num_keys = y.shape[1]     # 键值矩阵中的元素个数
        x = self.query(x)  # 查询矩阵
        y = self.key(y)    # 键值矩阵
        # 计算注意力分数
        attn_scores = torch.matmul(x, y.transpose(-2, -1)) / (self.out_dim ** 0.5)  # 计算注意力分数，注意力分数矩阵的大小为 batch_size x num_queries x num_keys x num_keys
        attn_weights = F.softmax(attn_scores, dim=-1)  # 对注意力分数进行 softmax 归一化
        # 计算加权和
        V = self.value(y)  # 通过值变换得到值矩阵 V
        output = torch.bmm(attn_weights, V)  # 计算加权和，output 的大小为 batch_size x num_queries x num_keys x out_dim
       
        return output
    
    
       


class TransformerClassifier(nn.Module):
    # 加载冻结A矩阵得Lora到qkv上


    #计算事件图像的稀疏度
    # def calculate_sparsity(event_image):
    #     non_zero_pixels = np.count_nonzero(event_image)
    #     total_pixels = event_image.size
    #     sparsity = non_zero_pixels / total_pixels
    #     return sparsity

           
        
   
  
      
    def __init__(self, attr_num,attr_words, dim=768, pretrain_path='/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c02/DATA/songhaoyu/VELoRA/ViT_checkpoint/jx_vit_base_p16_224-80ecf9dd.pth'):
        super().__init__()
        super().__init__()
        self.attr_num = attr_num
        self.word_embed = nn.Linear(512, dim)    
        # self.visual_embed= nn.Linear(512, dim)#1
        # dim修改为1024
        self.visual_embed= nn.Linear(512, dim)
        #self.vit我们也用CLIP
        # self.vit = OursModelFineTune("/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c02/DATA/yanghaoxiang/VBT/model/pr.pt",base_model = 'vits',num_classes=114,in_chans=2,mask_ratio=0).get()
        
        # 单独训练, 从空参数开始
        self.blocks = nn.Sequential(*vit_base().blocks[-1:])
        self.fuse_blocks = nn.Sequential(*vit_base().blocks[-1:])

        # 单独训练，从ImageNet 预训练权重开始
        # self.vit1 = vit_base()
        # self.vit1.load_param(pretrain_path)
        # # print(self.vit1)
        # # breakpoint()
        # self.blocks = nn.Sequential(*self.vit1.blocks[-1:])

        # self.vit2 = vit_base()
        # self.vit2.load_param(pretrain_path)
        # self.fuse_blocks = nn.Sequential(*self.vit2.blocks[-1:])

        self.norm = vit_base().norm
        #最后一层LoRA装到了MLP上 
        lora_config2 = LoraConfig(
            r=8,
            lora_alpha=16,
            #target_modules=["qkv","fc1","fc2","proj"],
            target_modules=["fc1","fc2"],  #在这里怎么确定需要加的模块？
            lora_dropout=0.01,
            # task_type="TOKEN_CLS",
            bias="none" 
        )
        self.blocks = get_peft_model(self.blocks,lora_config2)
        self.fuse_blocks = get_peft_model(self.fuse_blocks,lora_config2)
        # print(self.blocks)
        # breakpoint()
        
        
        # self.vit = get_peft_model(self.vit, lora_config)  
    
        # for name, param in self.vit.named_parameters():
        #     if "lora_A" in name:  # 参数中包含lora_A被冻结
        #         param.requires_grad = False
             
        
        # 暂时还是先用LOra做最后一层的融合
        # self.blocks = self.vit.blocks[-1:]
        # self.blocks = get_peft_model(self.blocks,lora_config2)
        # self.blocks = self.vit.blocks[-1:]

        # 取base用于多模态的融合
        # self.blocks = vit_base().blocks[-1:]
        # self.blocks = self.vit.blocks[-1:]
        # self.norm = self.vit.norm
        self.weight_layer = nn.ModuleList([nn.Linear(dim, 1) for i in range(self.attr_num)])
        self.remain_weight_layer = nn.ModuleList([nn.Linear(dim, 1) for _ in range(197)])
        self.bn = nn.BatchNorm1d(self.attr_num)
        #self.text = clip.tokenize(attr_words,prompt ="The content of the playing card is ").to(device)#1
        self.text = clip.tokenize(attr_words).to(device)
        self.rgb_embed   = nn.Parameter(torch.zeros(1, 1, dim))
        self.event_embed = nn.Parameter(torch.zeros(1, 1, dim))
        self.frame_embed = nn.Parameter(torch.zeros(1, 1, dim))
        self.tex_embed = nn.Parameter(torch.zeros(1, 1, dim))
        self.self_attention = SelfAttention(dim) 
        # 实例化CrossAttention对象
        self.cross_model = CrossAttention(in_dim=768, out_dim=768, in_q_dim=768, hid_q_dim=768)#除了第一个维度，其他自定
        # self.cross_model.to(device)
        self.CLS_token = nn.Parameter(torch.zeros(1, 10, dim))
        # self.fc1 = nn.Linear(768*10, self.attr_num)
        self.fc1 = nn.Linear(dim*10, self.attr_num)
        # self.fc2 = nn.Linear(768,4)
        # self.fc3 = nn.Linear(4,768)
        # self.mix_dim = 768
        # self.num_head = 12
        # self.CRM = CRM(dim=self.mix_dim, num_heads=self.num_head, miss=cfg.TEST.MISS,
        #                    depth=cfg.MODEL.RE_LAYER)


        
       

    def forward(self, rgb_videos, event_videos, ViT_model,Event_ViT_model,frame_ViT_model):
        rgb_ViT_features=[]
        event_ViT_features=[]
        rgb_frames=[]
        event_frames=[]
        rgb_frame_diffs = []
        event_frame_diffs = []
        frames_features = []
        
        if len(rgb_videos.size())<5 :
            rgb_videos.unsqueeze(1) 
        
        batch_size, num_frames, channels, height, width = rgb_videos.size() 
        rgb_imgs = rgb_videos.view(-1, channels, height, width) 
        event_imgs = event_videos.view(-1, channels, height, width)
        
        #CLIP 提取视频帧特征
        
        for img in rgb_imgs:     #img的数量是由batch_size*frame_num
            
            rgb_frames.append(img)
            img = img.unsqueeze(0)
            img = img.to(device)
            # input_size = (3, 224, 224) 
            # ######FLOPS#####
        
            # breakpoint()
            # macs, params = get_model_complexity_info(self.blocks,input_size,as_strings=True, print_per_layer_stat=True)
            # print(f"模型 FLOPs: {macs}")

            

            ##########
            rgb_ViT_features.append(ViT_model.encode_image(img).squeeze(0))
           
            
            
            #消融2（大模型换常规模
            #rgb_ViT_features.append(self.vit(img).squeeze(0))
        # 提取图片帧差信息
       
        groups = [rgb_frames[i:i + 8] for i in range(0, len(rgb_frames), 8)]


        for i, group in enumerate(groups):
            for j in range(1, len(group)):
                diff = torch.abs(group[j-1]-group[j])
                from torchvision.utils import save_image
                breakpoint()
                diff = diff.unsqueeze(0)
                save_image(diff, 'event.png')
                rgb_frame_diffs.append(diff)
           

    
    
       
            
        rgb_ViT_image_features = torch.stack(rgb_ViT_features).to(device).float()
         
        #同样也用CLIP提取视频帧特征
        for img in event_imgs:
            event_frames.append(img)
            img = img.unsqueeze(0)
            img = img.to(device)
            
           
            #消融2（大模型换常规模型）
            # breakpoint()
            # img = torch.stack([img[:,0], torch.sum(img[:,1:],dim=1)/2],dim=1)
            # event_ViT_features.append(ViT_model.encode_image(img).squeeze(0)) 
            event_ViT_features.append(Event_ViT_model.encode_image(img).squeeze(0)) 

        # 提取事件图片帧差信息
        event_groups = [event_frames[i:i + 8] for i in range(0, len(event_frames), 8)]



        for i, event_group in enumerate(event_groups):
            for j in range(1, len(event_group)):
                diff = torch.abs(event_group[j-1]-event_group[j])
                event_frame_diffs.append(diff)
           
       
        rgb_frame_diffs = [tensor.to(device) for tensor in rgb_frame_diffs]
        event_frame_diffs = [tensor.to(device) for tensor in event_frame_diffs]
        # image和event帧四帧相加
        sum_frame = [a + b for a, b in zip(rgb_frame_diffs, event_frame_diffs)]
        
        #四帧输出
        for i, each in enumerate(sum_frame):
            each = each.unsqueeze(0)
            each = each.to(device)
            frames_features.append(frame_ViT_model.encode_image(each).squeeze(0)) 

        frames_ViT_image_features = torch.stack(frames_features).to(device).float()  # torch.Size([4, 197, 512])

        event_ViT_image_features = torch.stack(event_ViT_features).to(device).float()  #torch.Size([5, 197, 512])
        

        
        
        _, token_num, visual_dim = rgb_ViT_image_features.size()
      
        frames_ViT_image_features = frames_ViT_image_features.view(batch_size,num_frames-1,token_num,visual_dim)

        rgb_ViT_image_features   = rgb_ViT_image_features.view(batch_size, num_frames, token_num, visual_dim) 
        event_ViT_image_features = event_ViT_image_features.view(batch_size, num_frames, token_num, visual_dim)
        ##  这个是frame出来的特征
        frames_ViT_image_features = self.visual_embed(torch.mean(frames_ViT_image_features,dim=1))

        rgb_ViT_image_features = self.visual_embed(torch.mean(rgb_ViT_image_features, dim=1))  # torch.Size([1, 197, 768])
        event_ViT_image_features = self.visual_embed(torch.mean(event_ViT_image_features, dim=1)) 
        
         
        ############################### 互相重构，损失约束################################
        frame_embed = frames_ViT_image_features + self.frame_embed 
        rgb_embed   = rgb_ViT_image_features + self.rgb_embed       #torch.Size([4, 197, 768])
        event_embed = event_ViT_image_features + self.event_embed    #torch.Size([4, 197, 768])


        for b_c, blk in enumerate(self.blocks):
            RTE_feature = blk(rgb_embed)
            ETR_feature = blk(event_embed)

        RTE_feature = self.norm(RTE_feature)
        ETR_feature = self.norm(ETR_feature)
        # RTE_feature = RTE_CON_Transform(rgb_embed)
        # ETR_feature = ETR_CON_Transform(event_embed)

        # RTE_feature = ViT_model.visual.transformer.resblocks[-1:](rgb_embed)
        # ETR_feature = ViT_model.visual.transformer.resblocks[-1:](event_embed)
       
        
        loss_rgb =  nn.MSELoss()(ETR_feature, rgb_embed)
        loss_event =  nn.MSELoss()(RTE_feature, event_embed)

        loss_fn = loss_event + loss_rgb
        
        ###将三个特征再dim=1维度上拼接 #####################################
        x = torch.cat([frame_embed, rgb_embed,event_embed], dim=1) 
        # # #######将两个个特征在dim=1维度上拼接
        # x = torch.cat([rgb_embed, event_embed], dim=1)    #torch.Size([4, 394, 768])
        
       
        ########多模态特征融合########################################
        #####1.backbone融合 ###################################
       
        

        for b_c, blk in enumerate(self.fuse_blocks):
            x = blk(x)


        # 融合不共有vit——base
        # x =ETR_CON_Transform(x) 
        


        #####恢复形状走自注意力阶段 #####################################################
        frametoken_features      = x[:, :rgb_embed.shape[1], :]     #torch.Size([4, 114, 768])
        rgb_imgtoken_features   = x[:, rgb_embed.shape[1]:rgb_embed.shape[1]+rgb_embed.shape[1], :]    
        event_imgtoken_features = x[:, rgb_embed.shape[1]+rgb_embed.shape[1]:, :] 

        # #######不加lora,完全微调##########
        # vis_event_tokens = torch.cat((rgb_imgtoken_features,event_imgtoken_features), dim=1)
        # vis_event_tokens = self.self_attention(vis_event_tokens)
        
        # enhanced_vis_tokens   = vis_event_tokens[:, :197, :]
        # enhanced_event_tokens = vis_event_tokens[:, 197:, :] 
            
        # enhanced_features = enhanced_vis_tokens+enhanced_event_tokens
             
        # # cross_output1 = self.cross_model(enhanced_vis_tokens, enhanced_event_tokens) 
        # # cross_output2 = self.cross_model(enhanced_event_tokens, enhanced_vis_tokens)
        # # cross_features = cross_output1 + cross_output2
        
        # # final_features = enhanced_features+cross_features
        # # final_features = vis_event_tokens+
        # l = enhanced_features.shape[1]
        # final_features_CLS = enhanced_features[:, l-10:, :]

        
        # output_feat = final_features_CLS.view(final_features_CLS.shape[0], -1)
     
        
        # # 全连接层
        # output_feat = self.fc1(output_feat)
        

        # # softmax归一化
        # logits = F.log_softmax(output_feat, dim=1) 
        
        # return logits,loss_fn
        
        # ###################lora spe 微调 ##########################
        # vis_event_tokens = torch.cat((rgb_imgtoken_features,event_imgtoken_features), dim=1)
        # vis_event_tokens = self.self_attention(vis_event_tokens)
        # enhanced_vis_tokens   = vis_event_tokens[:, :197, :]
        # enhanced_event_tokens = vis_event_tokens[:, 197:, :] 
        # enhanced_features = enhanced_vis_tokens+enhanced_event_tokens
       
        # l = enhanced_features.shape[1]
        # final_features_CLS = enhanced_features[:, l-10:, :]
        # output_feat = final_features_CLS.view(final_features_CLS.shape[0], -1)
        # output_feat = self.fc1(output_feat)
        # logits = F.log_softmax(output_feat, dim=1) 
        # return logits,loss_fn
        







       #### origin ###################
        
        
        event_embed = Event_ViT_model.visual.transformer.resblocks[-1](event_embed)
        
      

        #######################################
          
        x = Event_ViT_model.visual.transformer.resblocks[-1].mlp.c_fc.lora_dropout['default'](event_imgtoken_features)
        x = Event_ViT_model.visual.transformer.resblocks[-1].mlp.c_fc.lora_A['default'](x)
        x = Event_ViT_model.visual.transformer.resblocks[-1].mlp.c_fc.lora_B['default'](x)
        x = Event_ViT_model.visual.transformer.resblocks[-1].mlp.gelu(x)
        x = Event_ViT_model.visual.transformer.resblocks[-1].mlp.c_proj.lora_dropout['default'](x)
        x = Event_ViT_model.visual.transformer.resblocks[-1].mlp.c_proj.lora_A['default'](x)
        x = Event_ViT_model.visual.transformer.resblocks[-1].mlp.c_proj.lora_B['default'](x)
        
        #############走qkv#################
        fused_event_token = event_imgtoken_features+x
        final_event_feature = event_embed+fused_event_token  

        # final_event_feature = self.norm(final_event_feature)
        # final_event_feature = self.fc3(event_embed)
        
        ####################这里让image最后一个transform结合多模态辅助信息###########################
        rgb_embed = ViT_model.visual.transformer.resblocks[-1](rgb_embed)

        ##################################
        
        x = ViT_model.visual.transformer.resblocks[-1].mlp.c_fc.lora_dropout['default'](rgb_imgtoken_features)
        x = ViT_model.visual.transformer.resblocks[-1].mlp.c_fc.lora_A['default'](x)
        x = ViT_model.visual.transformer.resblocks[-1].mlp.c_fc.lora_B['default'](x)
        x = ViT_model.visual.transformer.resblocks[-1].mlp.gelu(x)
        x = ViT_model.visual.transformer.resblocks[-1].mlp.c_proj.lora_dropout['default'](x)
        x = ViT_model.visual.transformer.resblocks[-1].mlp.c_proj.lora_A['default'](x)
        x = ViT_model.visual.transformer.resblocks[-1].mlp.c_proj.lora_B['default'](x)

        fused_vis_token = rgb_imgtoken_features +x 

        final_image_feature = fused_vis_token + rgb_embed

       
        # final_image_feature = ViT_model.visual.transformer.ln_2(final_image_feature)
        
        ############################################################
        ## self-attention and cross-attention module 
        ############################################################
        vis_event_tokens = torch.cat((final_image_feature,final_event_feature), dim=1)
        vis_event_tokens = self.self_attention(vis_event_tokens)
        
        enhanced_vis_tokens   = vis_event_tokens[:, :197, :]
        enhanced_event_tokens = vis_event_tokens[:, 197:, :] 
            
        enhanced_features = enhanced_vis_tokens+enhanced_event_tokens
             
        # cross_output1 = self.cross_model(enhanced_vis_tokens, enhanced_event_tokens) 
        # cross_output2 = self.cross_model(enhanced_event_tokens, enhanced_vis_tokens)
        # cross_features = cross_output1 + cross_output2
        
        # final_features = enhanced_features+cross_features
        # final_features = vis_event_tokens+
        l = enhanced_features.shape[1]
        final_features_CLS = enhanced_features[:, l-10:, :]

        
        output_feat = final_features_CLS.view(final_features_CLS.shape[0], -1)
     
        
        # 全连接层
        output_feat = self.fc1(output_feat)
        

        # softmax归一化
        logits = F.log_softmax(output_feat, dim=1) 
        
        return logits,loss_fn
        # return logits
