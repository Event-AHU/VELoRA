# VELoRA

**VELoRA: A Low-Rank Adaptation Approach for Efficient RGB-Event based Recognition**, 
  Lan Chen, Haoxiang Yang, Pengpeng Shao, Haoyu Song, Xiao Wang*, Zhicheng Zhao, Yaowei Wang, Yonghong Tian 
  [[Paper]()] 



### :dart: Abstract 
Pattern recognition leveraging both RGB and Event cameras can significantly enhance performance by deploying deep neural networks that utilize a fine-tuning strategy. Inspired by the successful application of large models, the introduction of such large models can also be considered to further enhance the performance of multi-modal tasks. However, fully fine-tuning these models leads to inefficiency and lightweight fine-tuning methods such as LoRA and Adapter have been proposed to achieve a better balance between efficiency and performance. To our knowledge, there is currently no work that has conducted parameter-efficient fine-tuning (PEFT) for RGB-Event recognition based on pre-trained foundation models. To address this issue, this paper proposes a novel PEFT strategy to adapt the pre-trained foundation vision models for the RGB-Event-based classification. Specifically, given the RGB frames and event streams, we extract the RGB and event features based on the vision foundation model ViT with a modality-specific LoRA tuning strategy. The frame difference of the dual modalities is also considered to capture the motion cues via the frame difference backbone network. These features are concatenated and fed into high-level Transformer layers for efficient multi-modal feature learning via modality-shared LoRA tuning. Finally, we concatenate these features and feed them into a classification head to achieve efficient fine-tuning. 

<p align="center">
<img src="https://github.com/Event-AHU/VELoRA/blob/main/figures/firstIMG.jpg" width="800">
</p>


### Framework 
<p align="center">
<img src="https://github.com/Event-AHU/VELoRA/blob/main/figures/frameworkVELoRA.jpg" width="800">
</p>



### Visualization 
<p align="center">
<img src="https://github.com/Event-AHU/VELoRA/blob/main/figures/HARDVSVIS.png" width="800">
</p>


### :hammer: Environment Configuration 


### Checkpoint 
| Model | File Size | Update Date  | Results on PokerEvent | Download Link                                            |
| ----- | --------- | ------------ | --------------------- | -------------------------------------------------------- |
| ViT-B/16   | 1.88GB  | Dec 29, 2024 |  57.99 |https://pan.baidu.com/s/1kCWwksfFJ_NoLpm9zFzPPQ?pwd=6ac4 | 



### Acknowledgement 



### :newspaper: Citation 
If you find this work helps your research, please star this GitHub and cite the following papers: 
```bibtex

```

If you have any questions about these works, please feel free to leave an issue.

