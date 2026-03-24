<div align="center">

<h1>SARMAE: Masked Autoencoder for SAR Representation Learning</h1>

Danxu Liu<sup>1,4 *</sup>, Di Wang<sup>2,4 *</sup>, Hebaixu Wang<sup>2,4 *</sup>, Haoyang Chen<sup>2,4 *</sup>, Wentao Jiang<sup>2</sup>, Yilin Cheng<sup>3,4</sup>, Haonan Guo<sup>2,4</sup>, 
Wei Cui<sup>1 †</sup>, Jing Zhang<sup>2,4 †</sup>.

<sup>1</sup> Beijing Institute of Technology,  <sup>2</sup> Wuhan University,  <sup>3</sup> Fudan University,  <sup>4</sup> Zhongguancun Academy.

<sup>*</sup> Equal contribution.  <sup>†</sup> Corresponding authors.

</div>

<p align="center">
  <a href="#-update">Update</a> |
  <a href="#-abstract">Abstract</a> |
  <a href="#-datasets">Datasets</a> |
  <a href="#-pre-training">Pre-training</a> |
  <a href="#-usage">Usage</a> |
  <a href="#-statement">Statement</a>
</p >
<p align="center">
<a href="https://arxiv.org/abs/2512.16635"><img src="https://img.shields.io/badge/Paper-arxiv-red"></a>  
<a href="https://pan.baidu.com/s/1ok4QCfeTVSJlPpAuLxEVxQ?pwd=0717"><img src="https://img.shields.io/badge/Dataset-BaiduYun-blue"></a>
<a href="https://pan.baidu.com/s/1DOsZolLZ--gMuNUgUXeyVg?pwd=0717"><img src="https://img.shields.io/badge/Pretrain Weight-BaiduYun-blue"></a>
<a href="https://huggingface.co/datasets/Wenquandan777/SAR-1M"><img src="https://img.shields.io/badge/Dataset-Hugging%20face-yellow"></a>
<a href="https://huggingface.co/Wenquandan777/SARMAE"><img src="https://img.shields.io/badge/Pretrain Weight-Hugging%20face-yellow"></a>
</p>

## 🔥 Update

**2026.3.24**
- The codes of pretraining and classification in fintuning are released!

**2026.3.23**
- SARMAE pretrained weights are publicly available on [Hugging Face](https://huggingface.co/Wenquandan777/SARMAE) and [Baidu Netdisk](https://pan.baidu.com/s/1DOsZolLZ--gMuNUgUXeyVg?pwd=0717).

**2026.3.16**

- SAR-1M dataset is publicly available on [Hugging Face](https://huggingface.co/datasets/Wenquandan777/SAR-1M) and [Baidu Netdisk](https://pan.baidu.com/s/1ok4QCfeTVSJlPpAuLxEVxQ?pwd=0717).

**2026.2.21**

- The paper is accepted by **CVPR 2026**! 🎉🎉🎉

**2025.12.19**

- The paper is post on arXiv! **([arXiv SARMAE](https://arxiv.org/abs/2512.16635))**

## 🌞 Abstract

Synthetic Aperture Radar (SAR) imagery plays a critical role in all-weather, day-and-night remote sensing applications. However, existing SAR-oriented deep learning is constrained by data scarcity, while the physically grounded speckle noise in SAR imagery further hampers fine-grained semantic representation learning. To address these challenges, we propose SARMAE, a Noise-Aware Masked Autoencoder for self-supervised SAR representation learning. Specifically, we construct SAR-1M, the first million-scale SAR dataset, with additional paired optical images, to enable large-scale pre-training. Building upon this, we design Speckle-Aware Representation Enhancement (SARE), which injects SAR-specific speckle noise into masked autoencoders to facilitate noise-aware and robust representation learning. Furthermore, we introduce Semantic Anchor Representation Constraint (SARC), which leverages paired optical priors to align SAR features and ensure semantic consistency. Extensive experiments across multiple SAR datasets demonstrate that SARMAE achieves state-of-the-art performance on classification, detection, and segmentation tasks.

<figure>
<div align="center">
<img src=Figs/model.png width="100%">
</div>

<div align='center'>

**Figure 1. Overview of the SARMAE pretraining framework. The framework consists of two branches: (i) a SAR branch following the MAE architecture with Speckle-Aware Representation Enhancement (SARE) to handle inherent speckle noise, and (ii) an optical branch using a frozen DINOv3 encoder. For paired SAR-optical data, Semantic Anchor Representation Constraint (SARC) aligns SAR features with semantic-rich optical representations. Unpaired SAR images are processed solely through the SAR branch.**

</div>

## 📖 Datasets

<figure>
<div align="center">
<img src=Figs/dataset.png width="40%">
</div>

<div align='center'>

**Figure 2. The organization of data sources in SAR-1M.**

</div>

SAR-1M is a large-scale synthetic aperture radar (SAR) image dataset designed for SAR representation learning. The dataset contains over one million SAR images, and about 75% of the SAR samples are paired with geographically aligned optical images, enabling multimodal remote sensing studies.



## 🚀 Pre-training

Environment:

- Python 3.8.20
- Pytorch 1.12.1+cu113
- torchvision 0.13.1+cu113
- timm 0.6.13 

### Step-by-step installation (suitable for 4090&A800&A100)

```
conda create -n sarmae python=3.8 -y
conda activate sarmae

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install requirements.txt
```
1. Preparing with SAR-1M: Download the [SAR-1M](https://huggingface.co/datasets/Wenquandan777/SAR-1M). The indices of paired images are provided in `paired.json`, while those of unpaired images are listed in `unpaired.json`. To extend the SAR-1M dataset for pretraining with additional data, users can append the corresponding image indices to these JSON files.

2. Pretraining: take ViT-B as an example (batchsize: 4096=8*512)

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7     python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --master_port 20003 \
    train_mae_contrastive.py \
    --model mae_vit_base_patch16 \
    --data_path ./data \
    --enable_sar_noise \
    --noise_ratio 0.5 --random_noise \
    --noise_min 0.0 --noise_max 0.7 \
    --output_dir ./output_vitb \
    --batch_size 512 --epochs 300 \
    --lr 1e-4 --mae_loss_weight 1 --alignment_loss_weight 0.8 \
    --loss_schedule cosine \
    --sar_pretrained ./mae_pretrain_vit_base.pth \
    --dinov3_pretrained ./dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
    --freeze_optical_completely \
    --clip_grad 1.0    
```

3. Fine-tuning: an example of evaluating the pretrained ViT-B weight on Fusar dataset

```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1   --master_port 20005 \
main_finetune.py \
--dataset 'fusar'  --data_path /data/FUSAR  \
--model 'vit_base_patch16'   \
--batch_size 8 --epochs 30  --exp_num=5  \
--finetune './SARMAE_vit_Base.pth' 
```

#### SARMAE pretrained weights
|Pretrain|Backbone | Input size | Pretrained model|
|-------|-------- | ---------- | ----- |
| SARMAE | ViT-B | 224 × 224 | [Weights](https://huggingface.co/Wenquandan777/SARMAE/blob/main/SARMAE_vit_Base.pth) |
| SARMAE | ViT-L | 224 × 224 | [Weights](https://huggingface.co/Wenquandan777/SARMAE/blob/main/SARMAE_vit_Large.pth) |

## 🔨 Usage

Coming Soon.

## 🍭 Results

<figure>
<div align="center">
<img src=Figs/radar.png width="50%">
</div>

<div align='center'>

**Figure 3. SARMAE outperforms SOTA methods on multiple datasets. <sup>1</sup>: 40-SHOT; <sup>2</sup>: 30% labeled. <sup>a</sup>: Multi-classes; <sup>b</sup>: Water.**

</div>

<table>
<thead>
  <tr>
    <th rowspan="2">Method</th>
    <th colspan="2" align="center"><b>FUSAR-SHIP</b></th>
    <th colspan="2" align="center"><b>MSTAR</b></th>
    <th colspan="1" align="center"><b>SAR-ACD</b></th>
  </tr>
  <tr>
    <th align="center">40-shot</th>
    <th align="center">30%</th>
    <th align="center">40-shot</th>
    <th align="center">30%</th>
    <th align="center">30%</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>ResNet-50</td>
    <td align="center">-</td>
    <td align="center">58.41</td>
    <td align="center">-</td>
    <td align="center">89.94</td>
    <td align="center">59.70</td>
  </tr>
  <tr>
    <td>Swin Transformer</td>
    <td align="center">-</td>
    <td align="center">60.79</td>
    <td align="center">-</td>
    <td align="center">82.97</td>
    <td align="center">67.50</td>
  </tr>
  <tr>
    <td>Bet</td>
    <td align="center">59.70</td>
    <td align="center">71.13</td>
    <td align="center">40.70</td>
    <td align="center">69.75</td>
    <td align="center">79.77</td>
  </tr>
  <tr>
    <td>LoMaR</td>
    <td align="center">82.70</td>
    <td align="center">-</td>
    <td align="center">77.00</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td>SAR-JEPA</td>
    <td align="center">85.80</td>
    <td align="center">-</td>
    <td align="center">91.60</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td>SUMMIT</td>
    <td align="center">-</td>
    <td align="center">71.91</td>
    <td align="center">-</td>
    <td align="center">98.39</td>
    <td align="center">84.25</td>
  </tr>
  <tr style="border-top: 2px solid #999;">
    <td><b>SARMAE(ViT-B)</b></td>
    <td align="center">89.30</td>
    <td align="center"><b>92.92<b></td>
    <td align="center">96.70</td>
    <td align="center"><b>99.61</b></td>
    <td align="center">95.06</td>
  </tr>
  <tr>
    <td><b>SARMAE(ViT-L)</b></td>
    <td align="center"><b>90.86</b></td>
    <td align="center">92.80</td>
    <td align="center"><b>97.24</b></td>
    <td align="center">98.92</td>
    <td align="center"><b>95.63</b></td>
  </tr>
</tbody>
</table>

**Table 1.** Performance comparison (Top1 Accuracy, %) of different methods on the target classification task.

</div>

<table>
<thead>
  <tr>
    <th align="center">Method</th>
    <th align="center">SARDet-100k</th>
    <th align="center">SSDD</th>
    <th align="center">Method</th>
    <th align="center">RSAR</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>ImageNet</td>
    <td align="center">52.30</td>
    <td align="center">66.40</td>
    <td>RoI Transformer</td>
    <td align="center">35.02</td>
  </tr>
  <tr>
    <td>Deformable DETR</td>
    <td align="center">50.00</td>
    <td align="center">52.60</td>
    <td>Def. DETR</td>
    <td align="center">46.62</td>
  </tr>
  <tr>
    <td>Swin Transformer</td>
    <td align="center">53.80</td>
    <td align="center">40.70</td>
    <td>RetinaNet</td>
    <td align="center">57.67</td>
  </tr>
  <tr>
    <td>ConvNeXt</td>
    <td align="center">55.10</td>
    <td align="center">-</td>
    <td>ARS-DETR</td>
    <td align="center">61.14</td>
  </tr>
  <tr>
    <td>CATNet</td>
    <td align="center">-</td>
    <td align="center">64.66</td>
    <td>R3Det</td>
    <td align="center">63.94</td>
  </tr>
  <tr>
    <td>MSFA</td>
    <td align="center">56.40</td>
    <td align="center">-</td>
    <td>ReDet</td>
    <td align="center">64.71</td>
  </tr>
  <tr>
    <td>SARAFE</td>
    <td align="center">57.30</td>
    <td align="center">67.50</td>
    <td>O-RCNN</td>
    <td align="center">64.82</td>
  </tr>
  <tr style="border-top: 2px solid #999;">
    <td><b>SARMAE(ViT-B)</b></td>
    <td align="center">57.90</td>
    <td align="center">68.10</td>
    <td><b>SARMAE(ViT-B)</b></td>
    <td align="center">66.80</td>
  </tr>
  <tr>
    <td><b>SARMAE(ViT-L)</b></td>
    <td align="center"><b>63.10</b></td>
    <td align="center"><b>69.30</b></td>
    <td><b>SARMAE(ViT-L)</b></td>
    <td align="center"><b>72.20</b></td>
  </tr>
</tbody>
</table>

**Table 2.** Performance comparison (mAP, %) of different methods on horizontal and oriented object detection tasks.

</div>

<table>
<thead>
  <tr>
    <th rowspan="2">Method</th>
    <th colspan="7" align="center"><b>Multiple classes</b></th>
    <th colspan="1" align="center"><b>Water</b></th>
  </tr>
  <tr>
    <th>Industrial Area</th>
    <th>Natural Area</th>
    <th>Land Use</th>
    <th>Water</th>
    <th>Housing</th>
    <th>Other</th>
    <th>mIoU</th>
    <th>IoU</th>
  </tr> 
</thead>
<tbody>
  <tr>
    <td>FCN</td>
    <td align="center">37.78</td>
    <td align="center">71.58</td>
    <td align="center">1.24</td>
    <td align="center">72.76</td>
    <td align="center">67.69</td>
    <td align="center">39.05</td>
    <td align="center">48.35</td>
    <td align="center">85.95</td>
  </tr>
  <tr>
    <td>ANN</td>
    <td align="center">41.23</td>
    <td align="center">72.92</td>
    <td align="center">0.97</td>
    <td align="center">75.95</td>
    <td align="center">68.40</td>
    <td align="center">56.01</td>
    <td align="center">52.58</td>
    <td align="center">87.32</td>
  </tr>
  <tr>
    <td>PSPNet</td>
    <td align="center">33.99</td>
    <td align="center">72.31</td>
    <td align="center">0.93</td>
    <td align="center">76.51</td>
    <td align="center">68.07</td>
    <td align="center">57.07</td>
    <td align="center">51.48</td>
    <td align="center">87.13</td>
  </tr>
  <tr>
    <td>DeepLab V3+</td>
    <td align="center">40.62</td>
    <td align="center">70.67</td>
    <td align="center">0.55</td>
    <td align="center">72.93</td>
    <td align="center">69.96</td>
    <td align="center">34.53</td>
    <td align="center">48.21</td>
    <td align="center">87.53</td>
  </tr>
  <tr>
    <td>PSANet</td>
    <td align="center">40.70</td>
    <td align="center">69.46</td>
    <td align="center">1.33</td>
    <td align="center">69.46</td>
    <td align="center">68.75</td>
    <td align="center">32.68</td>
    <td align="center">47.14</td>
    <td align="center">86.18</td>
  </tr>
  <tr>
    <td>DANet</td>
    <td align="center">39.56</td>
    <td align="center">72.00</td>
    <td align="center">1.00</td>
    <td align="center">74.95</td>
    <td align="center">67.79</td>
    <td align="center">56.28</td>
    <td align="center">39.56</td>
    <td align="center">89.29</td>
  </tr>
  <tr style="border-top: 2px solid #999;">
    <td><b>SARMAE(ViT-B)</b></td>
    <td align="center"><b>65.87</b></td>
    <td align="center">75.65</td>
    <td align="center">29.20</td>
    <td align="center">84.01</td>
    <td align="center">73.23</td>
    <td align="center"><b>71.21</b></td>
    <td align="center">66.53</td>
    <td align="center">92.31</td>
  </tr>
  <tr>
    <td><b>SARMAE(ViT-L)</b></td>
    <td align="center">65.84</td>
    <td align="center"><b>78.04</b></td>
    <td align="center"><b>29.47</b></td>
    <td align="center"><b>87.12</b></td>
    <td align="center"><b>75.22</b></td>
    <td align="center">69.34</td>
    <td align="center"><b>67.51</b></td>
    <td align="center"><b>93.06</b></td>
  </tr>
</tbody>
</table>

**Table 3.** Performance comparison of semantic segmentation methods on multiple classes and water classes.

## ⭐ Citation

If you find SARMAE helpful, please give a ⭐ and cite it as follows:

```
@misc{liu2025sarmaemaskedautoencodersar,
      title={SARMAE: Masked Autoencoder for SAR Representation Learning}, 
      author={Danxu Liu and Di Wang and Hebaixu Wang and Haoyang Chen and Wentao Jiang and Yilin Cheng and Haonan Guo and Wei Cui and Jing Zhang},
      year={2025},
      eprint={2512.16635},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.16635}, 
}
```

## 🎺 Statement

- This project is released under the [CC BY-NC 4.0](LICENSE).
- For any other questions please contact Danxu Liu at [bit.edu.cn](3120245436@bit.edu.cn) or [gmail.com](ldx.wenquandan@gmail.com).
