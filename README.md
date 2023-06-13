# Contrastive Learning from Text-Video for Efficient Local-Global Attention Temporal Grounding
> 重庆理工大学2023秋季《计算机视觉》期末课程论文代码实现.

![overview](/figures/overview.jpg)

## 预训练权重

模型在Charades-STA和TACoS两个数据集上的最终结果已上传至[google drive](https://drive.google.com/file/d/1MXAmPYmJi9J5cSatF7fAeUd5HnavU8W5/view?usp=sharing)，下载之后放至项目根目录即可。

测试:
```bash
# train VSLNet on Charades-STA dataset
python main_t7.py --task charades  --mode test --predictor rnn --model_dir pretrained_models
# train VSLNet on TACoS dataset
python main_t7.py --task tacos  --mode test --predictor rnn --model_dir pretrained_models
```

## Prerequisites
- python 3.x with, pytorch (`1.1.0`), torchvision, opencv-python, moviepy, tqdm, nltk, 
  transformers
- youtube-dl
- cuda10, cudnn

If you have [Anaconda](https://www.anaconda.com/distribution/) installed, the conda environment of VSLNet can be built 
as follow (take python 3.7 as an example):
```shell script
# preparing environment
conda create --name vslnet python=3.7
conda activate vslnet
conda install -c anaconda cudatoolkit=10.0 cudnn
conda install -c anaconda nltk pillow=6.2.1
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch
conda install -c conda-forge transformers opencv moviepy tqdm youtube-dl
# download punkt for word tokenizer
python3.7 -m nltk.downloader punkt
```

## Preparation
The details about how to prepare the `Charades-STA`, `TACoS` features are summarized 
here: [[data preparation]](/prepare). Alternatively, you can download the prepared visual features from 
[Box Drive](https://app.box.com/s/h0sxa5klco6qve5ahnz50ly2nksmuedw), and place them to the `./data/` directory.
Download the word embeddings from [here](http://nlp.stanford.edu/data/glove.840B.300d.zip) and place it to 
`./data/features/` directory.


## Quick Start
### Pytorch Version
**Train** and **Test**
```shell script
# the same as the usage of tf version
# train VSLNet on Charades-STA dataset
python main_t7.py --task charades --predictor rnn --mode train
# train VSLNet on ActivityNet Captions dataset
python main_t7.py --task activitynet --predictor rnn --mode train
# train VSLNet on TACoS dataset
python main_t7.py --task tacos --predictor rnn --mode train
```
> For unknown reasons, the performance of PyTorch codes is inferior to that of TensorFlow codes on some datasets.
