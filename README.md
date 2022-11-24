# Multi-modal Grouping Network for Weakly-Supervised Audio-Visual Video Parsing

Official implementation for MGN. 
MGN is a novel and lightweight baseline with explicitly semantic-aware grouping for weakly-supervised audio-visual video parsing.

<div align="center">
  <img width="100%" alt="MGN Illustration" src="images/framework.png">
</div>

## Environment

To setup the environment, please simply run

```
pip install -r requirements.txt
```

## Datasets

Data can be downloaded from [Unified Multisensory Perception: Weakly-Supervised Audio-Visual Video Parsing, ECCV 2020](https://github.com/YapengTian/AVVP-ECCV20)


## Model

A trained model **MGN_Net.pt** is provided for inference in **models** dir.


## Train & Test

For training an MGN model, please run

```
python main.py --mode train \
    --audio_dir path/to/vggish/feats/ \
    --video_dir path/to/res152/feats/ \
    --st_dir path/to/r2plus1d_18/feats/ \
    --model_save_dir models/ \
    --unimodal_assign soft --crossmodal_assign soft \
    --epochs 40 \
    --depth_aud 3 --depth_vis 3 --depth_av 6
```

For testing, simply run

```
python main.py --mode test \
    --audio_dir path/to/vggish/feats/ \
    --video_dir path/to/res152/feats/ \
    --st_dir path/to/r2plus1d_18/feats/ \
    --model_save_dir models/ \
    --unimodal_assign soft --crossmodal_assign soft
```
