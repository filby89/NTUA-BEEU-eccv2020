# NTUA-BEEU-ECCV

Winning solution for the BEEU (First International Workshop on Bodily Expressed Emotion Understanding) challenge organized at ECCV2020. Please read the accompanied paper for more details. 

### Preparation
* Download the [BoLD dataset](https://cydar.ist.psu.edu/emotionchallenge/index.php).
* Use [https://github.com/yjxiong/temporal-segment-networks](https://github.com/yjxiong/temporal-segment-networks) in order to extract rgb and optical flow for the dataset.
* Change the directories in "dataset.py" file.


### Training

Train an RGB Temporal Segment Network on BoLD dataset:

> python train_tsn.py -c config_tsn.json --modality "RGB" -b 32 --lr 1e-3 --arch resnet101 --workers 4 --num_segments 3 --exp_name "rgb tsn"  -d 0,1,2,3

Add context branch:

> python train_tsn.py -c config_tsn.json --modality "RGB" -b 32 --lr 1e-3 --arch resnet101 --workers 4 --num_segments 3 --exp_name "rgb with context tsn"  -d 0,1,2,3 --context

Add visual embedding loss:

> python train_tsn.py -c config_tsn.json --modality "RGB" -b 32 --lr 1e-3 --arch resnet101 --workers 4 --num_segments 3 --exp_name "rgb with context tsn"  -d 0,1,2,3 --context --embed

Change modality to Flow:

> python train_tsn.py -c config_tsn.json --modality "Flow" -b 32 --lr 1e-3 --arch resnet101 --workers 4 --num_segments 3 --exp_name "rgb tsn"  -d 0,1,2,3


## Citation
If you use this code for your research, consider citing our paper.
```
@inproceedings{NTUA_BEEU,
  title={Emotion Understanding in Videos Through Body, Context, and Visual-Semantic Embedding Loss},
  author={Filntisis, Panagiotis Paraskevas and Efthymiou, Niki and Potamianos, Gerasimos and Maragos, Petros},
  booktitle={ECCV Workshop on Bodily Expressed Emotion Understanding},
  year={2020}
}
```

### Acknowlegements

* [https://github.com/yjxiong/tsn-pytorch](https://github.com/yjxiong/tsn-pytorch)
* [https://github.com/victoresque/pytorch-template](https://github.com/victoresque/pytorch-template)


### Contact 
For questions feel free to open an issue.