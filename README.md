# MIPI Challenge 2023 Team LVGroup_HFUT

> This repository is the official [MIPI Challenge 2023](http://mipi-challenge.org/#) implementation of Team LVGroup_HFUT in [Nighttime Flare Removal](https://codalab.lisn.upsaclay.fr/competitions/9402).

> The [restoration results](https://pan.baidu.com/s/1klcUzBUyWXxZ3eHXGestUg) (Extraction Code：a9zs) of the tesing images and [pretrained model](https://pan.baidu.com/s/12hQGGC6IwQ-GhKvw7tqNUQ) (Extraction Code：l9zt) can be downloaded from Baidu Netdisk.

## Usage

### Single image inference

`cd your/script/path`

`python infer.py --data_source your/dataset/path --model_path ../pretrained/epoch_0090.pth --save_image --experiment your-experiment-name`

### Train

`cd your/script/path`

`python train.py --data_source your/dataset/path --experiment your-experiment`

### Dataset format

> The format of the dataset should meet the following code in datasets.py:

`self.img_paths = sorted(glob.glob(data_source + '/train' + '/Flare' + '/*.*'))`

`self.gt_paths = sorted(glob.glob(data_source + '/train' + '/Flickr24K' + '/*.*'))`

> or

`self.img_paths = sorted(glob.glob(data_source + '/val' + '/input' + '/*.*'))`

***data_source*** is given by the command line.

***mode*** can be 'train' or 'val'.

### Path to saving results

***when training and validating:***  the default path is `'../results/your-experiment'`

***when testing:***  the default path is `'../outputs/your-experiment/test'`

***when inferring:***  the default path is `'../outputs/your-experiment/infer'`
