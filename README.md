# Cycle GAN

## Introduction
We implement a simple form of Cycle GAN described in the article [Unpaired Image-to-Image Translation](https://arxiv.org/abs/1703.10593), in PyTorch. While preparing, we inspired by the repositories [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [PyTorch-CycleGAN](https://github.com/aitorzip/PyTorch-CycleGAN), 
and the course [*Apply GANs*](https://www.coursera.org/learn/apply-generative-adversarial-networks-gans). We trained the model on dataset [Horse2zebra](https://www.kaggle.com/datasets/balraj98/horse2zebra-dataset), which is availabe in the directory `datasets`.

## Setting up the environment

1. Install [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html), if not already installed.
2. Clone the repository
    ~~~
    git clone https://github.com/byrkbrk/cycle-gan.git
    ~~~
3. In the directory `cycle-gan`, for macos, execute:
    ~~~
    conda env create -f cycle-gan-env_macos.yaml
    ~~~
    For linux or windows run:
    ~~~
    conda env create -f cycle-gan-env_linux_or_windows.yaml
    ~~~
4. Activate the environment:
    ~~~
    conda activate cycle-gan-env
    ~~~

## Training and Inference

To train the model from scratch:

~~~
python3 train.py --dataset-name horse2zebra --n-epochs 200 --batch-size 1
~~~

To train the model from a checkpoint,

~~~
python3 train.py --checkpoint-name <your-checkpoint>
~~~
where replace the input `<your-checkpoint>` with your checkpoint name; `horse2zebra_checkpoint_10.pth` as an example.

For inference, suffices to execute

~~~
python3 generate.py <your-checkpoint>
~~~
where replace the input `<your-checkpoint>` with your checkpoint.

To generate images from our pretrained model, run

~~~
python3 generate.py pretrained_horse2zebra_checkpoint_219.pth --allow-checkpoint-download True --dataset-name horse2zebra
~~~

It downloads the pretrained checkpoint, generates the images using horse2zebra test dataset, and saves into directory `generated-images`.


## Results

### Horse (A) $\longrightarrow$ Zebra (B)
From our pretrained model, we present the images that are also shared in the CycleGAN article. The results below are foundable in the directory `generated-images/horse2zebra-AB` (with the indices 10, 12, 18, 33, 88).

<div style="display: flex;">
    <img src=files-for-readme/image_AB_10.jpeg alt="AB-jpeg" style="width: 48%; margin-right: 2%;">
    <img src=files-for-readme/image_AB_12.jpeg alt="AB-jpeg" style="width: 48%; margin-left: 2%;">
</div>

<div style="display: flex;">
    <img src=files-for-readme/image_AB_18.jpeg alt="AB-jpeg" style="width: 48%; margin-right: 2%;">
    <img src=files-for-readme/image_AB_33.jpeg alt="AB-jpeg" style="width: 48%; margin-left: 2%;">
</div>

<div style="display: flex;">
    <img src=files-for-readme/image_AB_88.jpeg alt="AB-jpeg" style="width: 48%; margin-right: 2%;">
</div>

### Zebra (B) $\longrightarrow$ Horse (A)
The results below are from the pretrained model, and foundable in the directory `generated-images/horse2zebra-BA` (with the indices 39, 72, 95, 113). The test images below are also presented in the CycleGAN article. 

<div style="display: flex;">
    <img src=files-for-readme/image_BA_39.jpeg alt="BA-jpeg" style="width: 48%; margin-right: 2%;">
    <img src=files-for-readme/image_BA_72.jpeg alt="BA-jpeg" style="width: 48%; margin-left: 2%;">
</div>

<div style="display: flex;">
    <img src=files-for-readme/image_BA_95.jpeg alt="BA-jpeg" style="width: 48%; margin-right: 2%;">
    <img src=files-for-readme/image_BA_113.jpeg alt="BA-jpeg" style="width: 48%; margin-left: 2%;">
</div>
