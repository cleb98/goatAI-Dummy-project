# 🐐 Pytorch Base Project
This repo is designed to be a good starting point for every PyTorch project.

## Create and setup the virtual environment
- Install _Anaconda_ or _Miniconda_ by following the instructions on the [official website](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).
- Open your terminal and run the following command to create the environment:
    ```bash
    conda create --name ptbase python=3.12
    ```
- Once the environment is created, activate it by running:
    ```bash
    conda activate ptbase
    ```
- Set the current directory as the root of the project by running:
    ```bash
    cd "<PROJECT_ROOT>"
    ```
  (replace `<PROJECT_ROOT>` with the absolute path to the root of the project, which should be a directory called `DASEval`) 
- Install the required packages by running:
    ```bash
    pip install -r requirements.txt
    ```

## Dummy Dataset

The dummy dataset is composed of pairs (`x`, `y_true`) in which:
* `x`: RGB image of size 128x128 representing a light blue circle 
   (radius = 16 px) a dark background (random circle position, randomly 
   colored dark background)
* `y_true`: copy of `x`, with the light blue circle surrounded with a red 
   line (4 px internal stroke)

|             `x` (input)              |          `y_true` (target)           |
|:------------------------------------:|:------------------------------------:|
| ![x0](dataset/samples/train/0_x.png) | ![y0](dataset/samples/train/0_y.png) |
| ![x1](dataset/samples/train/1_x.png) | ![y1](dataset/samples/train/1_y.png) |
| ![x2](dataset/samples/train/5_x.png) | ![y2](dataset/samples/train/5_y.png) |

## Dummy Model
* model input: `x`
* model output: `y_pred`
* loss: MSE between `y_pred` and `y_true`

```
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [bs, 32, 64, 64]             896
              SiLU-2           [bs, 32, 64, 64]               0
            Conv2d-3           [bs, 64, 32, 32]          18,496
              SiLU-4           [bs, 64, 32, 32]               0
            Conv2d-5           [bs, 64, 32, 32]          36,928
              SiLU-6           [bs, 64, 32, 32]               0
   ConvTranspose2d-7           [bs, 32, 64, 64]          18,464
              SiLU-8           [bs, 32, 64, 64]               0
   ConvTranspose2d-9         [bs, 32, 128, 128]           9,248
             SiLU-10         [bs, 32, 128, 128]               0
           Conv2d-11          [bs, 3, 128, 128]              99
================================================================
Total params: 84,131
Trainable params: 84,131
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.19
Forward/backward pass size (MB): 21.38
Params size (MB): 0.32
Estimated Total Size (MB): 21.88
----------------------------------------------------------------
```

## Output - Example

```
▶ experiment name [default]: magalli
┏━━━━━━━━━━━━━━━━━━━━━┓
┃ PytorchBase@shenron ┃
┗━━━━━━━━━━━━━━━━━━━━━┛

LR: 0.00015
EPOCHS: 64
N_WORKERS: 4
BATCH_SIZE: 8
MAX_PATIENCE: 64
DEVICE: cuda
DS_ROOT: ./dataset/samples

▶ Starting Experiment 'magalli' [seed: 6526]
tensorboard --logdir=/home/matteo/PycharmProjects/PytorchBase/log

[07-03@17:02] Epoch 0.256: │██████████████████████████████████████████████████│ 100.00% │ Loss: 0.029557 │ LR: 0.000075 │ T: 6.25 s
	● AVG Loss on TEST-set: 0.015271 │ T: 1.29 s
[07-03@17:02] Epoch 1.256: │██████████████████████████████████████████████████│ 100.00% │ Loss: 0.013235 │ LR: 0.000150 │ T: 5.90 s
	● AVG Loss on TEST-set: 0.010832 │ T: 1.30 s
[07-03@17:02] Epoch 2.256: │██████████████████████████████████████████████████│ 100.00% │ Loss: 0.008419 │ LR: 0.000150 │ T: 6.25 s
	● AVG Loss on TEST-set: 0.005636 │ T: 1.14 s
[07-03@17:02] Epoch 3.130: │█████████████████████████┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈│  50.78% │ Loss: 0.004958 │ LR: 0.000149
```
