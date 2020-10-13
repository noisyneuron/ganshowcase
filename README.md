# DC-GAN Training

This codebase is a slightly modified version of [ganshowcase](https://github.com/alantian/ganshowcase). View the web version of the original [here](https://alantian.net/ganshowcase/) to see what's possible.

## Requirements

- A GPU is (ideally) required for training. If you have an NVIDIA GPU on your machine, ensure `cuda` and `cudnn` are installed correctly. Or you can use a service like [Spell](https://www.spell.run/) or [Google Colab](https://colab.research.google.com/)  to access a remote GPU. Training is possible on a CPU, but will be *much* slower — taking weeks instead of hours. Instructions for training on Spell or NYU HPC (for NYU students) are [here](remote-gpu-instructions.md)
- Set up a Python environment, running Python 3.6 or higher



## Usage

Note: You can use [this Google Colab notebook](https://colab.research.google.com/drive/1WmZRWG1L1_SuBwDVyiexcLkGSn8gqYap?usp=sharing) to train a model without needing to install anything on your computer.

#### 1) Download this repository

```bash
git clone https://github.com/ml5js/training-dcgan
cd training-dcgan
```





#### 2) Collect training data

Collect a dataset of images and store them in the folder `images`. These could be images of the same type — eg. pictures of bedrooms or cityscapes — which would generate recognizable images of the same type, or you could use a more varied dataset that would probably generate more abstract output.

The code will resize and (center) crop the images to prepare for training, as 64px or 128px or 256px squares.





#### 3) Training

There are 4 steps to training, the method of execution differs slightly based on where/how you are training. (Instructions for training on Spell or NYU HPC (for NYU students) are [here](remote-gpu-instructions.md)):

1. Install dependencies

2. Convert image dataset to numpy arrays

3. Training (takes a few hours to days)

4. Converting the resulting model to a web-compatible model



* Install dependencies:

  ```
  pip install -r requirements.txt
  ```



* Convert to numpy array: Specify the cropped image size you would like to use for training (larger will take longer and require more memory) — 64, 128 or 256.  The following code will generate a `dataset.npz` file.

  ```
  ./datatool.py --size IMAGE_SIZE
  ```

  (If the images are stored elsewhere, use `--dir_path` to specify the path. You can change the path of the generated `.npz` file with the  `--npz_path`  flag)



* Train: Depending on the image size you used, specify the architecture to train with --

  64px : `./chainer_dcgan.py --arch dcgan64 --image_size 64 `

  128px : `./chainer_dcgan.py --arch resnet128 --image_size 128 `

  256px : `./chainer_dcgan.py --arch resnet256 --image_size 256 `

  (If you used a custom path for the `.npz` file, you can use the `--npz_path` flag to specify this. Output is saved to the `result` directory by default, this can be changed with the `--out` flag. See `chainer_dcgan.py` for more parameters)



* Convert model: After the training is complete, examine the `result` directory — you should see a collection of files generated at different points of the training — `SmoothedGenerator_XXXXX.npz`.

  You can see sample images from different iterations in the `result/preview_smoothed` directory — examine these to choose the iteration that produced the best results, and use that iteration number to  convert the model to a Tensorflow.js compatible format. Specify the architecture used.

  ```
  ./dcgan_chainer_to_keras.py --arch ARCHITECTURE_YOU_USED --chainer_model_path result/SmoothedGenerator_XXXXX.npz
  ```

  eg. using the `dcgan64` architecture with the model generated at the 50000th iteration:

  ```
  ./dcgan_chainer_to_keras.py --arch dcgan64 --chainer_model_path result/SmoothedGenerator_50000.npz
  ```

  This will generate a folder  `SmoothedGenerator_50000_tfjs`  which holds all the files you need to use the model with ml5 — take note of the `model.json` file.



###

#### 4) Using the model with ml5

See [this example](https://github.com/ml5js/ml5-examples/tree/release/p5js/DCGAN) for using DCGAN with ml5. To use your newly created model, make your own `manifest.json` file, that contains the path to your `model.json` file (inside the folder generated in the previous step).

The contents of your `manifest.json` file should be : (modify as appropriate)

```
{
    "description": "DESCRIPTION OF YOUR MODEL",
    "model": "PATH/TO/YOUR/MODEL.JSON/FILE",
    "modelSize": THE_IMAGE_SIZE_YOU_USED,
    "modelLatentDim": 128
}
```

For eg.

```
{
    "description": "Simpsons dataset from Kaggle",
    "model": "SmoothedGenerator_50000_tfjs/model.json",
    "modelSize": 256,
    "modelLatentDim": 128
}
```
