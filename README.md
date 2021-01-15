# motion_deblur_dl

## Single Image non-Uniform Blind Motion Deblur

The aim of the motion deblur problem is to recover a sharp image from a blurry image due to camera shake or object movements.

## The Approach

Deep Learning-based approach has been proposed to recover sharp images. Two fully convolutional networks have been proposed with encoder-decoder architecture:
- Resnet Encoder with custom decoder (ResnetEncDec)
- Transform Net

### Resnet Encoder with custom decoder (ResnetEncDec)

The network uses Resnet-50 as an encoder. The architecture of the network is shown below:

![resnetencdec](https://github.com/Mekhak/motion_deblur_dl/blob/main/images/resnetencdec.png)

### Transform Net

The "Image Transform Net" part from the Style Transfer Network suggested in [Perceptual Losses for Real-Time Style Transfer and Super-Resolution(https://arxiv.org/pdf/1603.08155.pdf) paper. The architecture of the network is shown below:

![transformnet](https://github.com/Mekhak/motion_deblur_dl/blob/main/images/transformnet.png)

### The Dataset

The [GOPRO_Large](https://seungjunnah.github.io/Datasets/gopro) dataset was used suggested in the [Deep Multi-scale Convolutional Neural Network for Dynamic Scene Deblurring](https://arxiv.org/pdf/1612.02177.pdf) paper. The dataset contains 3124 blurry and sharp image pairs.

### Training

Both proposed networks have been trained for 350 epochs on the Geforce GTX 1070 ti GPU. ImageNet pre-trained weights are used to initialize ResnetEncDec encoder part.

#### Loss

The Mean Squared Error (MSE) and Mean Absolute Error (MAE) were used as losses between blurry and sharp images. Our experiments showed that MSE performs better at least at the firsts steps of the training.

#### Evaluation metrics

- Peak Signal-To-Noise Ratio ([PSNR](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio))
- Mean Squared Error ([MSE](https://en.wikipedia.org/wiki/Mean_squared_error))

#### Optimizer

The Adam optimizer was used with a learning rate of 0.001 for both networks. For Transform Net the training was continued additionally for 200 epochs with SGD optimizer with learning rate 0.0001 with momentum 0. However, that was not lead to significant improvements.

### Results

The PSNR and MSE results on test data are shown below (note that we used 256 x 256 images). [1] refers to [Deep Multi-scale Convolutional Neural Network for Dynamic Scene Deblurring](https://arxiv.org/pdf/1612.02177.pdf) paper.

![results](https://github.com/Mekhak/motion_deblur_dl/blob/main/images/res.png)

Some results for visual comparison are shown below:

![results](https://github.com/Mekhak/motion_deblur_dl/blob/main/images/res1.png)

![results](https://github.com/Mekhak/motion_deblur_dl/blob/main/images/res2.png)

![results](https://github.com/Mekhak/motion_deblur_dl/blob/main/images/res3.png)

![results](https://github.com/Mekhak/motion_deblur_dl/blob/main/images/res4.png)

![results](https://github.com/Mekhak/motion_deblur_dl/blob/main/images/res5.png)







