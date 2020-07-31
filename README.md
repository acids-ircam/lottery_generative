# Generative structured lottery

This repository contains experiments on enhancing the lottery ticket hypothesis for deep generative models. It also supports the DaFX 2020 submission with code, sound examples and supplementary informations

## Supporting webpage

For a better viewing experience, please **visit the corresponding [supporting website](https://acids-ircam.github.io/lottery_generative/ "Generative lottery")**.

It embeds the following:
  * Supplementary figures
  * Audio examples
	* Reconstruction
	* Interpolation
  
You can also directly parse through the different sub-directories of the main [`docs`](docs) directory.

## Dataset

The examples in the paper have been computed on different audio datasets.

## Code

### Dependencies

#### Python

Code has been developed with `Python 3.7`. It should work with other versions of `Python 3`, but has not been tested. Moreover, we rely on several third-party libraries, listed in [`requirements.txt`](requirements.txt). They can be installed with

```bash
$ pip install -r requirements.txt
```

As our experiments are coded in PyTorch, no additional library is required to run them on GPU (provided you already have CUDA installed).

### Usage

The code is mostly divided into two scripts `train.py` and `evaluate.py`. The first script `train.py` allows to train a model from scratch as described in the paper. The second script `evaluate.py` allows to generate the figures of the papers, and also all the supporting additional materials visible on the [supporting page](https://acids-ircam.github.io/lottery_generative)) of this repository.

#### train.py arguments
```

```

## Pre-trained models

Note that a set of pre-trained models are availble in the `code/results` folder.

### Models details

As discussed in the paper, the very large amount of baseline models implemented did not allow to provide all the parameters for reference models (which are defined in the source code). However, we provide these details inside the documentation page in the [models details section](https://acids-ircam.github.io/lottery_generative/#models-details)
