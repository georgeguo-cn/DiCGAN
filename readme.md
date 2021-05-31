# DiCGAN

This is my Tensorflow implementation for the paper:
>Zhiqiang Guo, Chaoyang Wang, Jianjun Li, Guohui Li, Peng Pan(2020). DiCGAN: A Dilated Convolutional Generative Adversarial Network for Recommender Systems, [Paper in Springer](https://link.springer.com/chapter/10.1007/978-3-030-73200-4_18). In DASFAA 2020.

## Configuration.

We run this code on the follwing hardware configuration:
* CPU: Intel Core i7-6850K 3.60GHz, 6 core
* GPU: NVIDIA Titan Xp (11G x 2)
* CUDA: cudatoolkit9.0
* cudnn: cudnn7.6.5

The required packages are as follows:
* Python  == 3.6.8
* TensorFlow-gpu == 1.15.0

## Dataset.
You can download CiaoDVD dataset from "https://www.librec.net/datasets.html" and Amazon dataset from "http://jmcauley.ucsd.edu/data/amazon/" respectively.

For each dataset, we hold the first 80% items in each user's interaction as the training set, and the remaining 20% items are used as the test set.
Meanwhile, we use the test set as a positive sample and randomly select 9 times negative samples.
So, you need run the file 'preprocessing.py' to get the train set, test set, test-neg set and the new ID of users and items.
```
python preprocessing.py
```

## Run.

You can train our model by running "python main.py" with default parameters.
```
python main.py
```

Moreover, you also can set different parameters to train it.
