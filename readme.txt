This is a description file about the code of ours model 'DiCGAN'. 

1. run configuration.
CPU: Intel Core i7-6850K 3.60GHz, 6 core
GPU: NVIDIA Titan Xp (11G x 2)
CUDA: cudatoolkit9.0
cudnn: cudnn7.6.5
Python version: Python 3.6.8
TensorFlow version: TensorFlow-GPU 1.15.0

2. about dataset.
You can download CiaoDVD dataset from "https://www.librec.net/datasets.html" and Amazon dataset from "http://jmcauley.ucsd.edu/data/amazon/" respectively.
For each dataset, we hold the first 80% items in each user's interaction as the training set, and the remaining 20% items are used as the test set.
Meanwhile, we use the test set as a positive sample and randomly select 9 times negative samples.
So, you need run the file 'preprocessing.py' to get the train set, test set, test-neg set and the new ID of users and items.

3. run DiCGAN.
We have added some key comments in ours model to make it easy to read.
You can train our model by running "python main.py" with default parameters.
Moreover, you also can set different parameters to train it.