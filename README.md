# Deep Learning Lab
## Installation
**Conda**:
```
$ git clone https://github.com/felixboelter/Deep-Learning-Lab
$ conda config --append channels pytorch
$ conda create --name <env_name> --file requirements.txt
```
**Pip**:
```
$ git clone https://github.com/felixboelter/Deep-Learning-Lab
$ pip install torchtext==0.10.0 torchvision==0.10.0 torch==1.9.0 tqdm jupyter numpy matplotlib
```

## [Assignment 2](https://github.com/felixboelter/Deep-Learning-Lab/blob/main/Assignment_2)
Using a convolutional neural network (CNN) to classify the images in the CIFAR-10 dataset. The model includes Convolution layers accompanied with maxpooling layers, finally the model uses fully connected layers to obtain a lower dimensional output of 10 features corresponding to each class in the dataset. PyTorch and Torchvision is used to train this neural network.
## [Assignment 3](https://github.com/felixboelter/Deep-Learning-Lab/blob/main/Assignment_3)
Implemented a character-level language model using a recurrent neural network (RNN), specifically the long short-term memory (LSTM) model. The model is trained on the [Aesop's Fables](https://www.gutenberg.org/files/49010/49010-0.txt) from [Project Gutenberg](https://www.gutenberg.org/). In addition, it is also trained on Donald Trump's rally speeches, to validate the results. The implementation, training procedure, training losses and hyperparameters can be seen in the [Report](https://github.com/felixboelter/Deep-Learning-Lab/blob/main/Assignment_3/Report/Assignment_3_Felix_Boelter.pdf).
## [Assignment 4](https://github.com/felixboelter/Deep-Learning-Lab/blob/main/Assignment_4)
Implemented the Transformer sequence-to-sequence model to solve mathematical problems based on the [DeepMind Mathematics Dataset](https://github.com/deepmind/mathematics_dataset) 
which includes three difficulties. A further review of the implementation, hyperparameter tuning, the training and validation losses and their respective accuracies can be seen in the [Report](https://github.com/felixboelter/Deep-Learning-Lab/blob/main/Assignment_4/Report/Assignment_4_Felix_Boelter.pdf).

