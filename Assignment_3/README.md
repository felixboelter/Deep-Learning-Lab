# Assignment 3
Implemented a character-level language model using a recurrent neural network (RNN), specifically the long short-term memory (LSTM) model. The model is trained on the [Aesop's Fables](https://www.gutenberg.org/files/49010/49010-0.txt) from [Project Gutenberg](https://www.gutenberg.org/). In addition, it is also trained on Donald Trump's rally speeches, to validate the results. The implementation, training procedure, training losses and hyperparameters can be seen in the [Report](https://github.com/felixboelter/Deep-Learning-Lab/blob/main/Assignment_3/Report/Assignment_3_Felix_Boelter.pdf).
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
$ pip install torchtext==0.10.0 torch==1.9.0 tqdm jupyter numpy matplotlib
```
## Usage
Open [model_trainer.ipynb](https://github.com/felixboelter/Deep-Learning-Lab/blob/main/Assignment_3/model_trainer.ipynb) and run the code blocks from top to bottom. This will train the model with the default hyperparameter settings. You can change the hyperparameters by looking under the header "Hyperparameters".

## Results
**_Fable - A FOX was once caught in a trap by his_**
```
generated text: tail. He succeeded in getting<eos>away, but was forced to leave his “brush” behind. 
He soon realized that<eos>his life would be a burden, for the handing down day.<eos>The Mouse showed
the Frog his nest and everything he could think<eos>out they were untied: “I shall get that way, you 
would never reach half its size.” Vexed that her child should disparance<eos>opproached warmraid was hard 
for the preceding small be foes. What<eos>a little waiting is load warming, the Eagle swam a good price.
```
**_Trump - President Donald J. Trump_**
```
generated text: 01:14)<eos>Thank you very much and hello, Kenosha. It’s nice to be back. It’s nice to be back. We spent 
a little time with you, a little law and order. We brought law and order to Kenosha. Right? That’s what we want. And 
hello, Wisconsin. Big day, tomorrow, big, big day, big day. And I think we’re going to do very well in Wisconsin just 
like we did four years ago. And it’s an honor to be with you. Thank you.<eos><eos>Audience: (01:41)<eos>USA! USA! 
USA! USA!<eos><eos>President Donald J. Trump: (01:51)<eos>And this is a lot of people. This is a lot of people. 
See, you know what that means? That means we don’t have to pay
```

