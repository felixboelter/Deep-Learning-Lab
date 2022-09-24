# Assignment 4
Implemented the Transformer sequence-to-sequence model to solve mathematical problems based on the [DeepMind Mathematics Dataset](https://github.com/deepmind/mathematics_dataset) 
which includes three difficulties. A further review of the implementation, hyperparameter tuning, the training and validation losses and their respective accuracies can be seen in the [Report](https://github.com/felixboelter/Deep-Learning-Lab/blob/main/Assignment_4/Report/Assignment_4_Felix_Boelter.pdf).
### Installation
Make sure to use `torchtext: version 0.10.0` the full install procedure is:
```
$ git clone https://github.com/felixboelter/Deep-Learning-Lab
$ pip install torchtext==0.10.0 torch==1.9.0
```
### Usage
Open [model_trainer.ipynb](https://github.com/felixboelter/Deep-Learning-Lab/blob/main/Assignment_4/model_trainer.ipynb) and run the code blocks from top to bottom. This will train the model with the default hyperparameter settings. You can change the hyperparameters by looking under the header "Hyperparameters and Constants".

### Results
**_numbers - place value_**
```
Example Question: What is the tens digit of 93283843? | Expected Answer: 4 | Generated Answer: 4<eos>
Example Question: What is the units digit of 93215897? | Expected Answer: 7 | Generated Answer: 7<eos>
Example Question: What is the thousands digit of 58179700? | Expected Answer: 9 | Generated Answer: 9<eos>
```
**_compare - sort_**
```
Example Question: Put 0.4, 5, 30, 50, -2, 16 in descending order. | Expected Answer: 50, 30, 16, 5, 0.4, -2 
| Generated Answer: 50, 30, 16, 5, 0.4, -2<eos>

Example Question: Sort -25/127, -2/13, 0.2. | Expected Answer: -25/127, -2/13, 0.2 
| Generated Answer: -25/127, -2/13, 0.2<eos>

Example Question: Sort 3, -0.2, 927897, 3/7 in ascending order. | Expected Answer: -0.2, 3/7, 3, 927897 
| Generated Answer: -0.2, 3/7, 3, 927897<eos>
```
**_algebra - linear 1d_**
```
Example Question: Solve -282*d + 929 - 178 = -1223 for d. | Expected Answer: 7 | Generated Answer: 7<eos>
Example Question: Solve 0 = -i - 91*i - 1598*i - 64220 for i. | Expected Answer: -38 | Generated Answer: -38<eos>
Example Question: Solve -25*m - 2084 = -2559 for m. | Expected Answer: 19 | Generated Answer: 19<eos>
```
