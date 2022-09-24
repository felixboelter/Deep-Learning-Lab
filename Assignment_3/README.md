# Assignment 3
Implemented a character-level language model using a recurrent neural network (RNN), specifically the long short-term memory (LSTM) model. The model is trained on the [Aesop's Fables](https://www.gutenberg.org/files/49010/49010-0.txt) from [Project Gutenberg](https://www.gutenberg.org/). In addition, it is also trained on Donald Trump's rally speeches, to validate the results. The implementation, training procedure, training losses and hyperparameters can be seen in the [Report](https://github.com/felixboelter/Deep-Learning-Lab/blob/main/Assignment_3/Report/Assignment_3_Felix_Boelter.pdf).

### Usage
Open [model_trainer.ipynb](https://github.com/felixboelter/Deep-Learning-Lab/blob/main/Assignment_3/model_trainer.ipynb) and run the code blocks from top to bottom. This will train the model with the default hyperparameter settings. You can change the hyperparameters by looking under the header "Hyperparameters".

### Results
**_Fable _**
```
Sample prompt: A FOX was once caught in a trap by his | generated text: tail. He succeeded in getting<eos>away, but was forced to leave his “brush” behind. He soon realized that<eos>his life would be a burden, for the handing down day.<eos>The Mouse showed the Frog his nest and everything he could think<eos>out they were untied: “I shall get that way, you would never reach half its size.” Vexed that her child should disparance<eos>opproached warmraid was hard for the preceding small be foes. What<eos>a little waiting is load warming, the Eagle swam a good price.

Sample prompt: <eos> | generated text: to come and live<eos>with me; I have plenty of food and water, and nothing to disturb me;<eos>and it is so pleasant in my pond. Now here there is very little food,<eos>and not much water, and the road passes through your pool, so that you<eos>must always be afraid of passers-by.”<eos><eos>“Thank you,” said the other Frog; “you are very kind, but I am quite<eos>content here. There is water enough; those who pass never trouble me;<eos>and as to food, I had a good dinner day before yesterday. I am used to<eos>this place, you know, and do not like change. If you are outside the<eos>United States, you'll have to check the laws of the cou

```
**_Trump_**
```
Sample prompt: Money | generated text:  this is a poll. See, this is a poll. When you draw crowds like this. This is not the crowd of somebody that’s going to lose the state of Michigan. This is not. This is not. And look at this crowd. You can’t even see the end of it. You can’t. This is not the crowd of a second place finisher. Do you agree with that? No. No. This is our crowd, all together. We’re in this together and we’re doing it together. As long as I’m President, we will remain the number one producer of oil and natural gas anywhere on this planet. And for the first time, we are energy independent. You never heard that term 

Sample prompt: Covid 19 | generated text: 5 million jobs in the last five months alone, which is another record. That’s another big record. We’ve never hired that many people that fast. While foreign nations are in a free fall, we’re creating an economic powerhouse unrivaled anywhere in the world. A recent Gallup Poll found that 56\% of Americans say they are better off today than they were four years ago under Obama and Biden. And if Biden and Kamala … You don’t have to say it “Kamala.” “Kamala.” If Biden and Kamala Harris, who’s further left by far than crazy Bernie Sanders, right, he’s considered a strict conservative compared to he

Sample prompt: President Donald J. Trump:  | generated text: 01:14)<eos>Thank you very much and hello, Kenosha. It’s nice to be back. It’s nice to be back. We spent a little time with you, a little law and order. We brought law and order to Kenosha. Right? That’s what we want. And hello, Wisconsin. Big day, tomorrow, big, big day, big day. And I think we’re going to do very well in Wisconsin just like we did four years ago. And it’s an honor to be with you. Thank you.<eos><eos>Audience: (01:41)<eos>USA! USA! USA! USA!<eos><eos>President Donald J. Trump: (01:51)<eos>And this is a lot of people. This is a lot of people. See, you know what that means? That means we don’t have to pay
```

