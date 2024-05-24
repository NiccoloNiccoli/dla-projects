# Lab 2: LLMs
In this laboratory session we studied and conducted some experiments on Large Language Models.

## Experiment 1
In the first experiment we implemented a GPT (following [Andrej Karpathy's tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY)) from scratch and then we trained it on the Inferno by Dante Alighieri.
Our aim was to make the model generate some text that look like it was written by Dante.

The code is in `DanteGenerator.py` and the training corpus is `inferno.txt`.

We trained the model for 100k steps.
The optimizer we used is AdamW with a learning rate of 6e-4.

After the training we generated some text. Here there is a passage:
> Inferno: Cantoo del giovanetto anno<br>
  che 'l sole i crin sotto l'una,<br>
  s'alcun trafatto; e 'l pie' de la ripa uscia,<br>
  e sua nazion sara` tra feltro e feltro.<br><br>
  Di quella umile Italia fia salute
  per cennar la zuffa:<br>
  le via, che 'ntorno si mira<br>
  tutto smarrito de la grande angoscia<br>
  ch'elli ha sofferta, ed ei ritenuti;

The rest of the generated text is in `output_dante.txt`.

This passage shows us that the model we trained is capable of generating an Old Italian-looking text (even though without any sense) in just 30 minutes of training. Probably using a larger corpus would produce better results.

## Experiment 2
In this experiment we compared the length of input with the encoded sequence length. For this experiment we used GPT2 from HuggingFace.

The code is in `exercise2_1.py`.

Input length | Encoded sequence length
--|--
143 | 78 

## Experiment 3
In this experiment we used a pretrained GPT2 to generate text from a prompt.

We changed some of the parameters of the generation to see how they influence the output.

The parameter we experimented with are:
- `Do_sample` : to sample from model's predicted probabilities
- `temperature` : to control the randomness of the output
- `no_repeat_ngram_size` : to avoid repeating n-grams in the output

The code is in `generateFromPrompt.py` and the output is in `output_gpt2.txt`.

Here there are some lines from the output:
| | Text|
|--|--|
|Prompt| *He was a boy, she was a girl*|
|Base output| *He was a boy, she was a girl, she was a girl. She was a girl. She was a girl. She was a girl. She was a girl. She was a girl. She was a girl. She was a girl. She was a girl. She was a girl. She was a girl. She was a girl. She was a girl. She was a girl. She was a girl. She was a girl. She was a girl. She was a girl. She was a girl. She was a girl*|
| Do_sample = True, temperature = 0.5| *He was a boy, she was a girl. He was a boy, she was a girl. He was a boy, she was a girl. He was a boy, she was a girl. He was a boy, she was a girl. He was a boy, she was a girl. She was a boy, she was a girl. He was a boy, she was a girl. He was a boy, she was a girl. He was a boy, she was a girl. He was a boy, she was a girl* |
| Do_sample = True, temperature = 0.5, no_repeat_ngrams = 2 | *He was a boy, she was a girl. He was in the middle of the night, he was having a hard time sleeping. I was there, I saw him. <br> "I said, 'You are not going to help me, your mother is not gonna help you.' He said he would, but he said no, and he didn't want to. And I said to him, you know, if I could help him this way, that's what I would do. But he did not want me to do it.* |

From this experiment we see that to get a "good" output it is crucial to set a high (>=1) temperature or to set `no_repeat_ngrams` to at least 2.

## Experiment 4
In this experiment we used a pretrained BERT (as backbone) to make a text classifier.

The model uses BERT as backbone and two fully connected layers (with a ReLU) as classification head.

We trained the model on IMBD dataset.

We tested the model in three different settings:
- not training the head
- training the head
- fine-tuning the model

We ran 1 epoch of training / fine-tuning.

The results show that the model basically gives a random answer every time not training the head, while  the predictions are more accurate training the head or fine-tuning the model.
| Strategy | Accuracy |
|--|--|
| No training | 0.49 |
| Training the head | 0.85 |
| Fine-tuning the model | **0.92** |

## Experiment 5
In the last experiment of this session we trained a question answering model.

The model we used is BERT. The dataset we trained it on is SWAG.
We trained the model for 3 epochs.

The code is in `multiple_choice_answer.py`.
We made the model print in a text file the question we asked, the choices and the correct answer.

All the questions we asked the model are in `questions.txt`.

An example of this interaction between us and the model is the following:
> Which, among these, is the country where people eat more pasta?<br>• Italy<br>• France<br>• Mexico<br>Correct answer: Italy

We don't know if the answer is correct but we like to think it is.


