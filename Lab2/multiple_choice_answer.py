# Exercise 3.2
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMultipleChoice, TrainingArguments, Trainer

from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
import torch

import evaluate
import numpy as np

import argparse

def preprocess_function(examples):
    first_sentences =[[context] * 4 for context in examples["sent1"]]
    question_headers = examples["sent2"]
    second_sentences = [
        [f"{header} {examples[end][i]}" for end in endings_names] for i, header in enumerate(question_headers)
    ]

    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])

    tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)
    return {k : [v[i:i+4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}

@dataclass
class DataCollatorForMultipleChoice:
    '''
    Data collator that will dynamically pad the inputs for multiple choice received.
    '''
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype = torch.int64)
        return batch
    
def parse_args():
    parser = argparse.ArgumentParser(description="Train a question answering model on the SWAG dataset")
    parser.add_argument("--train", action="store_true", help="Train the model on the SWAG dataset")
    parser.add_argument("--model-path", type=str, help="Path to the model to use for prediction")
    parser.add_argument("--prompt", type=str, help="The question you are asking the model. It is better to add a little bit of context :)")
    parser.add_argument("-n", "--answers", nargs="+", default=[])
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.train:
        dataset = load_dataset("swag", "regular")
        tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        # Preprocessing:
        # 1. Make four copies of the sent1 field and combine each of them with sent2 to recreate how a sentence starts
        # 2. Combine sent2 with each of the four possible sentence endings
        # 3. Flatten these two lists so it is possible to tokenize them. Then unflatten them afterward so each example has a coreresponding input_ids, attention_mask and labels field

        endings_names = ["ending0", "ending1", "ending2", "ending3"]
        tokenized_dataset = dataset.map(preprocess_function, batched=True)

        # Metric to see how well the model is doing
        accuracy = evaluate.load("accuracy")
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=1)
            return accuracy.compute(predictions=predictions, references=labels)

        model = AutoModelForMultipleChoice.from_pretrained("google-bert/bert-base-uncased")
        training_args = TrainingArguments(
            output_dir="./swag_model",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            learning_rate = 5e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
            push_to_hub=False,
            report_to="wandb",
            run_name="swag",
            logging_dir="./logs",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            data_collator=DataCollatorForMultipleChoice(tokenizer),
            compute_metrics=compute_metrics,
        )

        trainer.train()
    if args.prompt and args.answers and args.model_path:
        n_answers = len(args.answers)
        tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        prompt_answer_pairs = [[args.prompt, answer] for answer in args.answers]
        inputs = tokenizer(prompt_answer_pairs, return_tensors="pt", padding=True)
        labels = torch.tensor(0).unsqueeze(0)

        model = AutoModelForMultipleChoice.from_pretrained(args.model_path)
        outputs = model(**{k: v.unsqueeze(0) for k, v in inputs.items()}, labels=labels)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)
        print(f"The model predicts the answer to be {args.answers[predictions]}")

        with open("questions.txt", "a") as f:
            f.write(f"{args.prompt}\n")
            for i, answer in enumerate(args.answers):
                f.write(f"{i}: {answer}\n")
            f.write(f"Correct answer: {args.answers[predictions]}\n\n\n")



