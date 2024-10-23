import os
import torch
import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback

class TransformerModelHandler:
    def __init__(self, model_name: str, train_dataset, test_dataset, output_dir, column, batch_size=16, num_epochs=3, lr=5e-5, class_weights=None, tokenizer_name=None):
        self.model_name = model_name
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.column = column
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if class_weights is not None:
            self.class_weights = class_weights.to(self.device)

        tokenizer_path = tokenizer_name if tokenizer_name else model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.num_labels = len(set(self.train_dataset['labels']))

        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=self.num_labels).to(self.device)

        self.training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            logging_dir=f'{output_dir}/logs',
            weight_decay=0.05,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            learning_rate=lr,
            metric_for_best_model="f1",
        )

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, average='macro')
        recall = recall_score(labels, predictions, average='macro')
        f1 = f1_score(labels, predictions, average='macro')
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    def train(self):
        self.trainer.train()
        curr_datetime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        best_model_path = f"{self.column}_{self.model_name}_{curr_datetime}"
        self.save_model(best_model_path)

    def evaluate(self):
        eval_metrics = self.trainer.evaluate()
        predictions_output = self.trainer.predict(self.test_dataset)
        logits = predictions_output.predictions
        predicted_labels = logits.argmax(axis=-1)
        true_labels = predictions_output.label_ids
        class_report = classification_report(true_labels, predicted_labels, output_dict=True)
        eval_metrics["per_class_metrics"] = {
            class_label: {
                "precision": class_report[str(class_label)]["precision"],
                "recall": class_report[str(class_label)]["recall"],
                "f1-score": class_report[str(class_label)]["f1-score"],
            }
            for class_label in set(true_labels)
        }
        return eval_metrics

    def save_model(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Model and tokenizer saved at {save_path}")
