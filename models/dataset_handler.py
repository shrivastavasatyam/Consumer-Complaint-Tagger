import torch
import numpy as np
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer

class DatasetHandler:
    def __init__(self, dataframe, model_name, test_size=0.3):
        self.dataframe = dataframe
        self.model_name = model_name
        self.test_size = test_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def prepare_datasets(self):
        df = self.dataframe.reset_index(drop=True)
        train_df, test_df = train_test_split(
            df,
            test_size=self.test_size,
            stratify=df['labels'],
            random_state=42
        )
        train_dataset = Dataset.from_pandas(train_df)
        test_dataset = Dataset.from_pandas(test_df)

        train_dataset = train_dataset.map(lambda examples: self.tokenizer(examples['Consumer complaint narrative'], truncation=True, padding="max_length", max_length=512), batched=True)
        test_dataset = test_dataset.map(lambda examples: self.tokenizer(examples['Consumer complaint narrative'], truncation=True, padding="max_length", max_length=512), batched=True)

        columns_to_remove = ['Product', 'Sub-product', 'Issue', 'Sub-issue', 'Consumer complaint narrative']
        train_dataset = train_dataset.remove_columns(columns_to_remove)
        test_dataset = test_dataset.remove_columns(columns_to_remove)

        return train_dataset, test_dataset
    
    def compute_class_weights(self, train_labels, device="cpu"):
        class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
        return torch.tensor(class_weights, dtype=torch.float).to(device)
