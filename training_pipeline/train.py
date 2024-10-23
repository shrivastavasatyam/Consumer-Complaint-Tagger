import os
from dotenv import load_dotenv
import pandas as pd
import torch
from models.dataset_handler import DatasetHandler
from models.model_handler import TransformerModelHandler
from config.hyperparams import MODEL_NAME, COLUMN, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, OUTPUT_DIR
load_dotenv()
df = pd.read_csv(os.getenv('DATA_PATH'))

# Prepare datasets
dataset_handler = DatasetHandler(dataframe=df, model_name=MODEL_NAME)
train_dataset, test_dataset = dataset_handler.prepare_datasets()

# Compute class weights
class_weights_tensor = dataset_handler.compute_class_weights(train_dataset['labels'], device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Initialize and train the model
model_handler = TransformerModelHandler(
    model_name=MODEL_NAME,
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    output_dir=OUTPUT_DIR,
    column=COLUMN,
    batch_size=BATCH_SIZE,
    num_epochs=NUM_EPOCHS,
    lr=LEARNING_RATE,
    class_weights=class_weights_tensor
)

# Train and evaluate
model_handler.train()
evaluation_results = model_handler.evaluate()
print(evaluation_results)
