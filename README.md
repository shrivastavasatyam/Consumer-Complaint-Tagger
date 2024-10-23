# complaintTag



# ComplaintTag Classification System

Consumers often encounter challenges with financial products and services, leading to unresolved complaints with financial institutions. The Consumer Financial Protection Bureau (CFPB) serves as a mediator in these cases, but accurately categorizing complaints can be difficult for consumers, causing delays in the resolution process.

Our project streamlines complaint submission by automatically categorizing complaints based on the narrative descriptions. This improves the efficiency of complaint handling and ensures they are quickly routed to the appropriate teams for resolution.

This project provides a system to classify consumer complaints related to various financial products and issues handled by the Consumer Financial Protection Bureau (CFPB). It includes a **Streamlit application** for complaint classification and a pipeline to **train custom models** using transformer-based models such as BERT.

## Table of Contents

- [Features](#features)
- [Folder Structure](#folder-structure)
- [Running the Streamlit Application](#running-the-streamlit-application)
- [Training a Custom Model](#training-a-custom-model)
- [Environment Setup](#environment-setup)
- [How to Contribute](#how-to-contribute)

## Features

- Classify consumer complaints into CFPB categories like Product, Sub-product, Issue, and Sub-issue.
- Use pre-trained transformer models for classification.
- Train custom models on your own datasets.
- Display classification results with probabilities via an interactive Streamlit application.

## Folder Structure

```bash
complaintTag/
│
├── app.py                        # Streamlit application script
├── models/                       # Folder for model loading, training, and inference logic
│   ├── __init__.py               # Makes 'models' a package
│   ├── dataset_handler.py        # Dataset preparation (DatasetHandler)
│   ├── model_handler.py          # Model setup, training, evaluation (TransformerModelHandler)
│   └── classifier.py             # Classification functions used in Streamlit app
├── config/                       # Configuration files for hyperparameters
│   └── hyperparams.py            # Stores training hyperparameters
├── training_pipeline/            # Folder for training pipeline
│   └── train.py                  # Main script to train custom models
├── logs/                         # Stores training logs
├── output/                       # Stores trained models and tokenizer output
├── notebooks/                    # Jupyter notebooks for experimentation
├── data/                         # Folder for datasets
├── .env                          # Environment variable configuration (model paths, etc.)
├── requirements.txt              # Dependencies for the project
└── README.md                     # Project documentation
```

## Running the Streamlit Application

Follow these steps to run the Streamlit application for complaint classification.

### Prerequisites

- Python 3.7+
- Required dependencies from `requirements.txt`
- Pre-trained models for Product, Sub-product, Issue, and Sub-issue classification

### Steps

1. **Clone the repository**:

   ```bash
   git clone https://github.com/Harshan1823/complaintTag.git
   cd complaintTag
   ```

2. **Install dependencies**:

   It's recommended to use a virtual environment. You can install the dependencies with:

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:

   Create a `.env` file in the root directory with paths to your model checkpoints:

   ```bash
   PRODUCT_MODEL_PATH=./product_model_checkpoint
   SUB_PRODUCT_MODEL_PATH=./sub_product_model_checkpoint
   ISSUE_MODEL_PATH=./issue_model_checkpoint
   SUB_ISSUE_MODEL_PATH=./sub_issue_model_checkpoint
   ```

4. **Run the Streamlit app**:

   Start the Streamlit app by running:

   ```bash
   streamlit run app.py
   ```

5. **Use the application**:

   - Open the app in your browser at `http://localhost:8501`.
   - Enter a customer complaint narrative, and the app will classify it into the corresponding product, sub-product, issue, and sub-issue categories.

## Training a Custom Model

Follow these steps to train a custom model for classifying CFPB customer complaints.

### Prerequisites

- Datasets for training and testing (in Pandas DataFrame format).
- Python 3.7+
- Pre-trained transformer model (e.g., BERT) for fine-tuning.

### Steps

1. **Prepare the Dataset**:

   Ensure your dataset is in a Pandas DataFrame format with the following columns:
   
   - `Consumer complaint narrative`: The complaint text.
   - `Product`, `Sub-product`, `Issue`, `Sub-issue`: The categories for classification.
   - `labels`: The label column representing the class for training.

2. **Modify Hyperparameters**:

   Open `config/hyperparams.py` to modify the hyperparameters, such as:

   - `MODEL_NAME`: Pre-trained model name (e.g., `'distilbert-base-uncased'`).
   - `BATCH_SIZE`: Training batch size.
   - `NUM_EPOCHS`: Number of training epochs.
   - `LEARNING_RATE`: Learning rate for the model.

3. **Run the Training Pipeline**:

   To train the model, execute the `train.py` script:

   ```bash
   python training_pipeline/train.py
   ```

4. **Check Logs and Outputs**:

   - Training logs will be stored in the `logs/` folder.
   - Trained models and tokenizer will be saved in the `output/` folder.

5. **Evaluate the Model**:

   After training, the model will be evaluated on the test dataset, and the evaluation metrics (accuracy, precision, recall, F1-score) will be printed.

6. **Use the Trained Model**:

   The trained model can be used for inference via the Streamlit application by updating the model paths in the `.env` file with the new checkpoints.

## Environment Setup

To replicate this project on your machine, follow these steps:

1. **Install dependencies**:

   Install all required Python packages with:

   ```bash
   pip install -r requirements.txt
   ```

2. **Set up your environment variables**:

   Use a `.env` file to specify paths for your pre-trained models (as described above).

## How to Contribute

If you'd like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Push your branch.
5. Create a Pull Request (PR).

We welcome all contributions that enhance this project!
