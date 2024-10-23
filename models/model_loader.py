from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access environment variables
product_model_path = os.getenv('PRODUCT_MODEL_PATH')
sub_product_model_path = os.getenv('SUB_PRODUCT_MODEL_PATH')
issue_model_path = os.getenv('ISSUE_MODEL_PATH')
sub_issue_model_path = os.getenv('SUB_ISSUE_MODEL_PATH')

def load_tokenizer():
    """Loads and returns the tokenizer."""
    return AutoTokenizer.from_pretrained(product_model_path)

def load_models():
    """Loads and returns a dictionary of models."""
    product_model = AutoModelForSequenceClassification.from_pretrained(product_model_path)
    sub_product_model = AutoModelForSequenceClassification.from_pretrained(sub_product_model_path)
    issue_model = AutoModelForSequenceClassification.from_pretrained(issue_model_path)
    sub_issue_model = AutoModelForSequenceClassification.from_pretrained(sub_issue_model_path)

    return [
        ("Product Model", product_model, "product"),
        ("Sub-Product Model", sub_product_model, "sub_product"),
        ("Issue Model", issue_model, "issue"),
        ("Sub-Issue Model", sub_issue_model, "sub_issue")
    ]
