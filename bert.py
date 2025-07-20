from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from transformers import BertTokenizerFast, BertForSequenceClassification, BertConfig
import data_validation.parser as parser

import torch

from bert_config import bert_settings

router = APIRouter()

class InputText(BaseModel):
    text: str

# Global variables for model components
tokenizer = None
model = None

def load_bert_model():
    """Load BERT model components"""
    global tokenizer, model
    import os
    project_root = "/Users/akhilvinayak/work/python/logbert-structured-api/app"
    
    # Load tokenizer
    try:
        # tokenizer = BertTokenizerFast.from_pretrained(
        #     project_root,
        #     tokenizer_file=os.path.join(project_root, bert_settings.tokenizer_file),
        # )
        tokenizer = BertTokenizerFast.from_pretrained(
            ".",
            tokenizer_file= bert_settings.tokenizer_file,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizer: {e}")

    # Load model configuration
    try:
        config = BertConfig.from_json_file(bert_settings.model_config_file)
    except Exception as e:
        raise RuntimeError(f"Failed to load model config: {e}")

    # Initialize model and load weights
    try:
        model = BertForSequenceClassification(config)
        state_dict = torch.load(bert_settings.model_weights_file, map_location="cpu")
        
        load_result = model.load_state_dict(state_dict, strict=False)
        print("❌ Missing keys:", load_result.missing_keys)
        print("❗ Unexpected keys:", load_result.unexpected_keys)
        
        model.eval()
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

@router.get("/")
def bert_health():
    return {"message": "BERT API is running!"}



@router.post("/predict")
def predict_bert(req: InputText):
    if tokenizer is None or model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    # Validate log format before prediction
    # Validate log format before prediction
    try:
        _ = parser.parse_log_line(req.text)
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=f"Invalid log format: {str(ve)}")
    try:
        inputs = tokenizer(
            req.text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=bert_settings.max_length
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
        
        return {"label": prediction, "logits": logits.tolist()}
    # except ValidationError as ve:
    #     raise HTTPException(status_code=422, detail=f"Validation failed: {ve.errors()}")

    except ValueError as ve:
        raise HTTPException(status_code=422, detail=f"Invalid log format: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")