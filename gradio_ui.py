import gradio as gr
# import sys
# from pathlib import Path

# # Add the app directory to the path
# file = Path(__file__).resolve()
# app_dir = file.parent.parent / "app"
# sys.path.append(str(app_dir))

import bert
import torch

# def predict_log(text):
#     # Ensure BERT model is loaded
#     if bert.tokenizer is None or bert.model is None:
#         bert.load_bert_model()
    
#     # Process input using the model from bert.py
#     inputs = bert.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
#     with torch.no_grad():
#         outputs = bert.model(**inputs)
    
#     logits = outputs.logits
#     prediction = torch.argmax(logits, dim=1).item()
    
#     # Return both the prediction and confidence scores
#     probabilities = torch.nn.functional.softmax(logits, dim=1)[0].tolist()
    
#     # Create a dictionary of class labels and their probabilities
#     class_names = ["Normal", "Anomaly"]  # Replace with your actual class names
#     result = {class_names[i]: round(prob * 100, 2) for i, prob in enumerate(probabilities)}
    
#     return result, class_names[prediction]

import gradio as gr
import requests

API_URL = "http://localhost:8002/bert/predict"

def predict_log(text):
    try:
        # Call the FastAPI server
        response = requests.post(API_URL, json={"text": text})
        response.raise_for_status()  # raises HTTPError if status is 4xx/5xx

        data = response.json()
        prediction = data["label"]
        logits = data["logits"]

        # Convert logits to probabilities
        import torch
        probabilities = torch.nn.functional.softmax(torch.tensor(logits), dim=1)[0].tolist()

        # Class names
        class_names = ["Normal", "Anomaly"]  # Adjust as needed
        result = {class_names[i]: round(prob * 100, 2) for i, prob in enumerate(probabilities)}

        return result, class_names[prediction]

    except requests.exceptions.RequestException as e:
        return {"Error": str(e)}, "Request Failed"

    except Exception as e:
        return {"Error": str(e)}, "Internal Error"


# Create Gradio Interface
demo = gr.Interface(
    fn=predict_log,
    inputs=gr.Textbox(lines=5, placeholder="Enter log text here..."),
    outputs=[
        gr.Label(label="Confidence Scores"),
        gr.Textbox(label="Prediction")
    ],
    title="LLogBERT Log Anomaly Detector",
    description="Enter a log message to classify it as normal or anomalous.",
    examples=[
        ["2025-07-20 10:23:45.123 2931 INFO nova.compute.manager [req-abc123] Starting instance: instance-001"],
        ["2025-07-20 10:23:45.123 2931 INFO nova.compute.manager [req-abc123] Starting instance: instance-001"],
    ]
)

if __name__ == "__main__":
    demo.launch(share=True)