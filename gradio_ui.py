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
import os
import requests
from llm import analyze_log  # Import your RCA function

API_URL = "http://localhost:8000/bert/predict"

api_host = os.environ.get("API_HOST", "http://localhost:8000")

#####
#Reson for commenting this,
# our current predict_log() function computes everything correctly, but it's missing a final return statement in the successful path.
# That’s why Gradio gets [None] — it falls off the function without returning anything when no exception occurs.
# 
# def predict_log(text):
#     try:
#         # Call the FastAPI server
#         response = requests.post( f"{api_host}/bert/predict", json={"text": text})
#         response.raise_for_status()  # raises HTTPError if status is 4xx/5xx

#         data = response.json()
#         prediction = data["label"]
#         logits = data["logits"]

#         # Convert logits to probabilities
#         import torch
#         probabilities = torch.nn.functional.softmax(torch.tensor(logits), dim=1)[0].tolist()

#         # Class names
#         class_names = ["Normal", "Anomaly"]  # Adjust as needed
#         result = {class_names[i]: round(prob , 2) for i, prob in enumerate(probabilities)}

#     except requests.exceptions.RequestException as e:
#         return {"Error": str(e)}, "Request Failed"

#     except Exception as e:
#         return {"Error": str(e)}, "Internal Error"

def predict_log(text):
    try:
        # Call the FastAPI server
        response = requests.post(f"{api_host}/bert/predict", json={"text": text})
        response.raise_for_status()

        data = response.json()
        prediction = data["label"]
        logits = data["logits"]

        # Convert logits to probabilities
        import torch
        probabilities = torch.nn.functional.softmax(torch.tensor(logits), dim=1)[0].tolist()

        # Class names
        class_names = ["Normal", "Anomaly"]
        result = {class_names[i]: round(prob, 2) for i, prob in enumerate(probabilities)}

        # ✅ Add this return line
        return prediction, str(result)

    except requests.exceptions.RequestException as e:
        return "Error", str(e)

    except Exception as e:
        return "Error", str(e)


# RCA processing logic
def process_log(log_input):
    result = analyze_log(log_input)
    if "error" in result:
        return f"Error: {result['error']}\nDetails: {result['details']}"
    return f"🧠 Root Cause:\n{result['root_cause']}\n\n🔧 Suggested Fix:\n{result['suggested_fix']}"



# # Create Gradio Interface
# demo = gr.Interface(
#     fn=predict_log,
#     inputs=gr.Textbox(lines=5, placeholder="Enter log text here..."),
#     outputs=[
#         gr.Label(label="Confidence Scores"),
#         gr.Textbox(label="Prediction")
#     ],
#     title="LLogBERT Log Anomaly Detector",
#     description="Enter a log message to classify it as normal or anomalous.",
#     examples=[
#         ["2025-07-20 10:23:45.123 2931 INFO nova.compute.manager [req-abc123] Starting instance: instance-001"],
#         ["2025-07-20 10:23:45.123 2931 INFO nova.compute.manager [req-abc123] Starting instance: instance-001"],
#     ]
# )


# --- Interface 1: Anomaly Detection ---
anomaly_ui = gr.Interface(
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

# --- Interface 2: LLM-based RCA ---
rca_ui = gr.Interface(
    fn=process_log,
    inputs=gr.Textbox(lines=5, placeholder="Paste anomalous log here..."),
    outputs=gr.Textbox(lines=15, label="LLM Output"),
    title="Root Cause Analysis (LLM-powered)",
    description="Uses Gemini LLM to analyze anomalous logs and suggest root causes + fixes.",
    examples=[
        "nova-compute.log.2017-05-14_21:27:09 2017-01-18 21:05:51 17409 CRITICAL cinder [-] Bad or unexpected response from the storage volume backend API: volume group cinder-volumes doesnt exist",
    ]
)

# Combine both UIs using tabs
demo = gr.TabbedInterface([anomaly_ui, rca_ui], ["Anomaly Detection", "Root Cause Analysis"])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
