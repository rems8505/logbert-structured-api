FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install  -r requirements.txt

COPY . .
COPY gradio_ui.py /app/gradio_ui.py

EXPOSE 7860

CMD ["python", "gradio_ui.py"]