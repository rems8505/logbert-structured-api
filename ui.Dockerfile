FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install  -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["python", "gradio.py"]