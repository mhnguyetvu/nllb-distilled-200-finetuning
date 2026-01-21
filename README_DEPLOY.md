# Deploying NLLB Translator Web App

This project provides a premium web interface for translating between Vietnamese and Korean (and other languages) using a fine-tuned NLLB model converted to ONNX format.

## Prerequisites

- Python 3.10+
- Virtual environment activated

## Installation

```bash
pip install "optimum[onnxruntime]" fastapi uvicorn python-multipart jinja2
```

## Step 1: Export Model to ONNX

If you have a fine-tuned model checkpoint, run:

```bash
python training/export_onnx.py --model path/to/your/checkpoint --output models/nllb_onnx
```

Otherwise, you can use the base model for testing:

```bash
python training/export_onnx.py --model facebook/nllb-200-distilled-600M --output models/nllb_onnx
```

## Step 2: Run the Web App

Start the FastAPI server:

```bash
python app.py
```

Open your browser and navigate to `http://localhost:8000`.

## Features

- **Premium UI:** Modern glassmorphism design with Dark Mode support.
- **ONNX Optimization:** Faster inference on CPU.
- **Real-time Translation:** Auto-translates as you type (with debounce).
- **Multi-language Support:** Ready for Vietnamese, Korean, and English.
