from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import torch
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import AutoTokenizer
import uvicorn
import os
from pathlib import Path

app = FastAPI(title="NLLB Premium Translator")

# Paths
MODEL_PATH = "models/nllb_onnx"
BASE_DIR = Path(__file__).parent

# Check if model exists
if os.path.exists(MODEL_PATH) and os.path.exists(os.path.join(MODEL_PATH, "encoder_model.onnx")):
    print(f"Loading ONNX model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = ORTModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
    print("✓ ONNX Model loaded successfully.")
else:
    print(f"⚠️ ONNX model not found. Attempting to load PyTorch base model...")
    # Fallback to base model if ONNX not ready
    MODEL_NAME = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    from transformers import AutoModelForSeq2SeqLM
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.eval()
    print(f"✓ PyTorch Base Model ({MODEL_NAME}) loaded successfully.")

# Setup templates
templates = Jinja2Templates(directory="templates")

class TranslationRequest(BaseModel):
    text: str
    src_lang: str
    tgt_lang: str

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/translate")
async def translate(request: TranslationRequest):
    if model is None:
        return {"translated_text": f"[DEMO MODE] Could not find model. Input was: {request.text}"}
    
    try:
        # Set languages
        tokenizer.src_lang = request.src_lang
        tgt_lang_token = request.tgt_lang
        tgt_token_id = tokenizer.convert_tokens_to_ids(tgt_lang_token)
        
        # Tokenize
        inputs = tokenizer(request.text, return_tensors="pt")
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=tgt_token_id,
                max_length=256,
                num_beams=5,
            )
        
        # Decode
        prediction = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return {"translated_text": prediction}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
