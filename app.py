# ============================================
# FINANCIAL NER FASTAPI - HUGGING FACE DEPLOYMENT
# ============================================

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

import os
import gdown
import torch
import fitz

from transformers import AutoTokenizer, AutoModelForTokenClassification

# ============================================
# MODEL DOWNLOAD + LOAD
# ============================================

MODEL_DIR = "./financial_bert_ner"

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

MODEL_FILE = os.path.join(MODEL_DIR, "model.safetensors")

# ============================================
# GOOGLE DRIVE MODEL FILE
# ============================================

FILE_ID = "1-quGx0gWGK6iWX4ZILYc39QmhCBZCoUR"

if not os.path.exists(MODEL_FILE):

    url = f"https://drive.google.com/uc?id={FILE_ID}"

    gdown.download(url, MODEL_FILE, quiet=False)

model_path = MODEL_DIR

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForTokenClassification.from_pretrained(model_path)

model.eval()

id2label = model.config.id2label

print("✅ Financial NER Model Loaded Successfully")

# ============================================
# FASTAPI APP
# ============================================

app = FastAPI(
    title="Financial NER API",
    description="BERT-based Financial Named Entity Recognition",
    version="1.0"
)

# ============================================
# INPUT FORMAT
# ============================================

class TextRequest(BaseModel):
    text: str

# ============================================
# HOME ROUTE
# ============================================

@app.get("/")
def home():

    return {
        "message": "Financial NER API Running Successfully"
    }

# ============================================
# ENTITY EXTRACTION
# ============================================

def extract_entities(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)

    predictions = torch.argmax(outputs.logits, dim=2)

    tokens = tokenizer.convert_ids_to_tokens(
        inputs["input_ids"][0]
    )

    predicted_labels = [
        id2label[p.item()]
        for p in predictions[0]
    ]

    entities = []

    current_tokens = []
    current_label = None

    for token, label in zip(tokens, predicted_labels):

        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue

        if token.startswith("##"):

            if current_tokens:
                current_tokens[-1] += token[2:]

            continue

        clean_label = label.replace("B-", "").replace("I-", "")

        if label.startswith("B-"):

            if (
                current_tokens
                and current_label == clean_label
                and current_tokens[-1].endswith(".")
            ):

                current_tokens.append(token)

            else:

                if current_tokens and current_label:

                    entity_text = "".join(current_tokens)

                    entities.append({
                        "entity": entity_text,
                        "label": current_label
                    })

                current_tokens = [token]
                current_label = clean_label

        elif label.startswith("I-") and current_label:

            current_tokens.append(token)

        elif token == "." and current_label:

            current_tokens.append(token)

        elif (
            token.isdigit()
            and current_label
            and len(current_tokens) > 0
            and current_tokens[-1].endswith(".")
        ):

            current_tokens.append(token)

        else:

            if current_tokens and current_label:

                entity_text = "".join(current_tokens)

                entities.append({
                    "entity": entity_text,
                    "label": current_label
                })

            current_tokens = []
            current_label = None

    if current_tokens and current_label:

        entity_text = "".join(current_tokens)

        entities.append({
            "entity": entity_text,
            "label": current_label
        })

    return entities

# ============================================
# RAW TEXT ROUTE
# ============================================

@app.post("/predict_text")
def predict_text(request: TextRequest):

    text = request.text[:5000]

    entities = extract_entities(text)

    return {
        "text": text,
        "entities": entities
    }

# ============================================
# TXT FILE ROUTE
# ============================================

@app.post("/predict_txt")
async def predict_txt(file: UploadFile = File(...)):

    content = await file.read()

    text = content.decode("utf-8")[:5000]

    entities = extract_entities(text)

    return {
        "filename": file.filename,
        "entities": entities
    }

# ============================================
# PDF FILE ROUTE
# ============================================

@app.post("/predict_pdf")
async def predict_pdf(file: UploadFile = File(...)):

    pdf_bytes = await file.read()

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    text = ""

    for page in doc:
        text += page.get_text()

    text = text[:5000]

    entities = extract_entities(text)

    return {
        "filename": file.filename,
        "entities": entities
    }
