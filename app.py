# ============================================
# FINANCIAL NER FASTAPI - FINAL COMPLETE API
# SUPPORTS:
# 1. Raw Text
# 2. TXT File Upload
# 3. PDF File Upload
# ============================================

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

import fitz  # PyMuPDF

# ============================================
# LOAD SAVED MODEL
# ============================================

model_path = r"C:\Users\HP\ner\financial_bert_ner"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)

model.eval()

# LABEL MAPPING
id2label = model.config.id2label

print("✅ Financial NER Model Loaded Successfully")


# ============================================
# CREATE FASTAPI APP
# ============================================

app = FastAPI(
    title="Financial NER API",
    description="BERT-based Financial Named Entity Recognition",
    version="1.0"
)


# ============================================
# INPUT FORMAT FOR RAW TEXT
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
# ENTITY EXTRACTION FUNCTION
# ============================================

def extract_entities(text):

    # ========================================
    # TOKENIZATION
    # ========================================
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    # ========================================
    # MODEL PREDICTION
    # ========================================
    with torch.no_grad():
        outputs = model(**inputs)

    predictions = torch.argmax(outputs.logits, dim=2)

    # ========================================
    # TOKENS + LABELS
    # ========================================
    tokens = tokenizer.convert_ids_to_tokens(
        inputs["input_ids"][0]
    )

    predicted_labels = [
        id2label[p.item()]
        for p in predictions[0]
    ]

    # ========================================
    # ENTITY MERGING
    # ========================================
    entities = []

    current_tokens = []
    current_label = None

    for token, label in zip(tokens, predicted_labels):

        # Skip special tokens
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue

        # ====================================
        # HANDLE SUBWORD TOKENS
        # ====================================
        if token.startswith("##"):

            if current_tokens:
                current_tokens[-1] += token[2:]

            continue

        # Remove B-/I-
        clean_label = label.replace("B-", "").replace("I-", "")

        # ====================================
        # BEGIN ENTITY
        # ====================================
        if label.startswith("B-"):

            # Decimal continuation
            if (
                current_tokens
                and current_label == clean_label
                and current_tokens[-1].endswith(".")
            ):

                current_tokens.append(token)

            else:

                # Save previous entity
                if current_tokens and current_label:

                    entity_text = "".join(current_tokens)

                    entities.append({
                        "entity": entity_text,
                        "label": current_label
                    })

                current_tokens = [token]
                current_label = clean_label

        # ====================================
        # INSIDE ENTITY
        # ====================================
        elif label.startswith("I-") and current_label:

            current_tokens.append(token)

        # ====================================
        # HANDLE DECIMAL POINT
        # ====================================
        elif token == "." and current_label:

            current_tokens.append(token)

        # ====================================
        # HANDLE NUMBER AFTER DECIMAL
        # ====================================
        elif (
            token.isdigit()
            and current_label
            and len(current_tokens) > 0
            and current_tokens[-1].endswith(".")
        ):

            current_tokens.append(token)

        # ====================================
        # OUTSIDE ENTITY
        # ====================================
        else:

            if current_tokens and current_label:

                entity_text = "".join(current_tokens)

                entities.append({
                    "entity": entity_text,
                    "label": current_label
                })

            current_tokens = []
            current_label = None

    # ========================================
    # SAVE FINAL ENTITY
    # ========================================
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

    text = request.text

    # Limit size
    text = text[:5000]

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

    # READ FILE
    content = await file.read()

    text = content.decode("utf-8")

    # Limit size
    text = text[:5000]

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

    # READ PDF
    pdf_bytes = await file.read()

    # OPEN PDF
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    # EXTRACT TEXT
    text = ""

    for page in doc:
        text += page.get_text()

    # Limit size
    text = text[:5000]

    entities = extract_entities(text)

    return {
        "filename": file.filename,
        "entities": entities
    }


# ============================================
# RUN SERVER:
#
# uvicorn app:app --reload
#
# OPEN:
# http://127.0.0.1:8000/docs
# ============================================