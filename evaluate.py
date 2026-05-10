# ===============================
# 1. IMPORT LIBRARIES
# ===============================
import torch
import numpy as np
import ast
import re
import pandas as pd
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForTokenClassification
from datasets import load_dataset, Dataset
from seqeval.metrics import classification_report

# ===============================
# 2. LOAD MODEL
# ===============================
model_path = "financial_bert_ner"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)

model.eval()

# ===============================
# 3. LABEL MAPPING
# ===============================
id2label = model.config.id2label
label2id = model.config.label2id

# ===============================
# 4. LOAD DATASET
# ===============================
dataset = load_dataset("Josephgflowers/Financial-NER-NLP")["train"]
dataset = dataset.shuffle(seed=42).select(range(20000))

# ===============================
# 5. CONVERT TO NER FORMAT
# ===============================
def convert_to_ner(data):
    new_data = []
    for item in data:
        text = item["user"]
        words = text.split()
        labels = ["O"] * len(words)

        try:
            entities = ast.literal_eval(item["assistant"])
            for label, values in entities.items():
                for value in values:
                    for match in re.finditer(re.escape(str(value)), text):
                        entity_words = text[match.start():match.end()].split()
                        for i in range(len(words)):
                            if words[i:i+len(entity_words)] == entity_words:
                                labels[i] = "B-" + label
                                for j in range(1, len(entity_words)):
                                    if i+j < len(labels):
                                        labels[i+j] = "I-" + label
        except:
            continue

        new_data.append({"tokens": words, "ner_tags": labels})

    return new_data


ner_data = convert_to_ner(dataset)
dataset = Dataset.from_list(ner_data)

# ===============================
# 6. TOKENIZATION
# ===============================
def tokenize_and_align_labels(example):
    tokenized = tokenizer(
        example["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=128
    )

    word_ids = tokenized.word_ids()
    labels = []
    prev_word = None

    for word_id in word_ids:
        if word_id is None:
            labels.append(-100)
        elif word_id != prev_word:
            labels.append(label2id.get(example["ner_tags"][word_id], 0))
        else:
            labels.append(-100)

        prev_word = word_id

    tokenized["labels"] = labels
    return tokenized


dataset = dataset.map(tokenize_and_align_labels, remove_columns=["tokens", "ner_tags"])
dataset = dataset.train_test_split(test_size=0.2)
test_dataset = dataset["test"]

# ===============================
# 7. PREDICTIONS
# ===============================
predictions = []
true_labels = []

for item in test_dataset:
    inputs = {
        "input_ids": torch.tensor([item["input_ids"]]),
        "attention_mask": torch.tensor([item["attention_mask"]])
    }

    with torch.no_grad():
        outputs = model(**inputs)

    preds = torch.argmax(outputs.logits, dim=2).numpy()[0]
    labels = item["labels"]

    pred_labels = []
    true_lab = []

    for p, l in zip(preds, labels):
        if l != -100:
            pred_labels.append(id2label.get(p, "O"))
            true_lab.append(id2label.get(l, "O"))

    predictions.append(pred_labels)
    true_labels.append(true_lab)

# ===============================
# 8. REPORT
# ===============================
print("\n===== FULL CLASSIFICATION REPORT =====")
print(classification_report(true_labels, predictions, digits=4))

report = classification_report(true_labels, predictions, output_dict=True)

# ===============================
# 9. DATAFRAMES
# ===============================
full_rows = []
filtered_rows = []

for label, metrics in report.items():
    if isinstance(metrics, dict):
        row = {
            "Label": label,
            "Precision": round(metrics["precision"], 4),
            "Recall": round(metrics["recall"], 4),
            "F1-score": round(metrics["f1-score"], 4),
            "Support": int(metrics["support"])
        }

        full_rows.append(row)

        if metrics["f1-score"] > 0:
            filtered_rows.append(row)

full_df = pd.DataFrame(full_rows)
filtered_df = pd.DataFrame(filtered_rows)

# ===============================
# 10. SUMMARY
# ===============================
summary = []
for avg in ["micro avg", "macro avg", "weighted avg"]:
    if avg in report:
        m = report[avg]
        summary.append({
            "Type": avg,
            "Precision": round(m["precision"], 4),
            "Recall": round(m["recall"], 4),
            "F1-score": round(m["f1-score"], 4)
        })

summary_df = pd.DataFrame(summary)

# ===============================
# 11. SAVE CSV
# ===============================
full_df.to_csv("full_report_all_labels.csv", index=False)
filtered_df.to_csv("filtered_report_no_zero.csv", index=False)
summary_df.to_csv("summary_report.csv", index=False)

print("\n✅ CSV files created")

# ===============================
# 12. GRAPHS
# ===============================
plt.figure()
plt.hist(full_df["Precision"], bins=20)
plt.title("Precision Distribution")
plt.savefig("precision_distribution.png")
plt.close()

top10 = filtered_df.sort_values(by="F1-score", ascending=False).head(10)
plt.figure()
plt.barh(top10["Label"], top10["F1-score"])
plt.title("Top 10 Labels")
plt.gca().invert_yaxis()
plt.savefig("top10_labels.png")
plt.close()

worst10 = filtered_df.sort_values(by="F1-score").head(10)
plt.figure()
plt.barh(worst10["Label"], worst10["F1-score"])
plt.title("Worst 10 Labels")
plt.gca().invert_yaxis()
plt.savefig("worst10_labels.png")
plt.close()

plt.figure()
plt.scatter(full_df["Precision"], full_df["Recall"])
plt.title("Precision vs Recall")
plt.savefig("precision_vs_recall.png")
plt.close()

print("\n✅ Graphs created")

# ===============================
# 13. CONFUSION MATRIX
# ===============================
true_flat = [t for seq in true_labels for t in seq]
pred_flat = [p for seq in predictions for p in seq]

labels = sorted(list(set(true_flat + pred_flat)))
label_to_idx = {l: i for i, l in enumerate(labels)}

cm = np.zeros((len(labels), len(labels)), dtype=int)

for t, p in zip(true_flat, pred_flat):
    cm[label_to_idx[t]][label_to_idx[p]] += 1

max_labels = 20
display_labels = labels[:max_labels]

plt.figure(figsize=(10, 8))
plt.imshow(cm[:max_labels, :max_labels])
plt.title("Confusion Matrix")
plt.colorbar()

plt.xticks(range(len(display_labels)), display_labels, rotation=90)
plt.yticks(range(len(display_labels)), display_labels)

plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

print("\n✅ Confusion matrix created")