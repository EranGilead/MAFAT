from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "avichr/heBERT"
target = Path("finetuning/models/hebert-baseline")

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)
tokenizer = AutoTokenizer.from_pretrained(model_name)

target.mkdir(parents=True, exist_ok=True)
model.save_pretrained(target)
tokenizer.save_pretrained(target)

print(f"Baseline model saved to {target}")
