# infer.py
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "models/distilbert-imdb"  # change if needed
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = ["negative", "positive"]
MAX_LENGTH = 256

def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=False)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=False)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model

def predict_label(text, tokenizer, model):
    enc = tokenizer([text], truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt")
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
    idx = int(probs.argmax())
    return LABELS[idx]

def main():
    if len(sys.argv) < 2:
        print("Usage: python infer.py \"your sentence here\"")
        sys.exit(1)
    text = " ".join(sys.argv[1:])
    try:
        tokenizer, model = load_model_and_tokenizer()
    except Exception as e:
        print("Error loading model/tokenizer:", e)
        sys.exit(1)

    label = predict_label(text, tokenizer, model)
    print(label)

if __name__ == "__main__":
    main()
