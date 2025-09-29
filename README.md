# DistilBERT Sentiment Classifier (IMDB) + Streamlit + LIME

## What

Fine-tune DistilBERT on IMDB movie reviews (binary sentiment). Demo with Streamlit + LIME explanations showing word-level contributions.

## Run (fast)

1. Install dependencies:
   pip install -r requirements.txt

2. Train (full dataset):
   python train.py
   (or for quick dev, edit train.py to use a subset)

3. Try CLI inference:
   python infer.py "I loved the movie!"

4. Run demo:
   streamlit run app.py

## Notes

- For development use smaller dataset slices to iterate faster.
- LIME explanation is an approximation; for more precise explanations consider integrated gradients / SHAP / attention visualization.
