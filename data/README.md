# Dataset Instructions

## Recommended real dataset

This project is designed to work with the publicly available Kaggle dataset:

**"Sentiment Analysis for Mental Health"**
Source: https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health

- ~53,000 social-media-style text statements
- Text column: `statement`
- Label column: `status`
- Classes typically include: `Normal`, `Anxiety`, `Depression`, `Suicidal`,
  `Stress`, `Bipolar`, `Personality disorder`

### How to install it

1. Create a free Kaggle account (if you don't already have one).
2. Download the dataset CSV from the link above (you may need `kaggle.json`
   API credentials, or just click "Download" on the website).
3. Rename the downloaded file to `dataset.csv`.
4. Place it in this folder: `data/dataset.csv`.
5. Open `src/config.py` and confirm the column names match:
   ```python
   TEXT_COLUMN = "statement"
   TARGET_COLUMN = "status"
   ```
   (the default config ships with `text` / `label`; update it to match
   whichever CSV you actually place here).
6. Run the training pipeline:
   ```bash
   python -m src.train
   ```

You may substitute **any** other labeled text dataset for mental-health /
social-media risk signal classification (e.g. the "Mental Health Corpus"
or "Reddit Mental Health" datasets on Kaggle) as long as you update
`TEXT_COLUMN` and `TARGET_COLUMN` in `src/config.py` to match its headers.

## Included demo dataset (`dataset_sample.csv`)

Because this environment has no internet access to download the real
Kaggle dataset automatically, a **small, clearly-labeled synthetic demo
dataset** (`dataset_sample.csv`, ~300 short example statements across
4 categories) is included so that:

- the full pipeline (EDA -> preprocessing -> TF-IDF -> training ->
  evaluation -> dashboard) can be run and demonstrated immediately, and
- you can verify every part of the project works end-to-end before
  plugging in the real dataset.

**This demo dataset is for pipeline demonstration only.** It is small,
synthetic, and not a substitute for a real published dataset. Any
accuracy/F1 numbers produced from it should not be reported as
representative research results — they simply prove the pipeline runs
correctly. For a real mini-project submission, download the real Kaggle
dataset above and re-run `python -m src.train`.

If `data/dataset.csv` is not found, the application automatically falls
back to `data/dataset_sample.csv` and displays a visible banner in the UI
stating that it is running on demo data.
