# BlameGame: Sentence-Level Attribution Detection

**BlameGame** is a Python package for detecting internal/external (dispositional/situational) attributions in text at the sentence level. Whether you're exploring psychological patterns or analyzing narrative frameworks, BlameGame leverages a fine-tuned RoBERTa model to classify attribution types with high accuracy.

---

## 🎯 Features

- **Sentence-Level Attribution**: Detects situational and dispositional attributions within individual sentences.
- **Transformer-Based Predictions**: Utilizes a fine-tuned RoBERTa model for robust and context-aware classification.
- **Seamless Processing**: Handles large datasets with ease and outputs results in CSV format.
- **Pretrained NLP Model**: Leverages a Hugging Face-hosted model (`ryanboyd/AttributioNet`) for precise attribution analysis. Model details can be found at [https://huggingface.co/ryanboyd/AttributioNet](https://huggingface.co/ryanboyd/AttributioNet)

---

## 🚀 Installation

Install BlameGame directly via pip:

```sh
pip install blamegame
```

---

## 🔍 Usage

### 1. Import and Initialize

```python
from blamegame.BlameGame import AttributionAnalyzer

analyzer = AttributionAnalyzer()
```

### 2. Analyze Text

```python
text = "She failed the test because she was unlucky."
sentence_results, overall_results = analyzer.process_text(text)
print(sentence_results)
```

### 3. Analyze a CSV File

```python
analyzer.process_csv(input_csv="data.csv", row_id_col="ID", text_col="Text", output_dir="attribution-results/")
```

---

## 📜 How It Works

BlameGame tokenizes input text into sentences and uses a fine-tuned RoBERTa model to classify each sentence into one (or more) of four attribution types:

- **Self-Dispositional** (attributing the cause of one's own behavior to their traits/personality)
- **Self-Situational** (attributing the cause of one's own behavior to situational/circumstantial factors)
- **Other-Dispositional** (attributing the cause of someone else's behavior to their traits/personality)
- **Other-Situational** (attributing the cause of someone else's behavior to situational/circumstantial factors)

The model outputs both classification labels (0 or 1) and probability scores for each sentence, which are then also aggregated to provide an overall attribution summary for the input text. The classification labels reflect whether a sentence did (1) or did not (0) contain at least one attribution of each given type.

---

## 📌 Example Output

### Sentence-Level Probabilities

| Sentence                                  | Self-Disp. (Probability) | Self-Sit. (Probability) | Other-Disp. (Probability) | Other-Sit. (Probability) |
|-------------------------------------------|------------|----------|------------|-----------|
| "She failed the test because she was unlucky." | 0.00       | 0.00     | 0.41       | 0.75      |
| "I failed the exam because I didn't try hard enough."  | 0.99       | 0.05     | 0.0.00       | 0.00      |


### Overall/Average Predictions


| Text                                                        | Num Sentences | Self-Disp. (M. Prediction) | Self-Sit. (M. Prediction) | Other-Disp. (M. Prediction) | Other-Sit. (M. Prediction) |
|------------------------------------------------------------|--------------|------------|----------|------------|-----------|
| "She failed the test because she was unlucky. I failed the exam because I didn't try hard enough." | 2            | 0.50       | 0.00     | 0.00       | 0.50      |


---

## 📖 Dependencies

BlameGame relies on the following libraries:

- `transformers` (for RoBERTa model loading)
- `torch` (for deep learning computations)
- `nltk` (for sentence tokenization)
- `tqdm` (for progress tracking)

These dependencies are installed automatically when using `pip install blamegame`.

---

## 🎓 Citation

Coming soon 😄


---

## 🤝 Contributing

Feel free to contribute to BlameGame! Open a pull request or file an issue on GitHub.

---

Enjoy using **BlameGame** for attribution detection!

