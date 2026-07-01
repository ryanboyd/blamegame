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

After importing the package and initializing the `AttributionAnalyzer()` class, you can process individual texts:

### 2a. Analyze Text

```python
text = "She failed the test because she was unlucky."
sentence_results, overall_results = analyzer.process_text(text)
print(sentence_results)
```

... or batch process texts in a CSV file:

### 2b. Analyze a CSV File

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
| :---         |     :---:      |     :---:      |     :---:      |     :---:      |
| "She failed the test because she was unlucky." | 0.00       | 0.00     | 0.41       | 0.75      |
| "I failed the exam because I didn't try hard enough."  | 0.99       | 0.05     | 0.00       | 0.00      |


### Overall/Average Predictions


| Text                                                        | Num Sentences | Self-Disp. (M. Prediction) | Self-Sit. (M. Prediction) | Other-Disp. (M. Prediction) | Other-Sit. (M. Prediction) |
| :---         |     :---:      |     :---:      |     :---:      |     :---:      |     :---:      |
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

```bibtex
@article{telidevara_cultural_2026,
	title = {A cultural explanation for parole decisions in the {United} {States}},
	volume = {5},
	issn = {2752-6542},
	url = {https://doi.org/10.1093/pnasnexus/pgag193},
	doi = {10.1093/pnasnexus/pgag193},
	abstract = {Racial disparities in discretionary parole decisions persist, yet their cultural underpinnings remain poorly understood. The US justice system, shaped by White American cultural norms, may implicitly favor parole candidates who explain their crimes through dispositional rather than situational attributions. Importantly, however, findings from studies of cultural psychology suggest that while White Americans tend to frame behavior in dispositional terms, Black, Hispanic, and East Asian Americans are more likely to acknowledge situational contributors. Across four studies, we tested whether these attributional differences can contribute to racial disparities in parole. In studies 1 and 2 (total n = 588), White American mock parole candidates were more likely to use dispositional attributions to explain their hypothetical crimes, while those from racial and cultural minorities either showed no preference or favored situational explanations. In study 3 (n = 512), White American mock parole commissioners perceived candidates providing dispositional (vs. situational) explanations about their crimes as less likely to reoffend. In study 4, analyzing 183 real parole hearings via natural language processing methods, we found that candidates granted parole used dispositional framing of their past criminal offenses more often. These findings highlight a cultural mismatch between minority parole candidates and the expectations of the US justice system, potentially perpetuating racial disparities. Addressing this misalignment may offer a new pathway for reducing systemic bias in discretionary parole decisions.},
	number = {7},
	urldate = {2026-07-01},
	journal = {PNAS Nexus},
	author = {Telidevara, Amrita Kaushik and Boyd, Ryan L and Park, BoKyung},
	month = jul,
	year = {2026},
	pages = {pgag193},
}
```


---

## 🤝 Contributing

Feel free to contribute to BlameGame! Open a pull request or file an issue on GitHub.

---

Enjoy using **BlameGame** for attribution detection!

