import csv
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize
import os
import torch
import re
from datetime import datetime
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# Download necessary NLTK data silently
nltk.download('punkt', quiet=True)


def sanitize_filename(filename, max_length=255):
    """
    Sanitize the filename by replacing invalid characters and limiting its length.
    """
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '_', filename)
    sanitized = sanitized.strip(' .')

    # Reserved filenames in Windows
    reserved_names = {
        "CON", "PRN", "AUX", "NUL",
        "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
        "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"
    }
    if sanitized.upper() in reserved_names:
        sanitized = f"_{sanitized}_"

    # Truncate long filenames while preserving extensions
    if len(sanitized) > max_length:
        ext_index = sanitized.rfind('.')
        if ext_index == -1 or ext_index > max_length - 5:
            sanitized = sanitized[:max_length]
        else:
            name_part = sanitized[:max_length - (len(sanitized) - ext_index)]
            sanitized = f"{name_part}{sanitized[ext_index:]}"
    return sanitized


class AttributionAnalyzer:
    def __init__(self, model_name="ryanboyd/AttributioNet", hf_token=None):
        """
        Initialize the Analyzer class with a specified model.
        :param model_name: The huggingface model name being used for this task. The default name is expected.
        :param hf_token: Your huggingface token, if necessary
        """
        self.model_name = model_name
        self.hf_token = hf_token
        self.sanitized_model_name = sanitize_filename(model_name)
        self._load_model()

    def _load_model(self):
        """
        Load the pre-trained model and tokenizer from Hugging Face.
        """
        print(f"Loading model: {self.model_name}...")
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name, token=self.hf_token)
        self.model = RobertaForSequenceClassification.from_pretrained(self.model_name, token=self.hf_token)
        self.model.eval()  # Set the model to evaluation mode
        print("Model loaded successfully!")

    def process_text(self, text: str):
        """
        Tokenize the input text into sentences and classify each using the model.
        """
        sentences = sent_tokenize(text)
        sentence_results = []
        overall_results = {
            "text": " ".join(sentences),
            "num_sentences": len(sentences),
            "self_dispositional_prob": 0,
            "self_dispositional_pred": 0,
            "self_situational_prob": 0,
            "self_situational_pred": 0,
            "other_dispositional_prob": 0,
            "other_dispositional_pred": 0,
            "other_situational_prob": 0,
            "other_situational_pred": 0,
        }

        for sentence in sentences:
            # Tokenize the sentence and process it with the model
            inputs = self.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)
            with torch.no_grad():  # Disable gradient calculations for efficiency
                logits = self.model(**inputs).logits
            probabilities = torch.sigmoid(logits).squeeze().tolist()
            predictions = [1 if prob >= 0.5 else 0 for prob in probabilities]

            # Store sentence-level results
            sentence_result = {
                "self_dispositional_prob": probabilities[0],
                "self_dispositional_pred": predictions[0],
                "self_situational_prob": probabilities[1],
                "self_situational_pred": predictions[1],
                "other_dispositional_prob": probabilities[2],
                "other_dispositional_pred": predictions[2],
                "other_situational_prob": probabilities[3],
                "other_situational_pred": predictions[3],
            }


            # Aggregate results for overall text-level analysis
            for key in overall_results:
                if key in sentence_result:
                    overall_results[key] += sentence_result[key] / len(sentences)

            # now that we've aggregated the values into the "overall_results" dictionary,
            # we can add the sentence text to the sentence-level result
            sentence_result["sentence_text"] = sentence
            sentence_results.append(sentence_result)

        return sentence_results, overall_results

    def process_csv(self, input_csv, row_id_col, text_col, file_encoding="utf-8", output_dir="output"):
        """
        Process a CSV file containing text data, classifying each entry and saving results.
        :param input_csv: The path of the CSV file that you would like to analyze
        :param row_id_col: The header of the column containing your row identifiers (e.g., SubjectNumber)
        :param text_col: The header of the column containing your text (e.g., ParticipantResponse)
        :param output_dir: The folder where you would like all of your output to be saved
        :return:
        """
        execution_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        sentence_predictions_filename = f'{output_dir}/{execution_datetime} - sentence_predictions.csv'
        overall_predictions_filename = f'{output_dir}/{execution_datetime} - overall_predictions.csv'
        print(f"\nProcessing input file: {input_csv}")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        with open(input_csv, 'r', encoding='utf-8') as infile, \
                open(sentence_predictions_filename, 'w', encoding=file_encoding, newline='') as sent_prediction_outfile, \
                open(overall_predictions_filename, 'w', encoding=file_encoding, newline='') as overall_prediction_outfile:

            reader = csv.DictReader(infile)
            sentence_prediction_writer = csv.writer(sent_prediction_outfile)
            overall_prediction_writer = csv.writer(overall_prediction_outfile)

            # Write headers for output CSV files
            sentence_prediction_writer.writerow([
                "row_id", "sentence",
                "self_dispositional_pred", "self_situational_pred",
                "other_dispositional_pred", "other_situational_pred",
                "self_dispositional_prob", "self_situational_prob",
                "other_dispositional_prob", "other_situational_prob"
            ])
            overall_prediction_writer.writerow([
                "row_id", "text", "num_sentences",
                "self_dispositional_pred", "self_situational_pred",
                "other_dispositional_pred", "other_situational_pred",
                "self_dispositional_prob", "self_situational_prob",
                "other_dispositional_prob", "other_situational_prob"
            ])

            for row in tqdm(reader, desc="Analyzing dataset"):
                text = row[text_col] or ""  # Ensure text is not None
                row_id = row[row_id_col]
                sentence_results, overall_results = self.process_text(text)

                # Write sentence-level predictions
                for result in sentence_results:
                    sentence_prediction_writer.writerow([row_id, result["sentence_text"],
                                                         result["self_dispositional_pred"],
                                                         result["self_situational_pred"],
                                                         result["other_dispositional_pred"],
                                                         result["other_situational_pred"],
                                                         result["self_dispositional_prob"],
                                                         result["self_situational_prob"],
                                                         result["other_dispositional_prob"],
                                                         result["other_situational_prob"]])

                # Write overall predictions
                overall_prediction_writer.writerow([row_id, overall_results["text"], overall_results["num_sentences"],
                                                    overall_results["self_dispositional_pred"],
                                                    overall_results["self_situational_pred"],
                                                    overall_results["other_dispositional_pred"],
                                                    overall_results["other_situational_pred"],
                                                    overall_results["self_dispositional_prob"],
                                                    overall_results["self_situational_prob"],
                                                    overall_results["other_dispositional_prob"],
                                                    overall_results["other_situational_prob"]])

        print(f"Finished processing: {input_csv}\nOutput stored in: {output_dir}")
