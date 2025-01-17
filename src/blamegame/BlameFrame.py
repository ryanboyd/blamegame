import csv
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
import joblib
import os
import re
from datetime import datetime

nltk.download('punkt', quiet=True)

class BlameFrame:
    def __init__(self, model_name="dmlls/all-mpnet-base-v2-negation"):
        """
        Initializes the BlameFrame class with pre-trained models and configurations.
        
        :param model_name: The name of the model to use for vectorization.
        :param model_dir: The directory where model files are stored.
        """
        self.model_name = model_name
        self.model_dir = os.path.join(os.path.dirname(__file__), "models")
        self.sanitized_model_name = sanitize_filename(model_name)
        self._load_models()

    def _load_models(self):
        """Loads the models and scaler parameters required for processing."""
        print("Loading models...")
        self.model = SentenceTransformer(f'{self.model_name}')
        
        self.pca_model = joblib.load(f'{self.model_dir}/PCA_model - {self.sanitized_model_name}.pkl')
        scaler_params = joblib.load(f'{self.model_dir}/scaler_params - {self.sanitized_model_name}.pkl')
        
        self.scaler_model = StandardScaler()
        self.scaler_model.mean_ = scaler_params["mean_"]
        self.scaler_model.scale_ = scaler_params["scale_"]
        
        self.self_situational_model = joblib.load(f'{self.model_dir}/self_situational_ridge_model - {self.sanitized_model_name}.pkl')
        self.self_dispositional_model = joblib.load(f'{self.model_dir}/self_dispositional_ridge_model - {self.sanitized_model_name}.pkl')
        self.other_situational_model = joblib.load(f'{self.model_dir}/other_situational_ridge_model - {self.sanitized_model_name}.pkl')
        self.other_dispositional_model = joblib.load(f'{self.model_dir}/other_dispositional_ridge_model - {self.sanitized_model_name}.pkl')
        print("Models loaded successfully!")

    def process_csv(self, input_csv, row_id_col, text_col, output_dir="output", output_raw_embeddings=False):
        """
        Processes the input CSV and generates attribution classifications.

        :param input_csv: Path to the input CSV file.
        :param row_id_col: Column name for the row identifier.
        :param text_col: Column name for the text data.
        :param output_dir: Directory where the output files will be saved.
        :param output_raw_embeddings: Whether to output raw sentence embeddings.
        """

        # Get the current datetime and format it
        execution_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        vectorized_filename = f'{output_dir}/{execution_datetime} - vectorized_sentences.csv'
        predictions_filename = f'{output_dir}/{execution_datetime} - sentence_predictions.csv'

        print(f"\nProcessing input file: {input_csv}")
        with open(input_csv, 'r', encoding='utf-8') as infile, \
             open(vectorized_filename, 'w', encoding='utf-8', newline='') as vector_outfile, \
             open(predictions_filename, 'w', encoding='utf-8', newline='') as prediction_outfile:

            reader = csv.DictReader(infile)
            vector_writer = csv.writer(vector_outfile)
            prediction_writer = csv.writer(prediction_outfile)

            # Write headers for the output files
            vector_header = ['row_id', 'sentence'] + [f'vector_{i}' for i in range(768)]
            prediction_header = ['row_id', 'sentence',
                                 'self_situational_pred', 'self_dispositional_pred', 
                                 'other_situational_pred', 'other_dispositional_pred',
                                 'self_situational_prob', 'self_dispositional_prob', 
                                 'other_situational_prob', 'other_dispositional_prob']
            if output_raw_embeddings:
                vector_writer.writerow(vector_header)
            prediction_writer.writerow(prediction_header)

            # Process rows with progress bar
            for row in tqdm(reader, desc="Analyzing dataset"):
                row_id = row[row_id_col]
                text = row[text_col]

                # Tokenize text into sentences
                sentences = sent_tokenize(text)
                for sentence in sentences:
                    # Step 1: Vectorize the sentence
                    sentence_vector = self.model.encode(sentence)

                    # Step 2: Standardize and reduce dimensionality
                    standardized_vector = self.scaler_model.transform([sentence_vector])
                    reduced_vector = self.pca_model.transform(standardized_vector)

                    # Step 3: Apply ridge regression models
                    self_situational_prob = self.self_situational_model.predict_proba(reduced_vector)[0, 1]
                    self_dispositional_prob = self.self_dispositional_model.predict_proba(reduced_vector)[0, 1]
                    self_situational_pred = int(self_situational_prob >= 0.5)
                    self_dispositional_pred = int(self_dispositional_prob >= 0.5)

                    other_situational_prob = self.other_situational_model.predict_proba(reduced_vector)[0, 1]
                    other_dispositional_prob = self.other_dispositional_model.predict_proba(reduced_vector)[0, 1]
                    other_situational_pred = int(other_situational_prob >= 0.5)
                    other_dispositional_pred = int(other_dispositional_prob >= 0.5)

                    # Optionally write full vectors to the vectorized file
                    if output_raw_embeddings:
                        vector_writer.writerow([row_id, sentence] + sentence_vector.tolist())

                    # Write predictions to the predictions file
                    prediction_writer.writerow([row_id, sentence,
                                                self_situational_pred, self_dispositional_pred, 
                                                other_situational_pred, other_dispositional_pred,
                                                self_situational_prob, self_dispositional_prob, 
                                                other_situational_prob, other_dispositional_prob])

        print(f"\nProcessing complete! Results saved to:\n- {output_dir}")


def sanitize_filename(filename, max_length=255):
    """
    Sanitizes a string to make it safe for use as a filename.

    :param filename: The original filename string.
    :param max_length: The maximum allowed length for the filename (default: 255).
    :return: A sanitized string safe for use as a filename.
    """
    # Replace invalid characters with underscores
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '_', filename)
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip(' .')
    # Handle reserved filenames (Windows)
    reserved_names = {
        "CON", "PRN", "AUX", "NUL",
        "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
        "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"
    }
    if sanitized.upper() in reserved_names:
        sanitized = f"_{sanitized}_"
    # Truncate to the maximum length
    if len(sanitized) > max_length:
        ext_index = sanitized.rfind('.')
        if ext_index == -1 or ext_index > max_length - 5:
            sanitized = sanitized[:max_length]
        else:
            # Ensure extension remains intact if present
            name_part = sanitized[:max_length - (len(sanitized) - ext_index)]
            sanitized = f"{name_part}{sanitized[ext_index:]}"
    return sanitized
