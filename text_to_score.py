# Import necessary libraries
import argparse  # For command-line argument parsing
import csv  # For working with CSV files
import json  # For handling JSON data

from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification  # For using Hugging Face Transformers

# Define the biomedical NER model
MODEL = "d4data/biomedical-ner-all"

# Initialize tokenizer and model for token classification
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForTokenClassification.from_pretrained(MODEL)

# Set up a pipeline for named entity recognition (NER)
pipe = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Define the main processing function
def process(*args):
    # Set up argument parser for command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--notes', help='Notes CSV', required=True)  # Input CSV file argument
    parser.add_argument('--out', help='Output', required=True)  # Output JSON file argument
    args = parser.parse_args()

    # Extract filepaths from command-line arguments
    filepath = args.notes
    outpath = args.out

    # Check if the input file is a CSV file
    if not filepath.endswith(".csv"):
        raise ValueError("Filepath must be a .csv file.")
    
    # Check if the output file is a JSON file
    if not outpath.endswith(".json"):
        raise ValueError("Output path must be a .json file.")
    
    processed = []  # List to store processed data
    with open(filepath, "r") as f:
        # Read CSV file as a dictionary
        reader = csv.DictReader(f)
        for row in reader:
            text = row["text"]  # Extract text from the row
            raw = pipe(text)  # Run NER on the text using the pipeline
            print(row)
            # Process and structure NER results along with additional information
            ner_content = {
                # "text": text,
                "obs_id": row["obs_id"],
                "entities": [
                    {
                        "entity": x["entity_group"],
                        "word": x["word"],
                        "score": round(float(x["score"]), 2),
                        "start": x["start"],
                        "end": x["end"],
                    }
                    for x in raw
                ],
            }
            processed.append(ner_content)  # Append processed data to the list
            
    # Write processed data as JSON to the specified output file
    with open(outpath, "w") as f:
        json.dump(processed, f)


# Execute the script if it is run as the main module
if __name__ == "__main__":
    import sys

    # Pass command-line arguments to the processing function
    process(*sys.argv[1:])


# example command : python process.py --notes input_data.csv --out output_data.json