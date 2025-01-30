import os

import re

import joblib

import pandas as pd

from collections import Counter

from PyPDF2 import PdfReader



# Load the trained model

MODEL_PATH = "logistic_regression_control_id_model.pkl"



try:

    model = joblib.load(MODEL_PATH)

except Exception as e:

    print(f"Error loading model: {e}")

    exit(1)



def extract_text_from_pdf(pdf_path):

    """Extract text from a PDF file."""

    try:

        reader = PdfReader(pdf_path)

        text_data = []

        for page in reader.pages:

            text = page.extract_text()

            if text:

                text_data.extend(text.split())  # Split by spaces

        return text_data

    except Exception as e:

        print(f"Error reading PDF: {e}")

        return []



def extract_regex_patterns(phrases):
    """Dynamically extract regex patterns from phrases."""
    regex_patterns = {}
    
    for phrase in phrases:
        # Convert numbers to a regex pattern
        generalized_pattern = re.sub(r"\d+", r"\\d+", phrase)  # Properly escape digits
        generalized_pattern = re.sub(r"[A-Za-z]+", r"[A-Za-z]+", generalized_pattern)  # Keep brackets for letters
        
        regex_patterns[phrase] = generalized_pattern

    return regex_patterns




def classify_phrases(phrases):

    """Classify phrases and filter Control IDs based on regex frequency."""

    results = []

    control_id_candidates = []



    # Initial classification

    for phrase in phrases:

        prediction = "Control ID" if model.predict([phrase])[0] == 1 else "Not a Control ID"

        results.append({"Phrase": phrase, "Prediction": prediction})

        

        if prediction == "Control ID":

            control_id_candidates.append(phrase)



    # Extract regex patterns for control ID candidates

    regex_patterns = extract_regex_patterns(control_id_candidates)

    pattern_counts = Counter(regex_patterns.values())  # Count occurrences of each regex pattern



    # Filter out control IDs that have unique regex patterns (not repeated in the doc)

    filtered_results = []

    for result in results:

        phrase = result["Phrase"]

        if result["Prediction"] == "Control ID":

            matched_pattern = regex_patterns.get(phrase, None)

            if matched_pattern and pattern_counts[matched_pattern] >= 2:

                filtered_results.append({"Phrase": phrase, "Prediction": "Control ID"})

            else:

                filtered_results.append({"Phrase": phrase, "Prediction": "Not a Control ID (Unique Pattern)"})

        else:

            filtered_results.append(result)



    return pd.DataFrame(filtered_results)



def main():

    """Interactive Input/Output Interface."""

    print("\n🔍 Control ID Classifier with Regex Frequency Validation\n")

    

    # Get user input for file path

    while True:

        pdf_path = input("📂 Enter the full path to the PDF file (or type 'exit' to quit): ").strip()

        if pdf_path.lower() == "exit":

            print("Exiting program. Goodbye! 👋")

            break

        if not os.path.exists(pdf_path):

            print("❌ Error: File not found! Please enter a valid PDF path.")

            continue

        

        print(f"\n📖 Extracting text from: {pdf_path}...")

        phrases = extract_text_from_pdf(pdf_path)

        

        if not phrases:

            print("❌ No text found in PDF or decryption required. Try another file.")

            continue



        print("\n🤖 Classifying and validating phrases... Please wait.")

        classified_df = classify_phrases(phrases)



        # Preview some results

        print("\n🔹 Classification Preview:")

        print(classified_df.head(10))  # Show first 10 results

        

        # Ask user to save results

        save_choice = input("\n💾 Save results to Excel? (y/n): ").strip().lower()

        if save_choice == "y":

            output_file = os.path.join(os.path.dirname(pdf_path), "classified_control_ids_with_regex.xlsx")

            classified_df.to_excel(output_file, index=False)

            print(f"\n✅ Results saved to: {output_file}")

        else:

            print("\n❌ Results were NOT saved.")



        # Ask if user wants to process another file

        another_file = input("\n🔄 Process another PDF? (y/n): ").strip().lower()

        if another_file != "y":

            print("\n👋 Exiting program. Have a great day!")

            break



if __name__ == "__main__":

    main()