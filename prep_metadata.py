import re
import os
import json
import numpy as np
import pandas as pd
import sys

def add_missing_abstracts(data_dir):

    meta = pd.read_csv(data_dir + "metadata.csv")
    meta = meta.dropna(subset=["title"])

    covid_regexs = [ '2019[-\\s‐]?n[-\\s‐]?cov',
                    "novel coronavirus.*2019", 
                    "2019.*novel coronavirus", 
                    "novel coronavirus pneumonia",
                    'coronavirus 2(?:019)?',
                    'coronavirus disease (?:20)?19',
                    'covid(?:[-\\s‐]?(?:20)?19)?',
                    'n\\s?cov[-\\s‐]?2019', 
                    'sars[-\\s‐]cov[-‐]?2',
                    'wuhan (?:coronavirus|cov|pneumonia)'
                ]
    covid_pattern = re.compile('|'.join(covid_regexs))

    def is_covid_related(text):
        if type(text) == str:
            if covid_pattern.search(text.lower()):
                return True
        return False

    covid_in_abstract = meta.abstract.apply(is_covid_related)
    covid_in_title = meta.title.apply(is_covid_related)
    covid_related = covid_in_abstract | covid_in_title
    no_abstract = meta.abstract.isna()
    has_json_files = ~meta.pdf_json_files.isna() | ~meta.pmc_json_files.isna()

    missing_start = sum(covid_related & no_abstract & has_json_files)    

    print(f"{sum(covid_related)} entries related to covid-19")
    print(f"{sum(covid_related & no_abstract)} without an abstract")
    print(f"{missing_start} of these have json text data")
    print(f"Creating abstracts from json text data for these entries...")


    def extract_text_from_json(filename):
        fp = open(data_dir + filename, 'r')
        entry_json = json.load(fp)
                    
        text = ""
        text_idx = 0
        while len(text) < 1500 and text_idx < len(entry_json["body_text"]):
            text += entry_json["body_text"][text_idx]["text"]
            text_idx += 1
        return text

    for entry in meta[covid_related & no_abstract & has_json_files].itertuples():
        if not pd.isna(entry.pmc_json_files):
            text = extract_text_from_json(entry.pmc_json_files)
            if len(text) > 1500:
                meta.at[entry.Index, "abstract"] = text
                
    no_abstract = meta.abstract.isna()

    for entry in meta[covid_related & no_abstract & has_json_files].itertuples():
        if not pd.isna(entry.pdf_json_files):
            if ';' in entry.pdf_json_files:
                entry_texts = []
                for file in entry.pdf_json_files.split(';'):
                    text = extract_text_from_json(file.strip())
                    entry_texts.append(text)
                longest_text = entry_texts[np.argmax(entry_texts)]
                if len(longest_text) > 700:
                    meta.at[entry.Index, "abstract"] = longest_text
            else:
                text = extract_text_from_json(entry.pdf_json_files)
                if len(text) > 700:
                    meta.at[entry.Index, "abstract"] = text
    
    no_abstract = meta.abstract.isna()
    missing_end = sum(covid_related & no_abstract & has_json_files)    
    print(f"{missing_start - missing_end} abstracts added from json text")

    return meta
                    
if __name__ == "__main__":

    if not len(sys.argv) == 2:
        raise ValueError("Must specify directory containing covid data i.e. py prep_meta.py data/")

    data_dir = sys.argv[1]
    if not data_dir[-1] == "/":
        data_dir += "/"

    elif not os.path.exists(data_dir):
        raise TypeError(f"Enter a valid directory. None found at {sys.argv[1]}")

    meta = add_missing_abstracts(data_dir)
    meta.to_csv(data_dir + "metadata_processed.csv", index=False)
