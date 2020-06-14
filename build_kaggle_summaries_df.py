import os
import pandas as pd
import string
from tqdm import tqdm
import re
import sys

SUMMARY_DIRS = [
    "1_population",
    "2_relevant_factors",
    "3_patient_descriptions",
    "4_models_and_open_questions",
    "5_materials",
    "6_diagnostics",
    "7_therapeutics_interventions_and_clinical_studies",
    "8_risk_factors",
    "unsorted_tables/key_scientific_questions",
    "unsorted_tables/risk_factors"
]


STUDY_TYPE_NAMES = {
    "systematic review and metaanalysis": ["systematic review and metaanalysis",
                                           "systematic review", 
                                           "systemic review", 
                                           "systematic review  metaanalysis", 
                                           "systemic review and metaanalysis", 
                                           "systematic literature review", 
                                           "systematic reviews"],
    "prospective observational study": ["prospective observational study",
                                        "prospective cohort", 
                                        "prospective cohort study"],
    "retrospective observational study": ["retrospective observational study",
                                          "retrospective cohort", 
                                          "retrospective study", 
                                          "retrospective observational", 
                                          "retrospective analysis", 
                                          "retrospective cohort study", 
                                          "retrospective review", 
                                          "retrospective observational review"],
    "crosssectional study": ["crosssectional study",
                            "cross sectional study", 
                            "crosssectional"],
    "case series": ["case study", 
                    "descriptive case series", 
                    "case report", 
                    "caseseries"],
    "expert review": ["expert review", 
                    "review", 
                    "literature review"],
    "editorial": ["editorial"],
    "ecological regression": ["ecological regression",
                            "ecological study"],
    "simulation":["simulation",
                "modeling study", 
                "modeling", 
                "simulation study", 
                "simlation study",
                "metaanalysis and simulation"],
}

COVID_REGEXS = ["2019[-\\s‐]?n[-\\s‐]?cov",
                "novel coronavirus.*2019", 
                "2019.*novel coronavirus", 
                "novel coronavirus pneumonia",
                "coronavirus 2(?:019)?",
                "coronavirus disease (?:20)?19",
                "covid(?:[-\\s‐]?(?:20)?19)?",
                "n\\s?cov[-\\s‐]?2019", 
                "sars[-\\s‐]cov[-‐]?2",
                "wuhan (?:coronavirus|cov|pneumonia)"
]

def build_summaries_df(data_dir):

    if not data_dir[-1] == "/":
        data_dir += "/"


    def make_alpha(title):
        if type(title) == str:
            return title.translate(str.maketrans('','',string.punctuation)).lower().strip()
        else:
            title


    summaries = []
    for summary_dir in SUMMARY_DIRS:

        full_path = f"{data_dir}Kaggle/target_tables/{summary_dir}"
        for file in os.listdir(full_path):

            df = pd.read_csv(f"{full_path}/{file}")
            
            fixed_names = {}
            for colname in df.columns:
                fixed_name = "_".join(make_alpha(colname).split())
                fixed_names[colname] = fixed_name
            df = df.rename(columns=fixed_names)
            
            df["summary_table"] = file[:-4]
            if not summary_dir[0] == 'u':
                df["task"] = summary_dir
            else:
                df["task"] = summary_dir[16:]
            if len(summaries):
                summaries = summaries.append(df)
            else:
                summaries = df
                
    summaries.reset_index(inplace=True, drop=True)

    meta = pd.read_csv("data/metadata_processed.csv")

    covid_pattern = re.compile('|'.join(COVID_REGEXS))
    def is_covid_related(text):
        if type(text) == str:
            if covid_pattern.search(text.lower()):
                return True
        return False


    meta = pd.read_csv("data/metadata.csv")
    covid_in_abstract = meta.abstract.apply(is_covid_related)
    covid_in_title = meta.title.apply(is_covid_related)
    meta = meta[covid_in_abstract | covid_in_title]

    cord_uid_title = meta[["cord_uid", "title"]]
    cord_uid_title["title"] = cord_uid_title.title.apply(make_alpha)

    def find_paper_id(title):
        cord_uids = []
        stripped_title = make_alpha(title)
        for entry in cord_uid_title.itertuples():
            if pd.isna(entry.title):
                continue
            elif entry.title == stripped_title:
                cord_uids.append(entry.cord_uid)
        return cord_uids

    cord_summaries = []

    for idx in tqdm(summaries.index, total=len(summaries)):
        entry = summaries.loc[idx]
        cord_uids = find_paper_id(entry.study)
        for cord_uid in cord_uids:
            new_entry = entry.copy()
            new_entry["cord_uid"] = cord_uid
            cord_summaries.append(new_entry)
            
    cord_summaries = pd.DataFrame(cord_summaries)

    study_type_synonyms = {}

    for study_type_name in STUDY_TYPE_NAMES.keys():
        for synonym in STUDY_TYPE_NAMES[study_type_name]:
            study_type_synonyms[synonym] = study_type_name
            
    cord_summaries["study_type"] = cord_summaries.study_type.apply(make_alpha)
    cord_summaries["study_type"] = cord_summaries.study_type.apply(
        lambda x: study_type_synonyms[x] if x in study_type_synonyms.keys() else "other"
    )

    return cord_summaries

if __name__ == "__main__":

    if not len(sys.argv) == 2:
        raise ValueError("Must specify directory containing covid data i.e. py prep_meta.py data/")

    data_dir = sys.argv[1]
    if not data_dir[-1] == "/":
        data_dir += "/"

    elif not os.path.exists(data_dir):
        raise TypeError(f"Enter a valid directory. None found at {sys.argv[1]}")

    cord_summaries = build_summaries_df(data_dir)
    cord_summaries.to_csv(f"{data_dir}summaries_processed.csv", index=False)