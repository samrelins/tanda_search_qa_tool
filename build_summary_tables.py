from collections import Counter
from IPython.core.display import display, HTML
import json 
import numpy as np
import pandas as pd
import requests
import re
import string
import spacy
from IPython.core.display import display, HTML


def disp(html):
    display(HTML(html))


def get_json(filename):
    with open("data/" + filename, 'r') as fp:
        return json.load(fp) 


def make_alpha(text):
    chars = [char for char in text if char.isalpha() or char in [' ', '\n']]
    string =  "".join(chars).lower()
    return re.sub("\s{2,}|\n", " ", string)


def get_paper_text(cord_uid, meta):
    
    # find entry and collect json text files
    meta_entry = meta[meta.cord_uid == cord_uid].iloc[0]
    json_results = []
    if not pd.isna(meta_entry.pmc_json_files):
        json_results.append(get_json(meta_entry.pmc_json_files))
    if not pd.isna(meta_entry.pdf_json_files):
        if ';' in meta_entry.pdf_json_files:
            for json_file in meta_entry.pdf_json_files.split(';'):
                json_results.append(get_json(json_file.strip()))
        else:
            json_results.append(get_json(meta_entry.pdf_json_files))
            
    # loop through json text files and build paper text
    paper_text = ""
    section_headers = []
    for result in json_results:
        if len(result["body_text"]) <= 1:
            continue
        for body_text in result["body_text"]:

            alpha_paper_text = make_alpha(paper_text)
            alpha_result_text = make_alpha(body_text["text"])
            already_in = False
            for i in range(3):
                if alpha_result_text[i*20+20:i*20+120] in alpha_paper_text:
                    already_in = True
                    break

            if not already_in:
                if not body_text["section"].lower() in section_headers:
                    section_headers.append(body_text['section'].lower())
                    paper_text += f"{body_text['section'].lower()}\n\n"
                paper_text += f"{body_text['text'].lower()}\n\n"

    if len(paper_text) < 2500:
        return ""

    if not pd.isna(meta_entry.abstract) and paper_text:
        alpha_paper_text = make_alpha(paper_text)
        alpha_abstract = make_alpha(meta_entry.abstract)
        already_in = False
        for i in range(3):
            if alpha_abstract[i*20+20:i*20+120] in alpha_paper_text:
                already_in = True
                break
        
        if not already_in:
            paper_text = f"{meta_entry.abstract.lower()}\n\n{paper_text}"
        
    return paper_text


nlp = spacy.load("en_core_web_sm")
accepted_tokens = ["amod", "compound", "npadvmod", "prep", "det", "compound", "punct", "nummod", "nsubj", "pobj"]

def extract_text_features(paper_text):

    doc = nlp(paper_text)

    results = { 
        "quantities": [],
        "cardinals": [],
        "percents": [],
        "gpes": [],
    }
    
    for idx, ent in enumerate(doc.ents):
        
        is_counter = False
        if ent.label_ is "CARDINAL" and ent.text.isnumeric():
            try: 
                int_n = int(ent.text)
            except:
                continue
            next_30 = [ent.text for ent in doc.ents[idx+1:idx+21]]
            next_10 = [ent.text for ent in doc.ents[idx+1:idx+6]]
            if str(int_n + 1) in next_10:
                is_counter = True
            elif str(int_n + 1) in next_30 and str(int_n + 2) in next_30:
                is_counter = True
                
        if is_counter:
            continue
        
        if re.sub(",|\.", "", ent.text).isnumeric() or ent.label_ is "CARDINAL":
            
            if ent.text[:2] in ["19", "20"] and ent.label_ is "DATE":
                results["cardinals"].append(ent.text)
                continue
            elif paper_text[ent.start_char - 1] in ['£','$','€']:
                results["quantities"].append((ent.text, paper_text[ent.start_char - 1]))
                continue
            
            quantity_of = None
            for i in range(10):
                if len(doc) <= ent.end + i:
                    break
                token = doc[ent.end + i]
                if token.tag_ == "NNS" and not token.dep_ == "compound":
                    quantity_of = token.text
                    break
                elif token.dep_ in accepted_tokens and not token.text in ['.', "in"]:
                    continue
                else:
                    break
                    
            if quantity_of is None:
        
                tokens_to_check = 0
                of_found = False
                while tokens_to_check <= 20:
                    token = doc[ent.start - tokens_to_check - 1]
                    if token.text == "of":
                        of_found=True
                        break
                    elif token.text in ['.', '(', '[', '{']:
                        break
                    else:
                        tokens_to_check += 1

                if of_found:
                    for i in range(tokens_to_check):
                        if len(doc) <= ent.start + i:
                            break
                        token = doc[ent.start - tokens_to_check + i]
                        if token.tag_ == "NNS" and not token.dep_ == "compound":
                            quantity_of = token.text
                            break
                        elif token.dep_ in accepted_tokens and not token.text in ['.', "in"]:
                            continue
                        else:
                            break
                
            if not quantity_of is None:
                results["quantities"].append((ent.text, quantity_of))  
            else:
                results["cardinals"].append(ent.text)

        if ent.label_ is "PERCENT":
            results["percents"].append(ent.text)

        if ent.label_ is "GPE" and ent.text.isalpha():
            results["gpes"].append(ent.text)
    
    plural_nouns = []
    for token in doc:
        if token.tag_ == "NNS" or token.ent_type_ in ["NORP", "GPE"]:
            plural_nouns.append((token.text))
            
    results["plural_nouns"] = Counter(plural_nouns).most_common()

    return results


study_design_keywords = {
    "meta\W?analysis": 'systematic review and metaanalysis',
    "systematic\W(literature)?review": 'systematic review and metaanalysis',
    "literature review": 'systematic review and metaanalysis',
    "scoping\Wreview": 'systematic review and metaanalysis',
    "case\Wseries": 'case series',
    "cross\Wsectional": "crosssectional study",
    "retrospective\W.{,20}cohort": 'retrospective observational study',
    "longitudinal\W.{,20}cohort": 'retrospective observational study',
    "retrospective\W.{,20}stud[iy]": 'retrospective observational study',
    "retrospective\W.{,20}review": 'retrospective observational study',
    "retrospective\W.{,20}analysis": 'retrospective observational study',
    "prospective\W.{,20}cohort": 'prospective observational study',
    "prospective\W.{,20}stud[iy]": 'prospective observational study',
    "prospective\W.{,20}review": 'prospective observational study',
    "prospective\W.{,20}analysis": 'prospective observational study',
    "simulat(e|ion)": 'simulation',
    "modell?ing": 'simulation',
    "model\Wbased": 'simulation',
    "mathematical model": "simulation",
    "machine\Wlearning": "simulation",
}

study_design_regexs = [
    "this (?!.{{,100}} a )(?!.{{,100}} existing )(?!.{{,100}} other ).{{,100}}{study_design}",
    "our (?!.{{,100}} a )(?!.{{,100}} existing )(?!.{{,100}} other ).{{,100}}{study_design}",
    "the present .{{,100}}{study_design}",
    "we( have)? conduct(ed)? .{{,200}}{study_design}",
    "we( have)? perform(ed)? .{{,200}}{study_design}",
    "we( have)? implement(ed)? .{{,200}}{study_design}",
    "we( have)? appl(y|ied) .{{,200}}{study_design}",
    "we( have)? use[\sd] .{{,200}}{study_design}",
    "{study_design}.{{,200}} was conducted",
    "{study_design}.{{,200}} was performed",
    "{study_design}.{{,200}} was implemented",
    "{study_design}.{{,200}} was applied",
    "{study_design}.{{,200}} was used",
    "we present .{{,100}}{study_design}",
    "we estimated .{{,200}}{study_design}",
    "we confirmed (?!.{{,200}} a )(?!.{{,100}} existing )(?!.{{,100}} other ).{{,100}}{study_design}",
    "we tested (?!.{{,200}} a )(?!.{{,100}} existing )(?!.{{,100}} other ).{{,100}}{study_design}",
    "based on .{{,100}}{study_design}",
    "design\W .{{,100}}{study_design}",
    "method[s\W].{{,100}}{study_design}"
]

design_keyword_lists = {}
for keyword, study_design in study_design_keywords.items():
    if study_design in design_keyword_lists.keys():
        design_keyword_lists[study_design].append(keyword)
    else:
        design_keyword_lists[study_design] = [keyword]

        
def count_study_design_mentions(paper_text):
    results = []
    for study_design, keyword_list in design_keyword_lists.items():
        total_hits = 0
        for keyword in keyword_list:
            keyword_re = [design_re.format(study_design=keyword)
                          for design_re in study_design_regexs]
            keyword_p = re.compile("|".join(keyword_re))
            total_hits += len([hit for hit in keyword_p.finditer(paper_text)])
        results.append(total_hits)
    return tuple(results)


stats_regexs = [
    "\Wci\W", "p\s?<", "n\s?=", "p\s?=", "\Wiqr\W", "interquartile",
    "standard deviation", "\Wsd\W", "median", "p\Wvalue", "pearson", 
    "±", "≥", "≦", "≤", "χ"
]


def count_statistical_features(text):
    stats_p = re.compile("|".join(stats_regexs))
    return len([hit for hit in stats_p.finditer(text.lower())])


title_keywords = {
    "meta\W?analysis": 'systematic review and metaanalysis',
    "systematic\W(literature)?review": 'systematic review and metaanalysis',
    "literature review": 'systematic review and metaanalysis',
    "scoping\Wreview": 'systematic review and metaanalysis',
    "simulat(e|ion)": 'simulation',
    "modell?ing": 'simulation',
    "model\Wbased": 'simulation',
    "mathematical model": "simulation",
    "machine\Wlearning": "simulation",
    "case\Wseries": 'case series',
    "cross\Wsectional": "crosssectional study",
    "retrospective cohort": 'retrospective observational study',
    "retrospective .{,20}stud[iy]": 'retrospective observational study',
    "regression": "ecological regression",
    "least squares": "ecological regression",
}


def assign_type_from_title(title):
    for keyword, study_design in title_keywords.items():
        if re.search(keyword, title.lower()):
            return study_design
    return None

design_types = np.array([key for key in design_keyword_lists.keys()])

model_terms = ["predict", "calculat", "fit", "apply", "param", "evaluat", "analy"]

model_regexs = []
for term in model_terms:
    model_regexs.append(f"{term}.{{,200}}model|model.{{,200}}{term}")

model_p = re.compile("|".join(model_regexs))


def assign_type_from_text(paper_text):

    text_types = np.array(find_study_designs_in_text(paper_text))
    
    if (text_types > 0).sum() == 1:
        return design_types[text_types > 0][0]
    elif text_types[np.where(design_types == "systematic review and metaanalysis")]:
        return "systematic review and metaanalysis"
    elif text_types[np.where(design_types == "prospective observational study")]:
        return "prospective observational study"
    elif text_types[np.where(design_types == "retrospective observational study")]:
        return "retrospective observational study"
    elif (text_types > 2).sum():
        return design_types[np.array(text_types).argmax()]
    elif (text_types > 0).sum():
        return design_types[np.array(text_types).argmax()]
                          
    modelling_mentions = len([match for match in model_p.finditer(paper_text)])
    if modelling_mentions > 5:
        return "simulation"
        
    stats_mentions = has_statistical_analysis(paper_text)
    follow_up = True if re.search("follow(ing|ed)?\Wup", paper_text) else False
    
    method_p = re.compile("\n\n.{,100}method.{,100}\n\n")
    methods_section = True if method_p.search(paper_text) else False

    results_p = re.compile("\n\n.{,100}result.{,100}\n\n")
    results_section = True if results_p.search(paper_text) else False
                          
    if (stats_mentions and follow_up) or (follow_up and results_section and methods_section):
        return "retrospective observational study"
    
    if stats_mentions or (methods_section and results_section):
        return "analytical study"
                          
    return "expert review"
    

def get_evidence_measures(text):
    
    features = extract_text_features(text)
    
    all_quantity_nouns = [noun for _, noun in features["quantities"]]
    most_common_nouns = Counter(all_quantity_nouns).most_common()
    mul_ref_nouns = [noun for noun, references in most_common_nouns
                     if references > 2]
    quantities_dict = {}
    for amount, noun in features["quantities"]:
        if not amount.isnumeric() or not noun in mul_ref_nouns:
            continue
        if not noun in quantities_dict.keys():
            if float(amount) > 9:
                quantities_dict[noun] = amount
        elif amount > quantities_dict[noun]:
            quantities_dict[noun] = amount
    
    accepted_nouns = [noun for noun, mentions in features["plural_nouns"][:10]
                      if mentions > 5]
    
    results = ""
    for noun in quantities_dict.keys():
        if noun in accepted_nouns:
            results += f"{noun}: {quantities_dict[noun]}, "
    
    locations = []
    for gpe, mentions in Counter(features["gpes"]).most_common():
        locations.append(gpe.title())
    if locations:
        results += f"Locations: {', '.join(locations)}"
                       
    return results
    
