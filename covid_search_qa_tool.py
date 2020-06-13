import json
import pandas as pd
import numpy as np
import nltk
import re
import requests
import string
from text_search_qa_tool import TextSearchQATool

class CovidText:
    def __init__(self, title, abstract, s2_id):
        self.title = title
        self.abstract = abstract
        self.s2_id = s2_id

    def text(self):
        output = self.title
        if output[-1] != ".":
            output += "."
        output += " "
        if type(self.abstract) == str:
            output += self.abstract
        return(output)


class CovidSearchQATool(TextSearchQATool):


    def __init__(self, covid_meta, qa_model_dir, only_covid=True):
        texts_dict = self._convert_to_texts(covid_meta, only_covid)
        super().__init__(texts=texts_dict,
                         qa_model_dir=qa_model_dir)

    def _convert_to_texts(self, meta, only_covid):
        """~
        Removing duplicates and entries with no abstract from metadata
        Returns:
            DataFrame (with useless entries removed)
        """
        # remove papers with placeholders for abstract
        meta.dropna(subset=["title"], inplace=True)

        # remove duplicate papers based on title and authors
        # function to remove caps and any punctuation

        if only_covid:
        # remove papers without covid-19 related keywords in title or abstract
            covid_regexs = [
                '2019[-\\s‐]?n[-\\s‐]?cov',
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
            meta = meta[covid_in_abstract | covid_in_title]

        meta.drop_duplicates(subset=["title"], inplace=True)
        texts_dict = {}
        for entry in meta.itertuples():
            texts_dict[entry.cord_uid] = CovidText(title=entry.title, 
                                                   abstract=entry.abstract,
                                                   s2_id=entry.s2_id)

        return texts_dict

    def search(self, containing=[], not_containing=[], search_name=None,
               containing_threshold=2):
        # find an unused search name if none entered
        if search_name is None:
            search_name = "search0"
            search_name_idx = 0
            while search_name in self.search_results.keys():
                search_name_idx += 1
                search_name = f"search{search_name_idx}"

        super().search(containing=containing,
                       not_containing=not_containing,
                       search_name=search_name,
                       containing_threshold=containing_threshold)

        missing_abstracts = 0
        for text_id in self.search_results[search_name].ids:
            if pd.isna(self.texts[text_id].abstract):
                missing_abstracts += 1
        
        print(f"{missing_abstracts} results do not have an abstract")

        

    def return_html_search_results(self, search_name, n_results=100):
        html_results = ""
        for idx, text_id in enumerate(self.search_results[search_name].ids):
            if idx == n_results:
                break
            text = self.texts[text_id]
            html_results += ("<h2><strong>" 
                             + text_id + " " + text.title 
                             + "</h2></strong>")
            if type(text.abstract) == str:
                html_results += "<p>" + text.abstract + "<p>"
        return html_results


    def return_html_answers(self, search_name, question, 
                            highlight_score=-2, top_n=100, max_length=128):

        answer_tuples = self.return_answers(search_name=search_name,
                                            question=question,
                                            max_length=max_length)
        
        answer_df = pd.DataFrame(answer_tuples, columns=["cord_uid",
                                                         "sentence_no",
                                                         "sentence",
                                                         "score"])

        title_mask = answer_df.sentence_no == 0

        title_scores = answer_df[title_mask][["cord_uid", "score"]]
        title_scores["score"] = title_scores.score * 2

        abstract_scores = answer_df[~title_mask][["cord_uid", "score"]]
        abstract_scores = abstract_scores.groupby("cord_uid").max()

        paper_scores = abstract_scores.merge(title_scores, on="cord_uid", how="left")
        paper_scores["score"] = np.sum(paper_scores.iloc[:,1:], axis=1)
        paper_scores.sort_values("score", ascending=False, inplace=True)

        html_output = f"<h1>Question: {question}</h1>"
        for idx, cord_uid in enumerate(paper_scores.cord_uid.values):
            if idx > top_n:
                break
            paper_mask = answer_df.cord_uid == cord_uid
            paper = answer_df[paper_mask].sort_values("sentence_no")
            title = paper.iloc[0]
            if title.score > highlight_score:
                html_output += ("<h3 style='color:yellow'>" 
                                + title.sentence.title() 
                                + " - " 
                                + title.cord_uid + "</h3>")
            else:
                html_output += ("<h3>" 
                                + title.sentence.title() 
                                + " - " 
                                + title.cord_uid 
                                + "</h3>")
            html_output += "<p>"
            for entry in paper.iloc[1:].itertuples():
                if entry.score > highlight_score:
                    html_output += ("<strong style='color:yellow'>" 
                                    + entry.sentence.capitalize() 
                                    + " .</strong>")
                else:
                    html_output += entry.sentence.capitalize() + ". "
            html_output += "</p><br>"
        
        return html_output


    def _split_text_to_sentences(self, text_ids, texts):

        # download nltk library to extract sentences from paragraphs
        nltk.download('punkt')
        sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        sentence_tuples = []
        for text_id, text in zip(text_ids, texts):
            sentences = sentence_tokenizer.tokenize(text.text())
            sentences = [sentence.lower() for sentence in sentences]
            for sentence_no, sentence in enumerate(sentences):
                sentence_tuples.append((text_id, sentence_no, sentence))
    
        return sentence_tuples


    def _search_by_texts_ids(self, search_texts_ids, containing, not_containing):
        output_ids = search_texts_ids
        if containing:
            containing_p = re.compile("|".join(containing))
            output_ids = (
                [text_id for text_id in output_ids
                 if containing_p.search(self.texts[text_id].text().lower())]
            )
        if not_containing:
            not_containing_p = re.compile("|".join(not_containing))
            output_ids = (
                [text_id for text_id in output_ids
                 if not not_containing_p.search(self.texts[text_id].text().lower())]
            )

        return output_ids

    
    def find_missing_abstracts(self, search_name):
        s2_api_address = "https://api.semanticscholar.org/v1/paper/CorpusID:{s2_id}"
        added_abstracts = 0
        print('=' * 100)
        print(f"Finding abstracts for search {search_name}")
        for text_id in self.search_results[search_name].ids:
            text = self.texts[text_id]
            if pd.isna(text.abstract) and not pd.isna(text.s2_id):
                r = requests.get(s2_api_address.format(s2_id=text.s2_id))
                s2_json = json.loads(r.text)
                if "abstract" in s2_json.keys():
                    if s2_json["abstract"] is not None:
                        print(f"Title: {text.title}")
                        print(f"Abstract: {s2_json['abstract']}\n")
                        text.abstract = s2_json["abstract"]
                        added_abstracts += 1
        print(f"Found and added {added_abstracts} missing abstracts")
