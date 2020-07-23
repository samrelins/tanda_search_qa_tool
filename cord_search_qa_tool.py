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


class CordSearchQATool(TextSearchQATool):

    """
    Tool to find covid papers that answer a specific research question.
    
    The search process is conducted in two stages: 
    
        1) Identify broad a subset of the corpus containing keywords 
        relating to the research question - the aim should be to include
        all possible papers of interest, whilst eliminating papers that
        bear no relation 
        
        See methods: search, refine_search, return_html_search_results
        and the search_results attribute
        
        2) Search over the abstracts for every paper identified in stage 1 
        for answers to a specific research question. A Roberta-Tanda model 
        generates scores for each sentence which are used to rank the most 
        relevant papers. 
        
        See methods: return_answers, return_html answers 
    
    Functionality and methods inherited from the base TextSearchQATool
    class.
    
    Parameters
    __________
    
        covid_meta: pandas.DataFrame 
            metadata.csv file from the CORD-19 dataset loaded in a pandas 
            DataFrame. Note: some preprocessing will be required if duplicate 
            cord_uids are present.
        qa_model_dir: str  
            Location of the tanda model directory containing a pre-trained
            model checkpoint
        only_covid: bool, default True
            If True, identify papers that relate specifically to covid-19 and 
            remove any papers that dont. Otherwise keep all entries in metadata.csv
            
    Attributes
    ___________

        texts: dict  
            Dict of [cord_uid]:[abstract] pairs 
        search_results: dict
            dict of [search_name]:[SearchResult] pairs. Uses the SearchResult
            class defined in the text_search_qa_tool module 
        tokenizer: RobertaTokenizer
            Hugging face's Transformers implementation of the Bert tokenizer
        device: torch.device
            Device on which to mount model (CPU or GPU)
        model: RobertaForSequenceClassification
            Hugging Face's Transformers implementation of the Roberta model
            specialised for sequence classification
        
    Methods
    _______
    
        search
        refine_search
        return_html_search_results
        clear_searches
        return_answers
        return_html_answers
    """

    def __init__(self, covid_meta, qa_model_dir, only_covid=True):
        texts_dict = self._convert_to_texts(covid_meta, only_covid)
        super().__init__(texts=texts_dict,
                         qa_model_dir=qa_model_dir)

    def _convert_to_texts(self, meta, only_covid):
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
                             + text_id + " - " + text.title 
                             + "</h2></strong><br>")
            if type(text.abstract) == str:
                html_results += "<p>" + text.abstract + "<p>"
            html_results += "<br><br>"
        return html_results


    def return_html_answers(self, search_name, question, min_score=None, 
                            highlight_score=-2, top_n=10, max_length=128):

        answer_tuples = self.return_answers(search_name=search_name,
                                            question=question,
                                            min_score=None,
                                            max_length=max_length)
        
        answer_df = pd.DataFrame(answer_tuples, columns=["cord_uid",
                                                         "sentence_no",
                                                         "sentence",
                                                         "score"])

        high_scores = answer_df[["cord_uid", "score"]]
        high_scores = high_scores.groupby("cord_uid", as_index=False).max()
        high_scores.sort_values("score", ascending=False, inplace=True)

        html_output = f"<h1>Question: {question}</h1><br><br>"
        for idx, cord_uid in enumerate(high_scores.cord_uid.values):
            if idx > top_n:
                break
            paper_mask = answer_df.cord_uid == cord_uid
            paper = answer_df[paper_mask].sort_values("sentence_no")
            title = paper.iloc[0]
            if title.score > highlight_score:
                html_output += ("<h3 style='color:magenta'>" 
                                + title.sentence.title() 
                                + " - " 
                                + title.cord_uid + "</h3><br>")
            else:
                html_output += ("<h3>" 
                                + title.sentence.title() 
                                + " - " 
                                + title.cord_uid 
                                + "</h3>")
            html_output += "<p>"
            for entry in paper.iloc[1:].itertuples():
                if entry.score > highlight_score:
                    html_output += ("<strong style='color:magenta'>" 
                                    + entry.sentence.capitalize() 
                                    + " .</strong>")
                else:
                    html_output += entry.sentence.capitalize() + ". "
            html_output += "</p><br><br>"
        
        if min_score is None:
            return answer_tuples, html_output

        output_answer_tuples = []
        for answer_tuple in answer_tuples:
            if answer_tuple[3] > min_score:
                output_answer_tuples.append(answer_tuple)

        return output_answer_tuples, html_output


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

    
    def _search_by_texts_ids(self, search_texts_ids, containing, not_containing, 
                             containing_threshold):

        output_ids = search_texts_ids
        if containing:
            output_ids = []
            containing_p = re.compile("|".join(containing))
            for text_id in search_texts_ids:
                text = self.texts[text_id].text().lower()
                n_hits = len([hit for hit in containing_p.finditer(text)])
                if n_hits > containing_threshold:
                    output_ids.append(text_id)
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
