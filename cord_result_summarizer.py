import nltk
import numpy as np
import os
import pandas as pd
import re
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm
from transformers.data.processors.utils import InputExample, InputFeatures
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from summarizer_helpers import *


class CordResultSummarizer:
    
    """
    Create summary tables of a specified list of papers.
    
    Summary tables are generated in the format specified in round 2 task 1 of 
    the Kaggle CORD-19 competition: Population studies.  
    
    Summary table features are: 
    
        cord_uid: 
            unique paper id 
        date: 
            date of publication 
        study: 
            name of the paper / study 
        study_type: 
            study design e.g. "simulation", "retrospective_cohort" 
        challenge: 
            The challenge the paper identifies / explores 
        solution: 
            The proposed solution to the paper's challenge 
        strength_of_evidence: 
            an indication of the scope of the study, in terms
            of the size of the study population e.g. "200 survey participants"
        addressed_population: 
            the group studied i.e. "mental health patients", 
        study_link: 
            link to the study online, 
        journal: 
            journal in which the study was published
    
    Parameters
    ----------
    
        cord_uids: list 
            List of cord_uids identifying papers to be summarized
        meta: pandas.DataFrame 
            Metadata dataframe from CORD-19 dataset
        data_dir: str 
            Location of directory containing CORD-19 dataset
        tanda_dir: str 
            Location of directory containing tanda model checkpoint 
    
    Attributes
    ----------

        data_dir: str 
            Location of directory containing CORD-19 dataset
        abstracts: dict 
            Dict containing [cord_uid]:[abstract text] key-value pairs 
        full_texts: dict
            Dict containing [cord_uid]:[body text] key-value pairs 
        tokenizer: RobertaTokenizer
            Hugging face's Transformers implementation of the Bert tokenizer
        device: torch.device
            Device on which to mount model (CPU or GPU)
        model: RobertaForSequenceClassification
            Hugging Face's Transformers implementation of the Roberta model
            specialised for sequence classification
        
    Methods
    -------
    
        summary_table
        
    """

    
    def __init__(self, cord_uids, meta, data_dir, tanda_dir):
        
        print('=' * 100)
        print("Building result summarizer")
        self.meta = meta
        
        # check data dir exists - assign to attribute if it does
        if not data_dir[-1] == '/':
            data_dir += '/'
        if not os.path.exists(data_dir):
            raise ValueError(f"{data_dir} doesn't exist")
            
        self.data_dir = data_dir
        self._init_text_atts(cord_uids, meta, data_dir)
        self._init_qa_model_atts(tanda_dir)

        
    def _init_text_atts(self, cord_uids, meta, data_dir):
        # function to collect and abstracts and texts for each paper
        
        full_texts_dict = {}
        for cord_uid in cord_uids:
            paper_text = get_paper_text(cord_uid, meta, data_dir)
            if paper_text:
                full_texts_dict[cord_uid] = paper_text

        abstracts_dict = {}
        for cord_uid in full_texts_dict.keys():
            abstracts_dict[cord_uid] = (meta[meta.cord_uid == cord_uid]
                                        .iloc[0].abstract) 

        self.abstracts, self.full_texts = abstracts_dict, full_texts_dict


    def _init_qa_model_atts(self, tanda_dir):
        # function to initialise objects for tanda model
        
        # check if tanda directory exists
        tanda_dir += "/models/tanda_roberta_base_asnq/ckpt"
        if not os.path.exists(tanda_dir):
            raise TypeError("TandA directory doesn't exist or "
                            f"path to ckpt doesn't match {tanda_dir}")
            
        # Load pretrained tokenizer
        tokenizer_class = RobertaTokenizer
        self.tokenizer = tokenizer_class.from_pretrained(tanda_dir,
                                                do_lower_case=True,
                                                cache_dir=None)
        
        # collect GPU variables to mount tensors
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # define model and set to eval mode
        print("Initialising QA Model")
        model_class = RobertaForSequenceClassification
        self.model = model_class.from_pretrained(tanda_dir)
        self.model.to(self.device)
        self.model.eval()
        print("QA Model Loaded. Ready to build summary tables")
        print('=' * 100)


    def summary_table(self, n_hits=2, 
                      challenge_question="what is the problem issue challenge", 
                      solution_question="what action should be taken"):
        
        """
        Generate a summary of the papers specified in cord_uids
        
        Gathers together all of the functions that identify individual features
        of the summary table, and bulds a dataframe from the results.
        
        Parameters
        ----------
        
            n_hits: int, default 2
                The number of mentions of a population in the paper text 
                required to be included in the addressed_population feature
            challenge_question: str, default "what is the problem issue challenge"
                The question given to the tanda model when identifying the
                challenge feature - search is conducted over the text in the
                abstract only.
            solution_question: str, default "what action should be taken"
                The question given to the tanda model when identifying the
                solution feature - search is conducted over the full paper text
                
        Returns
        -------
        
            pandas.DataFrame containing the summary data
        
        """
        
        nlp = spacy.load("en_core_web_sm")

        print("\nBuilding summary table from "
              f"{len(self.abstracts.keys())} papers", flush=True)
        summary_table = {
            "cord_uid": [], "date": [], "study": [], "study_type": [],
            "challenge": [], "solution": [], "strength_of_evidence": [],
            "addressed_population": [], "study_link": [], "journal": [],
        }
    
        print(f"Finding challenges from paper text", flush=True)
        challenge_answers = self._return_answers(
            question=challenge_question,
            texts_dict=self.abstracts
        )
        print(f"Finding solutions from paper text", flush=True)
        solution_answers = self._return_answers(
            question=solution_question,
            texts_dict=self.full_texts
        )


        print(f"\nBuilding table entries", flush=True)
        for cord_uid in tqdm(self.full_texts.keys()):
            
            paper_text = self.full_texts[cord_uid]
            meta_entry = self.meta[self.meta.cord_uid == cord_uid].iloc[0]
            doc = nlp(paper_text)
        
            summary_table["cord_uid"].append(cord_uid)
            summary_table["date"].append(meta_entry.doi)
            summary_table["study"].append(meta_entry.title)
            summary_table["study_link"].append(meta_entry.url) 
            summary_table["journal"].append(meta_entry.journal)
 
            type_from_title = assign_type_from_title(meta_entry.title)
            if not type_from_title is None:
                summary_table["study_type"].append(type_from_title)
            else:
                type_from_text = assign_type_from_text(paper_text)
                summary_table["study_type"].append(type_from_text)
        
            summary_table["strength_of_evidence"].append(
                get_strength_of_evidence(paper_text, doc)
            )

            addressed_population = find_populations(paper_text, doc, n_hits)
            if not addressed_population:
                addressed_population = find_populations_backup(paper_text, 
                                                               n_hits=5)
            summary_table["addressed_population"].append(addressed_population)

            paper_challenge_answers = [
                (answer, score) 
                for answer_uid, _, answer, score in challenge_answers
                if answer_uid == cord_uid
            ]
            challenge = ""
            challenge_score = -100
            for answer, score in paper_challenge_answers:
                if score > challenge_score:
                    challenge = answer
                    challenge_score = score
            summary_table["challenge"].append(challenge)
        
            paper_solution_answers = [
                (answer, score) 
                for answer_uid, _, answer, score in solution_answers
                if answer_uid == cord_uid
            ]
            solution = ""
            solution_score = -100
            for answer, score in paper_solution_answers:
                if score > solution_score:
                    solution = answer
                    solution_score = score
            summary_table["solution"].append(solution)
            
        return pd.DataFrame(summary_table)


    def _return_answers(self, question, texts_dict, min_score=None, 
                        max_length=128):

        sentence_tuples = self._split_text_to_sentences(texts_dict.keys(),
                                                       texts_dict.values())
        # create input examples with question and sentence (potential answer) pairs        
        input_examples = []
        for sentence_tuple in sentence_tuples:
            text_id, sentence_no, sentence = sentence_tuple
            input_example = InputExample(
                guid = str(text_id) + '_' + str(sentence_no),
                text_a = question,
                text_b = sentence
            )
            input_examples.append(input_example)

        # take input examples and convert to input features with appropriate padding
        input_features = []
        for idx, example in enumerate(input_examples):
            inputs = self.tokenizer.encode_plus(
                example.text_a,
                example.text_b,
                add_special_tokens=True,
                max_length=max_length
            )
            input_ids = inputs["input_ids"]
            token_type_ids = inputs["token_type_ids"]
    
            attention_mask = [1] * len(input_ids)
            padding_length = max_length - len(input_ids)
            pad_token = self.tokenizer.convert_tokens_to_ids(
                [self.tokenizer.pad_token])[0]
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)
            token_type_ids = token_type_ids + ([0] * padding_length)

            input_features.append(
                InputFeatures(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            label=None)
            )
    
        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in input_features], 
                                 dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask 
                                           for f in input_features], 
                                          dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids 
                                           for f in input_features], 
                                           dtype=torch.long)
        tensor_dataset = TensorDataset(all_input_ids, 
                                       all_attention_mask, 
                                       all_token_type_ids)

        # create dataloader to feed batches to torch model
        sampler = SequentialSampler(tensor_dataset)
        dataloader = DataLoader(tensor_dataset, 
                                 sampler=sampler, 
                                 batch_size=100)
        print(f"Ranking {len(sentence_tuples)} "
              f"possible answers from {len(texts_dict.keys())} texts:", 
              flush=True)
        # feed data to model and output logits i.e. [likelihood not answer, likelihood answer]
        all_logits = []

        with torch.no_grad():
            for batch in tqdm(dataloader, total=len(dataloader)):
                model_input = tuple(tensor.to(self.device) for tensor in batch)
                inputs = {'input_ids':      model_input[0],
                          'attention_mask': model_input[1]}
                batch_logits = self.model(**inputs)[0]
                if len(all_logits):
                    all_logits = np.concatenate([all_logits, batch_logits.cpu()])
                else:
                    all_logits = np.array(batch_logits.cpu())

        answer_score = all_logits[:,1] - all_logits[:,0]
        ranked_answers = answer_score.argsort()[::-1]

        answer_tuples = []
        for answer_idx in ranked_answers:
            if min_score is not None:
                if answer_score[answer_idx] < min_score:
                    break
            text_id, sentence_no, sentence = sentence_tuples[answer_idx]
            answer_tuples.append((text_id, 
                                  sentence_no, 
                                  sentence, 
                                  answer_score[answer_idx]))
        return answer_tuples


    def _split_text_to_sentences(self, text_ids, texts):

        # download nltk library to extract sentences from paragraphs
        nltk.download('punkt')
        sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        sentence_tuples = []
        for text_id, text in zip(text_ids, texts):
            sentences = sentence_tokenizer.tokenize(text)
            sentences = [sentence.lower() for sentence in sentences]
            for sentence_no, sentence in enumerate(sentences):
                sentence_tuples.append((text_id, sentence_no, sentence))
    
        return sentence_tuples
