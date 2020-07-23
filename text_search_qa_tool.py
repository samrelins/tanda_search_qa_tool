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


class SearchResult:
    def __init__(self, name, ids, containing=[], not_containing=[]):
        self.name = name
        self.ids = ids
        self.search_no = 1
        if containing:
            self.containing = [(self.search_no, containing)]
        else:
            self.containing = []
        if not_containing:
            self.not_containing = [(self.search_no, not_containing)]
        else:
            self.not_containing = []


    def update(self, ids, containing=[], not_containing=[]):
        self.ids = ids
        self.search_no += 1
        if containing:
            self.containing.append((self.search_no, containing))
        if not_containing:
            self.not_containing.append((self.search_no, not_containing))


    def __repr__(self):
        repr_string = f"## SearchResult {self.name} ##\n"
        repr_string += '=' * 50 + '\n'
        repr_string += f"{self.search_no} Searches:\n"
        for i in range(self.search_no):
            if i == 0:
                repr_string += "Original Search:\n"
            else:
                repr_string += f"Refined Search {i}:\n"
            for containing_item in self.containing:
                if containing_item[0] == i + 1:
                    repr_string += f"\tcontaining: {containing_item[1]}\n"
            for not_containing_item in self.not_containing:
                if not_containing_item[0] == i + 1:
                    repr_string += ("\tnot containing: "
                                    f"{not_containing_item[1]}\n")
        return repr_string


class TextSearchQATool:
    def __init__(self, texts, qa_model_dir):
        print('=' * 100)
        print("Building Search tool")
        self.texts = texts
        self.search_results = {}
        self._init_qa_model_atts(qa_model_dir)


    def _init_qa_model_atts(self, qa_model_dir):
        # Load pretrained tokenizer
        qa_model_dir += "/models/tanda_roberta_base_asnq/ckpt"
        if not os.path.exists(qa_model_dir):
            raise TypeError("TandA directory doesn't exist or path to ckpt "
                            f"doesn't match {qa_model_dir}")
            
        tokenizer_class = RobertaTokenizer
        self.tokenizer = tokenizer_class.from_pretrained(qa_model_dir,
                                                do_lower_case=True,
                                                cache_dir=None)
        
        # collect GPU variables to mount tensors
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else: 
            self.device = torch.device("cpu")

        print("Initialising QA Model")
        # define model and set to eval
        model_class = RobertaForSequenceClassification
        self.model = model_class.from_pretrained(qa_model_dir)
        self.model.to(self.device)
        self.model.eval()
        print("QA Model Loaded and waiting for questions")
        print('=' * 100)


    def clear_searches(self):
        print(f"Clearing {len(self.search_results)} searches")
        self.search_results = {}


    def search(self, containing=[], not_containing=[], search_name=None, 
               containing_threshold=0):
        """
        Search texts for the presence / absence of specific keywords.
        
        A regex search identifies appearances of regular expressions in 
        each of the texts from the "texts" attribute. Appearances of
        the expressions in the "containing" parameter list are counted, and 
        texts  that contain a number greater than the "containing_threshold" 
        parameter  are added to the search results. If keywords from the 
        "not_containing" parameter list are found in the text, these are 
        excluded from the results. 
        
        The search follows an OR logic. For example, a search for texts 
        containing ["cat", "dog"] will return any texts that feature the 
        regex "cat" or "dog" in sufficient number, or exclude texts that 
        contain either, depending on the parameters. A search must specify a 
        "containing" or "not_containing" parameter, or both.
        
        
        Parameters
        __________
        
            containing: list
                list of regular expressions, appearances of which are counted 
                for each text.
            not_containing: list
                list of regular expressions that are to be excluded from the 
                search results. 
            containing_threshold: int, default 2
                the number of hits from the containing list required for a
                text to be added to the results
            search_name: str, optional
                Name to be used as key in "search_results" dict attribute. If
                None, a name is automatically assigned in the format 
                "search[int]"
                
        Returns
        _______
        
        None -  Results are stored as a SearchResult object in the 
                "search_results" attribute.
        
        """
        
        # check inputs are correct
        if not containing and not not_containing:
            return TypeError("Must specify one of containing "
                             "/ not containing input variables")
        if containing and type(containing) is not list:
            return TypeError(f"containing cannot be type {type(containing)} "
                             "- must be type list of regular expressions")
        if not_containing and type(not_containing) is not list:
            return TypeError("not containing cannot be type "
                             f"`{type(not_containing)}` - must be type "
                             "`list` of regular expressions")
        search_names= self.search_results.keys()
        if search_name is not None and search_name in search_names:
            return TypeError(f"Search name {search_name} already exists. "
                             "If you wish to refine these results, "
                             "call the refine method.") 

        # find an unused search name if none entered
        if search_name is None:
            search_name = "search0"
            search_name_idx = 0
            while search_name in self.search_results.keys():
                search_name_idx += 1
                search_name = f"search{search_name_idx}"

        # print search details to console
        print('=' * 100)
        print(f"Search ## {search_name} ## created:")
        print(f"Searching {len(self.texts)} texts with search parameters:")
        if containing:
            print(f"\tcontaining: {containing}")
        if not_containing:
            print(f"\tnot containing: {(not_containing)}")
        print(f"above a threshold of {containing_threshold}")
        
        # call search function to return ids of required results
        search_results_ids = self._search_by_texts_ids(
            search_texts_ids = self.texts.keys(),
            containing=containing,
            not_containing=not_containing,
            containing_threshold=containing_threshold
        )

        # store results in search result object
        print('=' * 100)
        if search_results_ids:
            self.search_results[search_name] = SearchResult(
                name = search_name,
                ids = search_results_ids,
                containing = containing,
                not_containing = not_containing
            )
            print(f"{len(search_results_ids)} search results "
                   "returned and stored in {search_name}")
        else:
            print(f"Search returned no results")


    def refine_search(self, search_name, containing=[], not_containing=[], 
                      containing_threshold=2):

        # check for correct input
        if search_name not in self.search_results.keys():
            return TypeError(f"Invalid search name {search_name}. "
                             f"Try one of {self.search_results.keys()}") 
        if not containing and not not_containing:
            return TypeError("Must specify one of containing "
                             "/ not containing input variables")
        if containing and type(containing) is not list:
            return TypeError(f"containing cannot be type {type(containing)} "
                             "- must be type list of regular expressions")
        if not_containing and type(not_containing) is not list:
            return TypeError("not containing cannot be type "
                             f"{type(not_containing)} - must be "
                             "type list of regular expressions")

        # find required SearchResult from search_results dict
        search = self.search_results[search_name]

        # print search details to console
        print('=' * 100)
        print(f"Refining search results from ## {search.name} ##")
        print(f"Searching {len(search.ids)} with search parameters:")
        if containing:
            print(f"containing:\n {containing}")
        if not_containing:
            print(f"not containing:{not_containing}")

        # call search function over search ids to return refined search ids
        refined_results_ids = self._search_by_texts_ids(
            search_texts_ids = search.ids,
            containing=containing,
            not_containing=not_containing,
            containing_threshold=containing_threshold
        )

        # store refined results
        print('=' * 100)
        if refined_results_ids:
            search.update(
                ids = refined_results_ids,
                containing = containing,
                not_containing = not_containing
            )
            print(f"{len(refined_results_ids)} refined results returned "
                  f"and stored in {search_name}")
        else:
            print(f"Search returned no results")


    def _search_by_texts_ids(self, search_texts_ids, containing, not_containing,
                             containing_threshold):

        output_ids = search_texts_ids
        if containing:
            output_ids = []
            containing_p = re.compile("|".join(containing))
            for text_id in search_texts_ids:
                text = self.texts[text_id].lower()
                n_hits = len([hit for hit in containing_p.finditer(text)])
                if n_hits > containing_threshold:
                    output_ids.append(text_id)
        if not_containing:
            not_containing_p = re.compile("|".join(not_containing))
            output_ids = (
                [text_id for text_id in output_ids
                 if not not_containing_p.search(self.texts[text_id].lower())]
            )

        return output_ids


    def return_answers(self, question, search_name=None, 
                       min_score=None, max_length=128):

        if not search_name is None:
            search_texts_ids = self.search_results[search_name].ids
        else:
            search_texts_ids = self.texts.keys()

        print('=' * 100)
        print(f"Checking {len(search_texts_ids)} search results "
              f"for answers to {question}")

        # collect texts that correspond with ids from search 
        # and create (sentence, text_id) tuples
        search_texts = [self.texts[text_id] for text_id in search_texts_ids]
        sentence_tuples = self._split_text_to_sentences(search_texts_ids,
                                                        search_texts)
        # create input examples with question 
        # and sentence (potential answer) pairs 
        input_examples = []
        for sentence_tuple in sentence_tuples:
            text_id, sentence_no, sentence = sentence_tuple
            input_example = InputExample(
                guid = str(text_id) + '_' + str(sentence_no),
                text_a = question,
                text_b = sentence
            )
            input_examples.append(input_example)
        print("Inputs converted to BERT InputExamples")

        # take input examples and convert to input features with padding
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
    
        print("InputExamples converted to InputFeatures")
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
        print("InputFeatures converted to TensorDataset")

        # create dataloader to feed batches to torch model
        sampler = SequentialSampler(tensor_dataset)
        dataloader = DataLoader(tensor_dataset, 
                                 sampler=sampler, 
                                 batch_size=100)
        print("TensorDataset converted to torch DataLoader")
        print(f"Ranking {len(sentence_tuples)} possible answers from "
              f"{len(search_texts)} texts:", flush=True)
        # feed data to model and output logits 
        # i.e. [likelihood not answer, likelihood answer]
        all_logits = []

        with torch.no_grad():
            for batch in tqdm(dataloader, total=len(dataloader)):
                model_input = tuple(tensor.to(self.device) for tensor in batch)
                inputs = {'input_ids':      model_input[0],
                          'attention_mask': model_input[1]}
                batch_logits = self.model(**inputs)[0]
                if len(all_logits):
                    all_logits = np.concatenate([all_logits, 
                                                 batch_logits.cpu()])
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


    def return_html_answers(self, search_name, question, min_score=None,
                            highlight_score=-2, top_n=None, max_length=128):
        """
        Generate list of answer sentences from `texts` along with a 
        HTML output of the results.
        
        This method is used to identify sentences from the `texts` attribute
        that are potential answers to the `question` parameter, in the same 
        manner as the `return_answers` method. Given identical search 
        parameters, the `return_answers` and `return_html_answers` methods 
        will return an identical list of results. 
        
        ** See `return_answers` for more info on 
           the QA search process and output **
        
        Additionally, this method returns a HTML output of the answer results 
        to help visualise the QA search. Sentences within texts that 
        receive a QA score higher than the `highlight_score` parameter are 
        highlighted in cyan. Intended for use with the Ipython.core.display 
        module's `display` and `HTML` functions  e.g.:
        
        >>> answers_html = searchtool.return_html_answers(
            [your search parameters here]
        )
        >>> display(HTML(answers_html))
        
        (the above displays the results of the answer search in the cell output)
        
        Parameters
        __________
        
             search_name: str
                 Key of search, texts from which will be used to generate 
                 answers
             question: str 
                 Question to be used to score potential answers against
             min_score: int, optional
                 Minimum QA score for a sentence to be included in results.
                 If none, every sentence is included in the results
             highlight_score: int, default -2
                 Minimum QA score for a sentence to be highlighted in the HTML 
                 output
             top_n: int, default 10
                 The number of results to return in the HTML output
             max_length: int, default 128
                 Parameter used by the Roberta model to fix the input length
                
        Returns
        _______
        
            tuple: (list: QA sentence scores, str: HTML) 
        
        """

        answer_tuples = self.return_answers(search_name=search_name,
                                            question=question,
                                            min_score=None,
                                            max_length=max_length)

        answer_df = pd.DataFrame(answer_tuples, columns=["text_id",
                                                         "sentence_no",
                                                         "sentence",
                                                         "score"])
        
        top_answers = (answer_df[["text_id", "score"]]
                       .groupby("text_id", as_index=False)
                       .max().sort_values("score", ascending=False))

        html_results = ""
        for idx, text_id in enumerate(top_answers.text_id.values):

            if top_n is not None:
                if idx == top_n:
                    break  

            text_mask = answer_df.text_id == text_id
            text = answer_df[text_mask].sort_values("sentence_no")
            html_results += "<h2><strong>" + text_id + "</h2></strong>"

            html_results += "<p>"
            for entry in text.itertuples():
                if entry.score > highlight_score:
                    html_results += ("<strong style='color:yellow'>" 
                                     f"{entry.sentence.capitalize()} </strong>")
                else:
                    html_results += f"{entry.sentence.capitalize()} "
            html_results += "</p><br>"
            
        if min_score is None:
            output_answers = answer_tuples
        else:
            output_answers = [answer for answer in answer_tuples 
                              if answer[3] > min_score]
            
        return output_answers, html_results


    def return_html_search_results(self, search_name, n_results=100):
        """
        Generate HTML of the results of the `search` method.
        
        This method can be used to visualise the results of the `search` method
        in a more readable format than printing to the console. Intended for use 
        with the  Ipython.core.display module's `display` and `HTML` functions 
        e.g.:
        
        >>> search_html = searchtool.return_html_search_results("search0")
        >>> display(HTML(html_search_output))
        
        (the above will display the results of "search0" in the cell output)
        
        Parameters
        __________
        
            search_name: str
                A named search / key value from the `search_results` attribute
            n_results:
                The number of results to be returned. Results appear in the same
                order as the items in the `texts` attribute
                
        Returns
        _______
        
            Str: HTML of texts from the specified search
        
        """
        
        html_results = ""
        for idx, text_id in enumerate(self.search_results[search_name].ids):
            if idx == n_results:
                break
            html_results += "<h2><strong>" + text_id + "</h2></strong>"
            html_results += "<p>" + self.texts[text_id] + "<p>"
        return html_results
    
    