# tanda_search_qa_tool

Tools and utility scripts written as part of a submission to the [Kaggle COVID-19 Open Research Dataset Challenge (CORD-19)](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/). A report, including examples of the tools in use, can be found in [the task submission notebook](https://www.kaggle.com/dustyturner/cord19-task-2-population-studies) (winner of a community contribution award).

The tool uses the Amazon Alexa Team's Transfer and Adapt Bert framework and a pre-trained TandA RoBERTa Large model also provided by the team. All files in the `transformers` directory (adaptations of Hugging Face's popular transformers package) are copied directly from the [wqa_tanda repo](https://github.com/alexa/wqa_tanda) / [original arxiv paper](https://arxiv.org/abs/1911.04118) where all the necessary Ts & Cs can be found. Other files are the authors own:

## contents

* **build_kaggle_summaries_df.py**
  
  Short script used to concatenate the summary tables provided by Kaggle into one large table. Used to attempt semi-supervised learning (no success) and analysis of model output.

* **cord_result_summarizer.py**
  
  Tool to build summary table entries from individual papers. Uses Roberta-Tanda model to identify sentences from paper to use as **challenge** and **solution** features, and regex / spacy POS & NER tagging to find **study_type**, **strength_of_evidence** and **addressed_population**.
  
* **cord_search_qa_tool.py**
  
  Tool to search Cord-19 corpora to identify papers that answer a given research question. Utilises regex search terms to identify possible results and Roberta-TandA model to identify papers with sentence-level answers to a research question within the abstract. 

* **prep_metadata.py**
  
  A short script to clean metadata provided with Cord-19 dataset and add missing abstracts to papers that have available body text.
  
* **summarizer_helpers.py**
  
  Helper functions that facilitate the cord_search_qa_tool.
  
* **text_search_qa_tool.py**
  
  Generalised search tool from which the cord_search_qa_tool is built. Takes a corpora of texts and utilises regex search and Roberta-TandA sentence-level QA to find texts that contain an answer to a given question.
  
## Requirements

Alongside the python packages in requirements.txt, the model requires the following to function:

* a pre-trained variant of the TandA model, available [here](https://github.com/alexa/wqa_tanda). Extract in the main directory of the project and refer to the directory name when initializing the respective tool. Note: the model should function with any of the pre-trained TandA models, but has been tested using the "transferred" Roberta base and large variants only.
* The [CORD-19 dataset](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/) (if wanting to use the `cord_search_qa_tool` and `cord_result_summarizer`) again extracted in the main directory of the project. 
* The `en_core_web_sm` spacy model. This must be downloaded separately after installing the spacy library by running `python -m spacy download en_core_web_sm` at the CLI
