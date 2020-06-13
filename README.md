# tanda_search_qa_tool

Tools and utility scripts written as part of a submission to the [Kaggle COVID-19 Open Research Dataset Challenge (CORD-19)](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/)

The tool uses the Amazon Alexa Team's Transfer and Adapt Bert framework and a pre-trained TandA RoBERTa Large model also provided by the team. All files in the `transformers` directory (adaptations of Hugging Face's popular transformers package) are copied directly from the [wqa_tanda repo](https://www.google.com) / [original arxiv paper](https://arxiv.org/abs/1911.04118) where all the necessary Ts & Cs can be found:

## contents

* **build_summaries_df.py**
  
  A short script to build a dataframe consolidating the entries from the kaggle summary table examples. Used to test output of final model
