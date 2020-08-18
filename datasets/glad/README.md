# GLAD Benchmark

The General Language Analysis Datasets (GLAD) benchmark is an English-language benchmark for evaluating many different
natural language analysis tasks within a unified format of span labeling and relation prediction. It consists of 8
datasets covering 10 tasks in the areas of information extraction, syntactic analysis, semantic role labeling, and
sentiment analysis. See the [paper](https://www.aclweb.org/anthology/2020.acl-main.192/) for details.

## Setup

The GLAD benchmark consists of many different component tasks. Descriptions and citatiosn for all are available in
`glad.py`. First, make sure you have the requirements for preprocessing the data:

    pip install -r requirements.txt

Also install a model for SpaCy, which will be used for tokenization:

    python -m spacy download en_core_web_sm

Many of the tasks are easy to download/install (`wetlab`, `conll2003`, `semeval2010_8`, `oie2016`, `semeval2010_4`)
but some may require additional work or licenses. See below for details:

### Penn Treebank 3.0 (ptb3)

TODO: description

### OntoNotes 5.0 (ontonotes5)

TODO: description

### MPQA 3.0 (mpqa3)

TODO: description

## Reference

    @inproceedings{jiang-etal-2020-generalizing,
        title = "Generalizing Natural Language Analysis through Span-relation Representations",
        author = "Jiang, Zhengbao  and
          Xu, Wei  and
          Araki, Jun  and
          Neubig, Graham",
        booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
        month = jul,
        year = "2020",
        address = "Online",
        publisher = "Association for Computational Linguistics",
        url = "https://www.aclweb.org/anthology/2020.acl-main.192",
        doi = "10.18653/v1/2020.acl-main.192",
        pages = "2120--2133",
    }
