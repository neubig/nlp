import nlp
import textwrap

_CITATION = """\
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
"""

_DESCRIPTION = """\
The General Language Analysis Datasets (GLAD) benchmark is an English-language benchmark for evaluating many different
natural language analysis tasks within a unified format of span labeling and relation prediction. It consists of 8
datasets covering 10 tasks in the areas of information extraction, syntactic analysis, semantic role labeling, and
sentiment analysis.
"""

_DESCRIPTIONS = {
    "wetlab": textwrap.dedent(
        """\
            The wet lab protocols corpus is a corpus of natural language instructions consisting of 622 wet lab
            protocols to facilitate automatic or semi-automatic conversion of protocols into a machine-readable format
            and benefit biological research. Experimental results demonstrate the utility of our corpus for developing
            machine learning approaches to shallow semantic parsing of instructional texts."""
    ),
    "conll2003": textwrap.dedent(
        """\
            The CoNLL-2003 shared task dataset is a dataset to evaluate named entity recognition on English and German.
            It provides articles in the news domain and annotates them with several varieties of entities.
            """
    ),
    "semeval2010_8": textwrap.dedent(
        """\
        SemEval-2 Task 8 focuses on Multi-way classification of semantic relations between pairs of nominals in
        English. The task was designed to compare different approaches to semantic relation classification and to
        provide a standard testbed for future research."""
    ),
    "ontonotes5": textwrap.dedent(
        """\
        The OntoNotes corpus is a large-scale, multi-genre, multilingual corpus manually annotated with syntactic,
        semantic and discourse information. Specifically, it annotates a large corpus comprising various genres of text
        (news, conversational telephone speech, weblogs, usenet newsgroups, broadcast, talk shows) in three languages
        (English, Chinese, and Arabic) with structural information (syntax and predicate argument structure) and shallow
        semantics (word sense linked to an ontology and coreference)."""
    ),
    "ptb": textwrap.dedent(
        """\
        The Penn Treebank is a large corpus of American English annotated with POS tags and syntactic structure. It
        comprises approximately 7 million words of part-of-speech tagged text, 3 million words of skeletally parsed
        text, over 2 million words of text parsed for predicate-argument structure, and 1.6 million words of transcribed
        spoken text annotated for speech disfluencies."""
    ),
    "oie2016": textwrap.dedent(
        """\
        Open information extraction (Open IE) is an unrestricted variant of traditional information extraction. 
        The OIE2016 dataset is a supervised dataset for Open IE that leverages QA-SRL annotation to create an
        independent and large scale Open IE annotation in English for benchmarking Open IE systems."""
    ),
    "mpqa3": textwrap.dedent(
        """\
        The MPQA corpus is a rich span-annotated opinion corpus with entity and event target annotations. The new corpus
        provides a resource for developing systems for entity/event-level sentiment analysis."""
    ),
    "semeval2014_4": textwrap.dedent(
        """\
        SemEval2014 Task 4 aimed to foster research in the field of aspect-based sentiment analysis, where the goal is
        to identify the aspects of given target entities and the sentiment expressed for each aspect. The task provided
        datasets containing manually annotated reviews of restaurants and laptops, as well as a common evaluation
        procedure."""
    )
}

_CITATIONS = {
    "wetlab": textwrap.dedent(
        """\
        @inproceedings{kulkarni-etal-2018-annotated,
            title = "An Annotated Corpus for Machine Reading of Instructions in Wet Lab Protocols",
            author = "Kulkarni, Chaitanya  and
              Xu, Wei  and
              Ritter, Alan  and
              Machiraju, Raghu",
            booktitle = "Proceedings of the 2018 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers)",
            month = jun,
            year = "2018",
            address = "New Orleans, Louisiana",
            publisher = "Association for Computational Linguistics",
            url = "https://www.aclweb.org/anthology/N18-2016",
            doi = "10.18653/v1/N18-2016",
            pages = "97--106",
            abstract = "We describe an effort to annotate a corpus of natural language instructions consisting of 622 wet lab protocols to facilitate automatic or semi-automatic conversion of protocols into a machine-readable format and benefit biological research. Experimental results demonstrate the utility of our corpus for developing machine learning approaches to shallow semantic parsing of instructional texts. We make our annotated Wet Lab Protocol Corpus available to the research community.",
        }"""
    ),
    "conll2003": textwrap.dedent(
        """\
        @inproceedings{tjong-kim-sang-de-meulder-2003-introduction,
            title = "Introduction to the {C}o{NLL}-2003 Shared Task: Language-Independent Named Entity Recognition",
            author = "Tjong Kim Sang, Erik F.  and
              De Meulder, Fien",
            booktitle = "Proceedings of the Seventh Conference on Natural Language Learning at {HLT}-{NAACL} 2003",
            year = "2003",
            url = "https://www.aclweb.org/anthology/W03-0419",
            pages = "142--147",
        }"""
    ),
    "semeval2010_8": textwrap.dedent(
        """\
        @inproceedings{hendrickx-etal-2010-semeval,
            title = "{S}em{E}val-2010 Task 8: Multi-Way Classification of Semantic Relations between Pairs of Nominals",
            author = "Hendrickx, Iris  and
              Kim, Su Nam  and
              Kozareva, Zornitsa  and
              Nakov, Preslav  and
              {\'O} S{\'e}aghdha, Diarmuid  and
              Pad{\'o}, Sebastian  and
              Pennacchiotti, Marco  and
              Romano, Lorenza  and
              Szpakowicz, Stan",
            booktitle = "Proceedings of the 5th International Workshop on Semantic Evaluation",
            month = jul,
            year = "2010",
            address = "Uppsala, Sweden",
            publisher = "Association for Computational Linguistics",
            url = "https://www.aclweb.org/anthology/S10-1006",
            pages = "33--38",
        }"""
    ),
    "ontonotes5": textwrap.dedent(
        """\        
        @inproceedings{pradhan-etal-2013-towards,
            title = "Towards Robust Linguistic Analysis using {O}nto{N}otes",
            author = {Pradhan, Sameer  and
              Moschitti, Alessandro  and
              Xue, Nianwen  and
              Ng, Hwee Tou  and
              Bj{\"o}rkelund, Anders  and
              Uryupina, Olga  and
              Zhang, Yuchen  and
              Zhong, Zhi},
            booktitle = "Proceedings of the Seventeenth Conference on Computational Natural Language Learning",
            month = aug,
            year = "2013",
            address = "Sofia, Bulgaria",
            publisher = "Association for Computational Linguistics",
            url = "https://www.aclweb.org/anthology/W13-3516",
            pages = "143--152",
        }"""
    ),
    "ptb": textwrap.dedent(
        """\        
        @article{marcus-etal-1993-building,
            title = "Building a Large Annotated Corpus of {E}nglish: The {P}enn {T}reebank",
            author = "Marcus, Mitchell P.  and
              Santorini, Beatrice  and
              Marcinkiewicz, Mary Ann",
            journal = "Computational Linguistics",
            volume = "19",
            number = "2",
            year = "1993",
            url = "https://www.aclweb.org/anthology/J93-2004",
            pages = "313--330",
        }"""
    ),
    "oie2016": textwrap.dedent(
        """\
        @inproceedings{stanovsky-dagan-2016-creating,
            title = "Creating a Large Benchmark for Open Information Extraction",
            author = "Stanovsky, Gabriel  and
              Dagan, Ido",
            booktitle = "Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing",
            month = nov,
            year = "2016",
            address = "Austin, Texas",
            publisher = "Association for Computational Linguistics",
            url = "https://www.aclweb.org/anthology/D16-1252",
            doi = "10.18653/v1/D16-1252",
            pages = "2300--2305",
        }"""
    ),
    "mpqa3": textwrap.dedent(
        """\
        @inproceedings{deng-wiebe-2015-mpqa,
            title = "{MPQA} 3.0: An Entity/Event-Level Sentiment Corpus",
            author = "Deng, Lingjia  and
              Wiebe, Janyce",
            booktitle = "Proceedings of the 2015 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies",
            month = may # "{--}" # jun,
            year = "2015",
            address = "Denver, Colorado",
            publisher = "Association for Computational Linguistics",
            url = "https://www.aclweb.org/anthology/N15-1146",
            doi = "10.3115/v1/N15-1146",
            pages = "1323--1328",
        }"""
    ),
    "semeval2014_4": textwrap.dedent(
        """\
        @inproceedings{pontiki-etal-2014-semeval,
            title = "{S}em{E}val-2014 Task 4: Aspect Based Sentiment Analysis",
            author = "Pontiki, Maria  and
              Galanis, Dimitris  and
              Pavlopoulos, John  and
              Papageorgiou, Harris  and
              Androutsopoulos, Ion  and
              Manandhar, Suresh",
            booktitle = "Proceedings of the 8th International Workshop on Semantic Evaluation ({S}em{E}val 2014)",
            month = aug,
            year = "2014",
            address = "Dublin, Ireland",
            publisher = "Association for Computational Linguistics",
            url = "https://www.aclweb.org/anthology/S14-2004",
            doi = "10.3115/v1/S14-2004",
            pages = "27--35",
        }"""
    )
}

_TEXT_FEATURES = {
    "wetlab": {"todo": "todo"},
    "conll2003": {"todo": "todo"},
    "semeval2010_8": {"todo": "todo"},
    "ontonotes5": {"todo": "todo"},
    "ptb": {"todo": "todo"},
    "oie2016": {"todo": "todo"},
    "mpqa3": {"todo": "todo"},
    "semeval2014_4": {"todo": "todo"},
}

_DATA_URLS = {
    "wetlab": "https://github.com/jeniyat/WNUT_2020/tree/master/data",
    "conll2003": "https://www.clips.uantwerpen.be/conll2003/ner.tgz",
    "semeval2010_8": "https://github.com/sahitya0000/Relation-Classification/blob/master/corpus/SemEval2010_task8_all_data.zip",
    "ontonotes5": "",
    "ptb": "",
    "oie2016": "https://github.com/gabrielStanovsky/oie-benchmark/blob/master/snapshot_oie_corpus.tar.gz",
    "mpqa3": "",
    "semeval2014_4": "http://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools",
}

_URLS = {
    "wetlab": "https://github.com/jeniyat/WNUT_2020/tree/master/data",
    "conll2003": "https://www.clips.uantwerpen.be/conll2003/ner/",
    "semeval2010_8": "https://docs.google.com/document/d/1QO_CnmvNRnYwNWu1-QCAeR5ToQYkXUqFeAJbdEhsq7w/preview",
    "ontonotes5": "https://catalog.ldc.upenn.edu/LDC2013T19",
    "ptb": "https://catalog.ldc.upenn.edu/LDC99T42",
    "oie2016": "https://github.com/gabrielStanovsky/oie-benchmark",
    "mpqa3": "https://mpqa.cs.pitt.edu/corpora/mpqa_corpus/",
    "semeval2014_4": "http://alt.qcri.org/semeval2014/task4/",
}

