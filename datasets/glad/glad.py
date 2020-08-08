import nlp
import textwrap
import six
import json
import os

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

_NAMES = ["wetlab", "conll2003", "semeval2010_8", "ontonotes5", "ptb", "oie2016", "mpqa3", "semeval2014_4"]

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
    "conll2003": {"words": "", "pos_tags": "", "chunk_spans": "", "ner_spans": ""},
    "semeval2010_8": {"todo": "todo"},
    "ontonotes5": {"todo": "todo"},
    "ptb": {"todo": "todo"},
    "oie2016": {"todo": "todo"},
    "mpqa3": {"todo": "todo"},
    "semeval2014_4": {"todo": "todo"},
}

_DATA_URLS = {
    "wetlab": "https://github.com/jeniyat/WNUT_2020/tree/master/data",
    "conll2003": "https://raw.githubusercontent.com/glample/tagger/master/dataset/",
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

class GladConfig(nlp.BuilderConfig):
    """BuilderConfig for Break"""

    def __init__(self, data_url, citation, url, text_features, **kwargs):
        """

        Args:
            text_features: `dict[string, string]`, map from the name of the feature
        dict for each text field to the name of the column in the tsv file
            label_column:
            label_classes
            **kwargs: keyword arguments forwarded to super.
        """
        super(GladConfig, self).__init__(
            version=nlp.Version("1.0.0", "New split API (https://tensorflow.org/datasets/splits)"), **kwargs
        )
        self.text_features = text_features
        self.data_url = data_url
        self.citation = citation
        self.url = url

class Glad(nlp.GeneratorBasedBuilder):
    """\
    The General Language Analysis Datasets (GLAD) benchmark is an English-language benchmark for evaluating many different
    natural language analysis tasks within a unified format of span labeling and relation prediction. It consists of 8
    datasets covering 10 tasks in the areas of information extraction, syntactic analysis, semantic role labeling, and
    sentiment analysis.
    """

    # TODO(glad): Set up version.
    VERSION = nlp.Version("0.1.0")
    BUILDER_CONFIGS = [
        GladConfig(
            name=name,
            description=_DESCRIPTIONS[name],
            citation=_CITATIONS[name],
            text_features=_TEXT_FEATURES[name],
            data_url=_DATA_URLS[name],
            url=_URLS[name],
        )
        for name in _NAMES
    ]

    @property
    def manual_download_instructions(self):
        if self.config.name.startswith("ptb") or self.config.name.startswith("ontonotes"):
            return textwrap.dedent("""\
             To use the Penn Treebank Version 3.0 (LDC99T42) and Ontonotes 5.0 (LDC2013T19) components of GLAD, you need
             to download them from the LDC web site (https://catalog.ldc.upenn.edu/) and save the corresponding
             tarballs (LDC99T42.tgz) and (LDC2013T19.tgz) in a folder. The folder containing the saved files can be
             used to load the dataset via `nlp.load_dataset("glad", data_dir="<path/to/folder>")`.
            """)
        if self.config.name.startswith("mpqa"):
            return textwrap.dedent("""\
             To use the MPQA 3.0 component of GLAD, you need to go to the MPQA web site
             (https://mpqa.cs.pitt.edu/corpora/mpqa_corpus/), fill out the form to download MPQA 3.0, and save the file
             (mpqa_3_0_database.zip) in a folder. The folder containing the saved file can be used to load the dataset
             via `nlp.load_dataset("glad", data_dir="<path/to/folder>")`.
            """)
        return None


    def _info(self):
        features = {text_feature: nlp.Value("string") for text_feature in six.iterkeys(self.config.text_features)}
        # TODO: These need to be added appropriately
        # if "answers" in features.keys():
        #     features["answers"] = nlp.features.Sequence(
        #         {"answer_start": nlp.Value("int32"), "text": nlp.Value("string")}
        #     )
        # if self.config.name.startswith("PAWS-X"):
        #     features["label"] = nlp.Value("string")
        # if self.config.name == "XNLI":
        #     features["gold_label"] = nlp.Value("string")

        if self.config.name.startswith("conll2003"):
            features = nlp.Features(
                {
                    "words": nlp.Sequence(nlp.Value("string")),
                    "pos_tags": nlp.Sequence(nlp.Value("string")),
                    "chunk_spans": nlp.features.Sequence(
                        {"start": nlp.Value("int32"), "end": nlp.Value("int32"), "tag": nlp.Value("string") }
                    ),
                    "ner_spans":
                        {"start": nlp.Value("int32"), "end": nlp.Value("int32"), "tag": nlp.Value("string")}
                }
            )
        return nlp.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=self.config.description + "\n" + _DESCRIPTION,
            # nlp.features.FeatureConnectors
            features=nlp.Features(
                features
                # These are the features of your dataset like images, labels ...
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage="https://github.com/neulab/cmu-multinlp" + "\t" + self.config.url,
            citation=self.config.citation + "\n" + _CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # dl_manager is a nlp.download.DownloadManager that can be used to
        # download and extract URLs

        if self.config.name == "wetlab":
            raise NotImplementedError('not implemented')
        elif self.config.name == "conll2003":
            urls_to_download = {
                "train": os.path.join(self.config.data_url, "eng.train"),
                "dev": os.path.join(self.config.data_url, "eng.testa"),
                "test": os.path.join(self.config.data_url, "eng.testb"),
            }
            downloaded_files = dl_manager.download_and_extract(urls_to_download)
            return [
                nlp.SplitGenerator(name=nlp.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
                nlp.SplitGenerator(name=nlp.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["dev"]}),
                nlp.SplitGenerator(name=nlp.Split.TEST, gen_kwargs={"filepath": downloaded_files["test"]}),
            ]
        elif self.config.name == "semeval2010_8":
            raise NotImplementedError('not implemented')
        elif self.config.name == "ontonotes5":
            raise NotImplementedError('not implemented')
        elif self.config.name == "ptb":
            raise NotImplementedError('not implemented')
        elif self.config.name == "oie2016":
            raise NotImplementedError('not implemented')
        elif self.config.name == "mpqa3":
            raise NotImplementedError('not implemented')
        elif self.config.name == "semeval2014_4":
            raise NotImplementedError('not implemented')
        else:
            raise ValueError(f'Invalid config name {self.config.name}')

    def _generate_examples(self, filepath):
        """Yields examples."""

        if self.config.name == "wetlab":
            raise NotImplementedError('not implemented')
        elif self.config.name.startswith("conll2003"):
            guid_index = 1
            with open(filepath, encoding="utf-8") as f:
                words, pos_tags, chunk_spans, ner_spans = [], [], [], []
                chunk_start, chunk_tag, ner_start, ner_tag = 0, None, 0, None
                for line in f:
                    if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                        if words:
                            yield guid_index, {"words": words, "pos_tags": pos_tags,
                                               "chunk_spans": chunk_spans, "ner_spans": ner_spans}
                            guid_index += 1
                            words, pos_tags, chunk_spans, ner_spans = [], [], [], []
                    else:
                        # conll2003 data is space separated, 'IO' tagging scheme for chunks and named entities
                        splits = line.strip().split(" ")
                        tag = splits[2][2:] if splits[2].startswith('I-') else None
                        if tag != chunk_tag:
                            if chunk_tag is not None:
                                chunk_spans.append( {'start': chunk_start, 'end': len(words)+1, 'tag': chunk_tag} )
                            chunk_tag, chunk_start = tag, len(words)
                        tag = splits[3][2:] if splits[3].startswith('I-') else None
                        if tag != ner_tag:
                            if ner_tag is not None:
                                ner_spans.append( {'start': ner_start, 'end': len(words)+1, 'tag': ner_tag} )
                            ner_tag, ner_start = tag, len(words)
                        words.append(splits[0])
                        pos_tags.append(splits[1])
        elif self.config.name == "semeval2010_8":
            raise NotImplementedError('not implemented')
        elif self.config.name == "ontonotes5":
            raise NotImplementedError('not implemented')
        elif self.config.name == "ptb":
            raise NotImplementedError('not implemented')
        elif self.config.name == "oie2016":
            raise NotImplementedError('not implemented')
        elif self.config.name == "mpqa3":
            raise NotImplementedError('not implemented')
        elif self.config.name == "semeval2014_4":
            raise NotImplementedError('not implemented')
        else:
            raise ValueError(f'Invalid config name {self.config.name}')

# TODO: This is for debugging, remove before final commit
if __name__ == "__main__":
    from nlp import load_dataset
    dataset = load_dataset("./datasets/glad", "conll2003")
    for spl in ('train', 'validation', 'test'):
        dataset_spl = dataset[spl]
        print(dataset_spl[0])