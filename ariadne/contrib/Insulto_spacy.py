import os

import pandas as pd
from typing import List

from cassis import Cas

from ariadne.classifier import Classifier
from ariadne.contrib.inception_util import create_prediction, SENTENCE_TYPE, TOKEN_TYPE
from ariadne.protocol import TrainingDocument
import spacy
from ariadne.spacy_pattern import InsultoPattern
from spacy.language import Language

@Language.factory("insultos_merger")
def create_insulto_pattern_merger(nlp, name):
    return InsultoPattern(nlp.vocab, nlp)

class Insulto_spacy(Classifier):

    def __init__(self, is_classification: bool = False, model_name: str = "es_core_news_sm"):
        try:
            self.nlp = spacy.load(model_name)
            self.nlp.add_pipe("insultos_merger", last=True)
            self.is_classification = is_classification
        except OSError:
            print("Models doesn't exist; insulto")
            print(f"Downloading {model_name}...")
            spacy.cli.download(model_name)
            self.nlp = spacy.load(model_name)
            self.nlp.add_pipe("insultos_merger", last=True)
            self.is_classification = is_classification

    def split_text(self, row):
        id = row['text'].split()[0]
        row['text'] = row['text'].split(id)[1][1:]
        row['id'] = int(id)
        return row

    def _predict(self, sentences):
        """ Predicts the labels for the given sentences using the model.
            return: A list of predictions.
        """
        doc_spacy = [self.nlp(s) for s in sentences]
        fields = []
        begin = 0
        for sentence, doc in zip(sentences, doc_spacy):
            end = begin + len(sentence)
            for span in doc._.insultos_pattern:
                begin_span = begin + span['start']
                end_span = begin + span['end']
                fields.append({"begin": begin_span, "end": end_span, "Existe": 'Si'})
            begin = end + 1
        return fields

    def predict(self, cas: Cas, layer: str, feature: str, project_id: str, document_id: str, user_id: str):
        # Extract the tokens from the CAS and create a spacy doc from it
        #cas_tokens = cas.select(TOKEN_TYPE)
        sentences = cas.select(SENTENCE_TYPE)
        doc_spacy = [self.nlp(s.get_covered_text()) for s in sentences]
        for sentence, doc in zip(sentences, doc_spacy):
            for span in doc._.insultos_pattern:
                begin = sentence.begin + span['start']
                end = sentence.begin + span['end']
                prediction = create_prediction(cas, layer, feature, begin, end, 'Si')
                cas.add_annotation(prediction)

    def fit(self, documents: List[TrainingDocument], layer: str, feature: str, project_id, user_id: str):
        print('')