import os

import pandas as pd
from typing import List
import json

from cassis import Cas

from ariadne.classifier import Classifier
from ariadne.contrib.inception_util import create_prediction, SENTENCE_TYPE
from active_learning_proccess.src.models.models import SequenceModel
from ariadne.protocol import TrainingDocument
from active_learning_proccess.src.scripts.generic_transformer_al import load_active_learning


class GenericTransformersClassifier(Classifier):

    def __init__(self, model_name: str, model_arg_path: str, config_path: str, map_labels: dict, is_unlabelled: bool = True, is_classification: bool = True):
        try:
            self.model_arg_path = model_arg_path
            self.config_path = config_path
            self.map_labels = map_labels
            self.is_unlabelled = is_unlabelled
            self.is_classification = is_classification
            df_model_args = pd.read_json(os.getcwd() + model_arg_path)
            model_args = df_model_args.to_dict(orient='records')[0]
            with open(os.getcwd() + config_path) as f:
                self.config = json.load(f)
            self._model = SequenceModel(self.config["model_type"], os.getcwd() + "/models/" + model_name, self.config["use_cuda"],
                                        self.config["classes"], '', False, "", False, 0, "", self.config["model_name"], "", model_args=model_args)
        except OSError:
            print(model_name + " model doesn't exist")

    def split_text(self, row):
        id = row['text'].split()[0]
        row['text'] = row['text'].split(id)[1][1:]
        row['id'] = id
        return row

    def _predict(self, df):
        """ Predicts the labels for the given sentences using the model.
            return: A list of predictions.
        """
        return self._model.predict(df)

    def predict(self, cas: Cas, layer: str, feature: str, project_id: str, document_id: str, user_id: str):
        # Extract the tokens from the CAS and create a spacy doc from it
        #cc = list(cas.select(SENTENCE_TYPE))
        #print(cc)
        sentences = cas.select(SENTENCE_TYPE)
        sentences_string = [s.get_covered_text() for s in sentences]
        df_featurized_sentences = pd.DataFrame(sentences_string, columns=["text"])
        df_featurized_sentences = df_featurized_sentences.apply(self.split_text, axis=1)
        predictions, sentences = self._model.predict(df_featurized_sentences)

        for sentence, label in zip(sentences, predictions):
            label = self.map_labels[label]
            prediction = create_prediction(cas, layer, feature, sentence.begin, sentence.end, label)
            cas.add_annotation(prediction)

    def fit(self, documents: List[TrainingDocument], layer: str, feature: str, project_id, user_id: str):
        sentences = []
        targets = []

        for document in documents:
            cas = document.cas

            for sentence in cas.select(SENTENCE_TYPE):
                # Get the first annotation that covers the sentence
                annotations = cas.select_covered(layer, sentence)

                if len(annotations):
                    annotation = annotations[0]
                else:
                    continue

                assert (
                    sentence.begin == annotation.begin and sentence.end == annotation.end
                ), "Annotation should cover sentence fully!"

                label = getattr(annotation, feature)

                if label is None:
                    continue

                sentences.append(cas.get_covered_text(sentence))
                targets.append(label)
        df = pd.DataFrame(list(zip(sentences, targets)), columns=['text', 'label'])
        df = df.apply(self.split_text, axis=1)
        load_active_learning(config_path=self.config_path,
                         model_arg=self.model_arg_path, df=df, is_unlabelled=self.is_unlabelled)