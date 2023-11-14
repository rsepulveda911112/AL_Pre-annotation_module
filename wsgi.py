from ariadne.server import Server
from ariadne.util import setup_logging
from ariadne.contrib.Insulto_spacy import Insulto_spacy
from ariadne.contrib.Generic_transformers_classifier import GenericTransformersClassifier

setup_logging()
server = Server()

VIL = GenericTransformersClassifier("VIL_1", "/active_learning_proccess/config/vil_model.json",
                                      "/active_learning_proccess/config/vil.json",
                                      map_labels={0: 'NOVIOLENTO', 1: 'VIOLENTO'})

VIL_grado = GenericTransformersClassifier("VIL_grado_1", "/active_learning_proccess/config/vil_grado_model.json",
                                            "/active_learning_proccess/config/vil_grado.json",
                                            map_labels={1: 'MODERADO',
                                                        0: 'GRAVE'},
                                            is_unlabelled=False)

Comment = GenericTransformersClassifier("Comment_1", "/active_learning_proccess/config/comment_model.json",
                                          "/active_learning_proccess/config/comment.json",
                                          map_labels={0: '1. No tóxico', 1: '2. Ligeramente tóxico',
                                                      2: '3. Tóxico', 3: '4. Muy tóxico'})

Comment_const = GenericTransformersClassifier("Comment_const_1",
                                                "/active_learning_proccess/config/comment_const_model.json",
                                                "/active_learning_proccess/config/comment_const.json",
                                                map_labels={0: 'sí', 1: 'no'},
                                                is_unlabelled=False)

server.add_classifier("twitter_classifier", VIL)
server.add_classifier("twitter_grado_classifier", VIL_grado)
server.add_classifier("toxicidad_classifier", Comment)
server.add_classifier("Constructividad_classifier", Comment_const)
server.add_classifier("Insulto_ann", Insulto_spacy())

# server.add_classifier("spacy_ner", SpacyNerClassifier("es_core_news_sm"))
# server.add_classifier("spacy_pos", SpacyPosClassifier("en_core_web_sm"))
# server.add_classifier("sklearn_sentence", SklearnSentenceClassifier())
# server.add_classifier("jieba", JiebaSegmenter())
# server.add_classifier("stemmer", NltkStemmer())
# server.add_classifier("leven", LevenshteinStringMatcher())
# server.add_classifier("sbert", SbertSentenceClassifier())
# server.add_classifier(
#     "adapter_pos",
#     AdapterSequenceTagger(
#         base_model_name="bert-base-uncased",
#         adapter_name="pos/ldc2012t13@vblagoje",
#         labels=[
#             "ADJ",
#             "ADP",
#             "ADV",
#             "AUX",
#             "CCONJ",
#             "DET",
#             "INTJ",
#             "NOUN",
#             "NUM",
#             "PART",
#             "PRON",
#             "PROPN",
#             "PUNCT",
#             "SCONJ",
#             "SYM",
#             "VERB",
#             "X",
#         ],
#     ),
# )
#
# server.add_classifier(
#     "adapter_sent",
#     AdapterSentenceClassifier(
#         "bert-base-multilingual-uncased",
#         "sentiment/hinglish-twitter-sentiment@nirantk",
#         labels=["negative", "positive"],
#         config="pfeiffer",
#     ),
# )

app = server._app

if __name__ == "__main__":
    server.start(debug=True, port=40022)
