import os

from cassis import *
from ariadne.contrib.inception_util import create_prediction, SENTENCE_TYPE
import os

def write_sentence_documents(sentences, labels_dict_list, spans_list, typesystem, path, custom_layers, labeled=True):
    d = os.getcwd() + typesystem
    with open(os.getcwd() + typesystem, 'rb') as f:
        typesystem = load_typesystem(f)
    cas = Cas(typesystem=typesystem)

    SentenceType = typesystem.get_type("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")

    cas.sofa_string = " ".join(sentences)
    first_ann_labels = labels_dict_list[0]
    first_layer = custom_layers.pop(0)

    begin = 0
    annotations = []
    for sentence, label in zip(sentences, first_ann_labels):
        end = begin + len(sentence)
        cas_sentence = SentenceType(begin=begin, end=end)

        # begin = begin, end = end, value = label, grado = "GRAVE"
        kwargs ={'begin': begin, 'end': end}
        kwargs.update(label)
        custom_feature = typesystem.get_type(first_layer)
        custom_annotation = custom_feature(**kwargs)
        begin = end + 1

        cas.add_annotation(cas_sentence)

        if labeled:
            annotations.append(custom_annotation)
        # cas.add_annotation(sentiment_annotation)

    for custom_layer in custom_layers:
        custom_feature = typesystem.get_type(custom_layer)
        for span_list in spans_list:
            for span in span_list:
                annotations.append(custom_feature(**span))
    cas.add_annotations(annotations)
    cas.to_xmi(path, pretty_print=True)