import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import re
from small_text.base import LABEL_UNLABELED
from sklearn.model_selection import train_test_split
import os


def de_emojify(text):
    regrex_pattern = re.compile(pattern="["
                                        u"\U0001F600-\U0001F92F"  # emoticons
                                        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                        u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                        u"\U00002702-\U000027B0"
                                        u"\U000024C2-\U0001F251"
                                        u"\U0001F190-\U0001F1FF"
                                        u"\U0001F926-\U0001FA9F"                                        
                                        u"\u2640-\u2642"
                                        u"\u2600-\u2B55"
                                        u"\u200d"
                                        u"\u23cf"
                                        u"\u23e9"
                                        u"\u231a"
                                        u"\ufe0f"                                        
                                        "]+", flags=re.UNICODE)
    return regrex_pattern.sub(r'', text)


def preprocess(value):
    new_value = de_emojify(value)
    new_value = re.sub(r'http\S+', '', new_value)
    return new_value


def convert_insulto_text(insultos):
    text = ''
    for insulto in insultos:
        text += insulto + ', '
    text = text.strip(', ')
    return text


def load_data(file, label, filter_label=None, filter_label_value=None, labels_to_exclude=[],  is_features=False, is_preprocess=False):
    if file.endswith('.tsv'):
        df_in = pd.read_csv(os.getcwd() + file, sep='\t')
    else:
        df_in = pd.read_json(os.getcwd() + file, lines=True)

    for value in labels_to_exclude:
        df_in = df_in[df_in[label] != value]
    label_encoder = None
    if label in df_in.columns:
        if filter_label:
            df_in = df_in[df_in[filter_label] == filter_label_value]
        print(df_in[label].value_counts())
        labels = df_in[label]
        # To labels
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels.values)
    else:
        labels = [LABEL_UNLABELED] * len(df_in)

    ids = df_in['id']
    texts = df_in['text']
    features = [[]] * len(df_in)
    if is_features:
        pass
    if is_preprocess:
        df_in['text'] = df_in['text'].apply(preprocess)
    list_of_tuples = list(zip(list(ids), list(texts), list(labels), list(features)))
    df = pd.DataFrame(list_of_tuples, columns=['id', 'text', 'labels', 'features'])
    return df, label_encoder