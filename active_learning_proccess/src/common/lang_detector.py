import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector
import pandas as pd
import os

@Language.factory("language_detector")
def get_lang_detector(nlp, name):
   return LanguageDetector()


class Language_detector():
   def __init__(self):
      self.nlp = spacy.load('es_core_news_sm')
      self.nlp.add_pipe('language_detector', last=True)

   def predict(self, sentence):
      return self.nlp(sentence)


df_1 = pd.read_csv(os.getcwd() + "/data/VIL_1/tuits_unlabelled_clean.tsv", sep='\t')
ld = Language_detector()
new_docs = []
for index, row in df_1.iterrows():
   doc = ld.predict(row['text'])
   if doc._.language['language'] == 'es':
      new_docs.append(row)
df_result = pd.DataFrame(new_docs)
df_result.to_csv(os.getcwd() + '/data/VIL_1/tuits_unlabelled_clean_es.tsv', sep='\t', index=False)
print('')
