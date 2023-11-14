import numpy as np
import pandas as pd
import os

words = ["#anal",
         "#culazo",
         "#gay",
         "#verga",
         "#cogiendo",
         "#Fucking",
         "#Sexywife",
         "#Cuckold",
         "#hotwife",
         "#putiesposa",
         "#boobs",
         "#xxx",
         "#sexy",
         "#tetas",
         "#bisex",
         "#lesbian",
         "#cumslut",
         "#bigass",
         "#nalgona",
         "#salvadorea",
         "#swinger",
         "#culo",
         "#tetazo",
         "#putiesposa",
         "#putipobres",
         "#porn",
         "#putiselfie",
         "#bdsm",
         "#ass",
         "#dick",
         "#crossdresser",
         "#Scort",
         "#Cornudo",
         "#Homemade",
         "#parejahot",
         "#Ramera",
         "#piruja",
         "@Sexy_Girl_Hot_X",
         "#SexoAnal",
         "YEGUA",
         "puta",
         "#travesti",
         "#crossdres",
         "#sissy",
         "#maricon"]

df = pd.read_csv(os.getcwd() + "/data/VIL_1/tuiter_maria.csv")
df_bads = []
df.drop(columns=["agreement","score","votes","label"], inplace=True)
df.dropna(subset='text', inplace=True)
for word in words:
    df_bads.append(df.loc[df['text'].str.contains(word, na=False)])
df_bad = pd.concat(df_bads)
vv = df_bad['text'].duplicated()
df_bad.drop_duplicates(subset="text", inplace=True)

df.drop(axis=0, index=df_bad.index, inplace=True)
# df.drop(axis=1, columns=['Unnamed: 0'], inplace=True)
df.to_csv(os.getcwd() + '/data/VIL_1/tuits_maria_clean_V1.tsv', sep='\t')
print('')
