import numpy as np
from unicodedata import normalize

def infopld_normalize(df):

    submercado = df['Submercado'].apply(word_normalize)
    semana = df["Nº Semana"].astype("category").cat.codes + 1
    
    df= df.rename(columns={"Nº Semana":"semana", "Submercado":'submercado'})\
          .assign(semana=semana)\
          .assign(submercado=submercado)\
          .groupby(["submercado", "semana"],as_index=False).agg(np.mean)\
          .fillna(0)
    
    return df

def word_normalize(word):
    return normalize('NFKD',word).encode('ascii', errors='ignore')\
                                 .decode('utf-8')\
                                 .lower()

def preprocessing_submercado(submercados):
    return {word_normalize(estado):word_normalize(submercado) 
            for submercado in submercados 
            for estado in submercados[submercado]}
