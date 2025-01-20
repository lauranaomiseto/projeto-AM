# ################################################################
# PROJETO FINAL
#
# Universidade Federal de Sao Carlos (UFSCAR)
# Departamento de Computacao - Sorocaba (DComp-So)
# Disciplina: Aprendizado de Maquina
# Prof. Tiago A. Almeida
#
#
# Nome: Beatriz Rogers Tripoli Barbosa, Laura Naomi Seto
# RA: 792170, 813210 
# ################################################################

# Arquivo com todas as funcoes e codigos referentes ao preprocessamento

import numpy as np 
import pandas as pd


# IDENTIFICAR VALORES FALTANTES E INVÁLIDOS
def invalidar_nao_numerico(df, colunas):
    return df[colunas].apply(pd.to_numeric, errors='coerce')

def padronizar_string(df, colunas):
    return df[colunas].apply(lambda col: col.str.lower() if col.dtype == 'object' else col)

def invalidar_nao_temporais(df, colunas):
    return df[colunas].apply(pd.to_datetime, errors='coerce', format='%d/%m/%y')


# EXCLUIR REGISTROS COM MENOS DE 50% DOS VALORES PREENCHIDOS 
def remover_registros_incompletos(df, p):
    faltantes = df.isnull().sum(axis=1) > (df.shape[1]*p)
    return df[~faltantes]


# IMPUTADORES
def imputador_faltantes_media(df, colunas):
    from sklearn.impute import SimpleImputer
    imputador = SimpleImputer(strategy='mean')

    return imputador.fit(df[colunas])
    

def imputador_faltantes_moda(df, colunas):
    from sklearn.impute import SimpleImputer
    imputador = SimpleImputer(strategy='most_frequent')

    return imputador.fit(df[colunas])

def imputador_faltantes_knn(df, colunas, k):
    from sklearn.impute import KNNImputer
    imputador = KNNImputer(n_neighbors=k)

    return imputador.fit(df[colunas])


# EXLUIR DUPLICATAS (mesmos valores de atributos e mesma classifiação)
def remover_duplicatas(df):
    # ignora o índice 
    return df.drop_duplicates(subset=df.columns[1:], keep='first')


# EXCLUIR INCONSISTÊNCIAS (mesmos valores de atributos e classificação diferente)
def remover_inconsistencia(df):
    # ignora o índice e a classe (atributo alvo)
    return df.drop_duplicates(subset=df.columns[1:-1], keep=False)


# IDENTIFICAR OUTLIERS SEM CONSIDERAR RELAÇÃO ENTRE ATRIBUTOS
# identificação de outliers por atributo 
def invalidar_outliers_quartil(df, colunas):
    # marca outliers como NaN
    df_copy = df.copy()
    for coluna in colunas:
        Q1 = df_copy[coluna].quantile(0.25)
        Q3 = df_copy[coluna].quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        df_copy.loc[~df[coluna].between(limite_inferior, limite_superior), coluna] = np.nan
    return df_copy


# CONVERTER SIMBÓLICO-NUMÉRICO
def codificador_nominais(df, colunas):
    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False)
    return encoder.fit(df[colunas])

def codificador_ordinais(df, colunas, categorias):
    from sklearn.preprocessing import OrdinalEncoder
    encoder = OrdinalEncoder(categories=categorias)
    return encoder.fit(df[colunas])


# NORMALIZAR ATRIBUTOS
def normalizador(df, colunas):
    from sklearn.preprocessing import StandardScaler

    scalar = StandardScaler()
    return scalar.fit(df[colunas])