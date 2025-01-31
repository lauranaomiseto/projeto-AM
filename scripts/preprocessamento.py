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
import unicodedata

col_numerica = ['IMC', 'Peso', 'Altura', 'IDADE', 'PA SISTOLICA', 'PA DIASTOLICA', 'FC']   
col_categorica = ['PULSOS', 'B2', 'SOPRO', 'HDA 1', 'SEXO', 'MOTIVO1']

col_nominal = ['PULSOS', 'B2', 'SOPRO', 'SEXO']
col_ordinal = ['HDA 1', 'MOTIVO1']

motivo1_categorias = [
    'check-up',
    'outro',
    'suspeita de cardiopatia',
    'parecer cardiologico',
    'cardiopatia ja estabelecida'
]

sexo_categorias = [
    'f',
    'm',
    'outro'
]

hda1_categorias = [
    'assintomatico', 
    'ganho de peso', 
    'palpitacao', 
    'dor precordial', 
    'dispneia',
    'desmaio/tontura',
    'cianose',
    'outro'
]

sopro_categorias = [
    'ausente',
    'sistolico',
    'continuo',
    'diastolico',
    'sistolico e diastolico'
]

b2_categorias = [
    'normal',
    'hiperfonetica',
    'desdob fixo',
    'outro',
    'unica'
]

pulsos_categorias = [
    'normais',
    'amplos',
    'outro',
    'femorais diminuidos',
    'diminuidos'
]

col_ordinal_categorias = [hda1_categorias, motivo1_categorias]


# IDENTIFICAR VALORES FALTANTES E INVÁLIDOS

def invalidar_nao_numerico(series):
        return pd.to_numeric(series, errors='coerce')

def normalizar_strings(series):
        return (series.astype(str)
            .apply(lambda x: unicodedata.normalize('NFKD', x).encode('ascii', errors='ignore').decode('ascii')).str.lower().str.replace(r'^\d+ - ', '', regex=True))

def identificar_invalidos(df, teste):
    df_copy = df.copy()
    cols = df_copy.columns

    if 'IMC' in cols:
        df_copy['IMC'] = invalidar_nao_numerico(df_copy['IMC'])

    if 'Peso' in cols:
        df_copy['Peso'] = invalidar_nao_numerico(df_copy['Peso'])
        if teste:
            df_copy['Peso'] = df_copy['Peso'].apply(lambda x: x if 0<x else np.nan)
        else:
            df_copy = df_copy.loc[df_copy['Peso'] > 0]

    if 'Altura' in cols: 
        df_copy['Altura'] = invalidar_nao_numerico(df_copy['Altura'])
        if teste:
            df_copy['Altura'] = df_copy['Altura'].apply(lambda x: x if 0<x else np.nan)
        else:
            df_copy = df_copy.loc[df_copy['Altura'] > 0]

    if 'IDADE' in cols:
        df_copy['IDADE'] = invalidar_nao_numerico(df_copy['IDADE'])
        df_copy['IDADE'] = df_copy['IDADE'].apply(lambda x: x if 0<x else np.nan)
        # df_copy = df_copy.loc[df_copy['IDADE'] > 0]

    if 'PA SISTOLICA' in cols:
        df_copy['PA SISTOLICA'] = df_copy['PA SISTOLICA']       

    if 'PA DIASTOLICA' in cols:
        df_copy['PA DIASTOLICA'] = df_copy['PA DIASTOLICA']

    if 'FC' in cols:
        df_copy['FC'] = invalidar_nao_numerico(df_copy['FC'])


    if 'PULSOS' in cols:
        df_copy['PULSOS'] = normalizar_strings(df_copy['PULSOS'])
        df_copy['PULSOS'] = df_copy['PULSOS'].apply(lambda x: x if x in pulsos_categorias else np.nan)

    if 'B2' in cols:
        df_copy['B2'] = normalizar_strings(df_copy['B2'])
        df_copy['B2'] = df_copy['B2'].apply(lambda x: x if x in b2_categorias else np.nan)

    if 'SOPRO' in cols:
        df_copy['SOPRO'] = normalizar_strings(df_copy['SOPRO'])
        df_copy['SOPRO'] = df_copy['SOPRO'].apply(lambda x: x if x in sopro_categorias else np.nan)


    if 'HDA 1' in cols:
        df_copy['HDA 1'] = normalizar_strings(df_copy['HDA 1'])
        df_copy['HDA 1'] = df_copy['HDA 1'].apply(lambda x: x if x in hda1_categorias else np.nan)

    if 'SEXO' in cols:
        df_copy['SEXO'] = normalizar_strings(df_copy['SEXO'])
        df_copy['SEXO'] = df_copy['SEXO'].apply(lambda x: 'f' if x in ['f', 'feminino'] else 'm' if x in ['m', 'masculino'] else np.nan)

            
    if 'MOTIVO1' in cols:
        df_copy['MOTIVO1'] = normalizar_strings(df_copy['MOTIVO1'])
        df_copy['MOTIVO1'] = df_copy['MOTIVO1'].apply(lambda x: x if x in motivo1_categorias else np.nan)

    if 'CLASSE' in cols:
        df_copy['CLASSE'] = normalizar_strings(df_copy['CLASSE'])
        df_copy['CLASSE'] = df_copy['CLASSE'].apply(lambda x: 'normal' if x in ['normal', 'normais'] else 'anormal' if x in ['anormal'] else np.nan)

    return df_copy


# EXCLUIR REGISTROS ADULTOS
def remover_registros_adultos(df):
    adultos = df['IDADE'] > 19
    return df[~adultos]

# EXCLUIR REGISTROS COM MENOS DE 50% DOS VALORES PREENCHIDOS 
def remover_registros_incompletos(df, p):
    faltantes = df.isnull().sum(axis=1) > (df.shape[1]*p)
    return df[~faltantes]


# IMPUTADORES
def imputador_faltantes_media(df, colunas):
    from sklearn.impute import SimpleImputer
    imputador = SimpleImputer(strategy='mean')

    return imputador.fit(df[colunas])

def imputador_faltantes_mediana(df, colunas):
    from sklearn.impute import SimpleImputer
    imputador = SimpleImputer(strategy='median')

    return imputador.fit(df[colunas])
    
def imputador_faltantes_knn(df, colunas, k):
    from sklearn.impute import KNNImputer
    imputador = KNNImputer(n_neighbors=k)

    return imputador.fit(df[colunas])

def imputador_faltantes_moda(df, colunas):
    from sklearn.impute import SimpleImputer
    imputador = SimpleImputer(strategy='most_frequent')

    return imputador.fit(df[colunas])

def imputador_faltantes_constante(df, colunas):
    from sklearn.impute import SimpleImputer
    imputador = SimpleImputer(strategy='constant', fill_value='outro')

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
def invalidar_outliers_quartil(df, colunas):
    # marca outliers como NaN
    classes = df['CLASSE'].unique()
    df_copy = df.copy()
    for coluna in colunas:
        for classe in classes:
            subset = df_copy[df_copy['CLASSE'] == classe]
            Q1 = subset[coluna].quantile(0.25)
            Q3 = subset[coluna].quantile(0.75)
            IQR = Q3 - Q1
            limite_inferior = Q1 - 1.5 * IQR
            limite_superior = Q3 + 1.5 * IQR
            mask_outliers = ~subset[coluna].between(limite_inferior, limite_superior)
            df_copy.loc[subset[mask_outliers].index, coluna] = np.nan
    return df_copy

# CRIAR ATRIBUTOS NOVOS
def criar_novos_atributos(df):    
    df_copy = df.copy()
    
    # categorias para FC baseadas na idade
    def categorizar_fc(fc, idade):
        if idade <= 1:
            return 'bradicardia' if fc < 100 else 'taquicardia' if fc > 180 else 'normal'
        elif idade <= 5:
            return 'bradicardia' if fc < 80 else 'taquicardia' if fc > 140 else 'normal'
        elif idade <= 12:
            return 'bradicardia' if fc < 70 else 'taquicardia' if fc > 120 else 'normal'
        else:
            return 'bradicardia' if fc < 60 else 'taquicardia' if fc > 100 else 'normal'
    
    df_copy['FC_Categoria'] = df_copy.apply(lambda row: categorizar_fc(row['FC'], row['IDADE']), axis=1)

    return df_copy


# CONVERTER SIMBÓLICO-NUMÉRICO
def codificador_nominais(df, colunas):
    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
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