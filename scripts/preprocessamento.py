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

# biblioteca usada para trabalhar com vetores e matrizes
import numpy as np 

# biblioteca usada para trabalhar com dataframes e análise de dados
import pandas as pd

# bibliotecas usadas para geração de graficos
import seaborn as sns
import matplotlib.pyplot as plt


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

def preencher_faltantes(df, estrategia) :
    from sklearn.impute import SimpleImputer
    
    imputer = SimpleImputer(strategy=estrategia)
    df = imputer.fit_transform(df)

    return df

def preencher_teste(df, valores) :
    from sklearn.impute import SimpleImputer
    
    for valor in valores:
        print(valor)
        imputer_const = SimpleImputer(strategy='constant', fill_value=valor)
        df = imputer_const.fit_transform(df)        
    
    return df

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

def identificar_outlier(coluna):
    # considera medida de dispersão
    Q1 = coluna.quantile(0.25)
    Q3 = coluna.quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    print(f"Limite inf: {limite_inferior}\nLimite sup: {limite_superior}")
    return coluna[(coluna < limite_inferior) | (coluna > limite_superior)]

def identificar_outlier_percentil(coluna, p_inf, p_sup):
    # exluir os menores que p_inf e os maiores que p_sup
    # não considera dispersão, ilda apenas com as extremidades
    limite_inferior = np.percentile(coluna, p_inf*100)
    limite_superior = np.percentile(coluna, p_sup*100)
    print(f"Limite inf: {limite_inferior}\nLimite sup: {limite_superior}")
    return coluna[(coluna < limite_inferior) | (coluna > limite_superior)]

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

def invalidar_outliers_percentil(df, colunas, p_inf, p_sup):
    # marca outliers como NaN
    df_copy = df.copy()
    for coluna in colunas:
        limite_inferior = np.percentile(df_copy[coluna], p_inf*100)
        limite_superior = np.percentile(df_copy[coluna], p_sup*100)
        df_copy.loc[~df_copy[coluna].between(limite_inferior, limite_superior), coluna] = np.nan
    return df_copy


# CONVERTER SIMBÓLICO-NUMÉRICO

def converter_ordinais(df, colunas, categorias) :  
    from sklearn.preprocessing import OrdinalEncoder
    
    df_copy = df.copy()

    for coluna, categoria in zip(colunas, categorias):
        encoder = OrdinalEncoder(categories=[categoria])  # Ordem explícita
        df_copy[coluna] = encoder.fit_transform(df_copy[[coluna]])

    return df_copy

def converter_nominais(df, colunas) :
    from sklearn.preprocessing import OneHotEncoder
    
    df_copy = df.copy()

    for coluna in colunas:
        encoder = OneHotEncoder(sparse_output=False)  # Para evitar matriz esparsa
        # Use duplos colchetes para manter o formato bidimensional
        encoded = encoder.fit_transform(df_copy[[coluna]])
        # Transformar em DataFrame e concatenar ao original
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([coluna]), index=df_copy.index)
        df_copy = pd.concat([df_copy.drop(columns=[coluna]), encoded_df], axis=1)

    return df_copy

def codificador_converter_nominais(df, colunas):
    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False)
    return encoder.fit(df[colunas])


# NORMALIZAR ATRIBUTOS

def normalizar(df, treino) :
    from sklearn.preprocessing import StandardScaler
    df_copy = df.copy()
    
    scaler = StandardScaler()
    if(treino) :
        df_copy.iloc[:,1:-1] = scaler.fit_transform(df_copy.iloc[:,1:-1])
    else :
        df_copy.iloc[:,1:] = scaler.fit_transform(df_copy.iloc[:,1:])

    return df_copy