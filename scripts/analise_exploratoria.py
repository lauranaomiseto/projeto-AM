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

# Arquivo com todas as funcoes e codigos referentes a analise exploratoria
import pandas as pd

# ANÁLISE DE VALORES FALTANTES
def analise_valores_faltantes(df):
    missing_data= df.apply(lambda x: pd.Series([x.isnull().sum(), x.isnull().mean()*100], index=['qtd', '%']))

    print(f"Relação de atributos e porcentagem de nulos: ")
    display(missing_data.T)

def analise_registro_adulto(df):
    adult_data = df['IDADE'] > 18 
    print(f"Número e porcentagem de registros adultos: ")
    display(pd.Series([adult_data.sum(), adult_data.mean()*100], index=['qtd', '%']))

def analise_registro_incompleto(df, p):
    # identificar as linhas com mais de p% não preenchido
    faltantes = df.isnull().sum(axis=1) > (df.shape[1]*p)
    soma = faltantes.sum()
    
    print(f"{soma} de {df.shape[0]} ({soma/df.shape[0]*100}%) registros com mais de {p*100}% dos atributos faltantes")
    # display(df[faltantes])