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

# ANÁLISE DE VALORES FALTANTES
def analise_valores_faltantes(df):
    missing_data = df.isnull().mean() * 100  # porcentagem de valores faltantes
    print(f"Relação de atributos e porcentagem de nulos: ")
    print(missing_data)

def analise_registro_incompleto(df, p):
    # identificar as linhas com mais de p% não preenchido
    faltantes = df.isnull().sum(axis=1) > (df.shape[1]*p)
    soma = faltantes.sum()
    
    print(f"{soma} de {df.shape[0]} ({soma/df.shape[0]*100}%) registros com mais de {p*100}% dos atributos não preenchidos")
    # display(df[faltantes])
    
    return faltantes
