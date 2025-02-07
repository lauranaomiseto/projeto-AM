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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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
    
    print(f"{soma} de {df.shape[0]} ({(soma/df.shape[0]*100):.2f}%) registros com mais de {p*100}% dos atributos faltantes")
    # display(df[faltantes])

def matriz_correlacao(df, colunas):
    def color_corr(val):    
        color = 'yellow' if (abs(val) > 0.5 and abs(val) != 1) else 'white'
        return 'color: %s' % color

    correlation_matrix = df[colunas].corr()
    styled_corr = correlation_matrix.style.map(color_corr)
    display(styled_corr)

def plot_pca(dados, dim=2, col_classe=None,):
    from sklearn.decomposition import PCA

    dados_np = dados.values

    # aplicação do PCA
    pca = PCA(n_components=dim)
    principal_components = pca.fit_transform(dados_np)

    # criação de um dataframe para os dados transformados
    nomes_col = [f'PC{i+1}' for i in range(dim)]
    pca_df = pd.DataFrame(principal_components, columns=nomes_col)

    # adiciona coluna de classes se especificada
    if col_classe and col_classe in dados.columns:
        pca_df[col_classe] = dados[col_classe].values

    if dim == 3 :
        # cria um plot 3D
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        if col_classe and col_classe in pca_df.columns:
            scatter = ax.scatter(pca_df['PC1'], pca_df['PC2'], pca_df['PC3'], c=pca_df[col_classe], cmap='viridis')
            fig.colorbar(scatter, ax=ax, label=col_classe)
        else:
            ax.scatter(pca_df['PC1'], pca_df['PC2'], pca_df['PC3'])

        ax.set_xlabel('Componente Principal 1')
        ax.set_ylabel('Componente Principal 2')
        ax.set_zlabel('Componente Principal 3')
        plt.title('Visualização 3D PCA')
        plt.show()
    elif dim == 2:
        # cria um plot 2D
        plt.figure(figsize=(10, 7))
        if col_classe and col_classe in pca_df.columns:
            sns.scatterplot(x=pca_df['PC1'], y=pca_df['PC2'], hue=pca_df[col_classe], palette='viridis')
        else:
            sns.scatterplot(x=pca_df['PC1'], y=pca_df['PC2'])

        plt.xlabel('Componente Principal 1')
        plt.ylabel('Componente Principal 2')
        plt.title('Visualização 2D PCA')
        plt.show()

    return pca_df
