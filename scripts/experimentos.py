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

# Arquivo com todas as funcoes e codigos referentes aos experimentos


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split


def plot_pca(dados, dim=2, col_classe=None,):

    dados_np = dados.values
    
    # Aplicação do PCA
    pca = PCA(n_components=dim)
    principal_components = pca.fit_transform(dados_np)
    
    # Criação de um dataframe para os dados transformados
    nomes_col = [f'PC{i+1}' for i in range(dim)]
    pca_df = pd.DataFrame(principal_components, columns=nomes_col)
    
    # Adiciona coluna de classes se especificada
    if col_classe and col_classe in dados.columns:
        pca_df[col_classe] = dados[col_classe].values
    
    if dim == 3 :
    # Cria um plot 3D
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
        # Cria um plot 2D
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

def avaliar_modelos(models, param_grids, X_train, y_train, X_val, y_val):
    results = {}

    # Configurando o K-Folds
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        print(f"Avaliando {name}...")
        
        # Ajuste de hiperparâmetros, se aplicável
        if name in param_grids and param_grids[name]:
            grid_search = GridSearchCV(model, param_grid=param_grids[name], cv=kfold, scoring="accuracy", n_jobs=-1)
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            print(f"Melhores parâmetros para {name}: {grid_search.best_params_}")
        else:
            best_model = model
            best_model.fit(X_train, y_train)
        
        # Curvas de aprendizado
        train_sizes = np.linspace(0.1, 0.9, 10)
        train_errors = []
        val_errors = []
        
        for train_size in train_sizes:
            X_partial, _, y_partial, _ = train_test_split(X_train, y_train, train_size=train_size, random_state=42)
            best_model.fit(X_partial, y_partial)
            train_errors.append(1 - best_model.score(X_partial, y_partial))
            val_errors.append(1 - best_model.score(X_val, y_val))
        
        plt.figure()
        plt.plot(train_sizes, train_errors, label="Treino", marker='o')
        plt.plot(train_sizes, val_errors, label="Validação", marker='s')
        plt.xlabel("Tamanho do Treinamento")
        plt.ylabel("Erro")
        plt.title(f"Curva de Aprendizado - {name}")
        plt.legend()
        plt.show()
        
        # Predição
        y_pred = best_model.predict(X_val)

        # Cálculo da AUC-ROC
        if hasattr(best_model, "predict_proba"):
            y_proba = best_model.predict_proba(X_val)[:, 1]
            auc_roc = roc_auc_score(y_val, y_proba)
        else:
            auc_roc = None

        # Métricas
        acc = accuracy_score(y_val, y_pred)
        conf_matrix = confusion_matrix(y_val, y_pred)
        class_report = classification_report(y_val, y_pred)

        results[name] = {
            "Accuracy": acc,
            "AUC-ROC": auc_roc,
            "Confusion Matrix": conf_matrix,
            "Classification Report": class_report
        }

        print(f"Accuracy: {acc:.4f}")
        print(f"AUC-ROC: {auc_roc:.4f}" if auc_roc is not None else "AUC-ROC: Não disponível")
        print("Confusion Matrix:")
        print(conf_matrix)
        print("Classification Report:")
        print(class_report)

    return results

