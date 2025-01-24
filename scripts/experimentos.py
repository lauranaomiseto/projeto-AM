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

# Retorna a matriz de confusão
def matriz_confusao(Y_test, Y_pred):
    from sklearn.metrics import confusion_matrix
    
    return confusion_matrix(Y_test, Y_pred)

# Retorna precisao, revocação, fscore separado por classe
def metricas(Y_test, Y_pred, media):
    from sklearn.metrics import precision_recall_fscore_support
    
    return precision_recall_fscore_support(Y_test, Y_pred, average=media)

def acuracia(Y_test, Y_pred):
    from sklearn.metrics import accuracy_score
    
    return accuracy_score(Y_test, Y_pred)