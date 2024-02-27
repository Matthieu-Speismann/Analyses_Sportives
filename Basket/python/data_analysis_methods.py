# Importer les bibliothèques nécessaires:
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


# Standardiser les données:
def standardize(data: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise le DataFrame donné.
    -
    Fait en soustrayant la moyenne et en divisant par l'écart-type.
    
    Argument:
        - data (pd.DataFrame): Le DataFrame d'entrée à normaliser.
    
    Renvoie:
        - pd.DataFrame: Le DataFrame normalisé.
    --------------------------------------------------
    Standardizes the given DataFrame. 
    - 
    Done by subtracting the mean and dividing by the standard deviation.

    Argument:
        - data (pd.DataFrame): The input DataFrame to be standardized.

    Returns:
        - pd.DataFrame: The standardized DataFrame.
    """
    return (data - data.mean()) / data.std()


def acp_plot(data: pd.DataFrame, titre: str, var_used :list[str], 
                  standardization = True, individus = [], var_coloration:str = None):
    """
    Affiche graphiquement les résultats de l'Analyse en Composantes Principales (ACP).
    -
    Arguments:
        - data (pd.DataFrame): Le dataframe contenant les données.
        - titre (str): Le titre du graphique.
        - var_used (list[str]): La liste des variables utilisées pour l'ACP.
        - standardization (bool, optional): Indique si les données doivent être standardisées. 
            Par défaut, True.
        - individus (list, optional): La liste des noms des individus à afficher. Par défaut, [].
        - var_coloration (str, optional): Le nom de la variable utilisée pour la coloration des points. 
            Par défaut, None.
    --------------------------------------------------
    Plot the results of a Principal Component Analysis (PCA).
    -
    Arguments:
        - data (pd.DataFrame): The dataframe containing the data.
        - titre (str): The title of the plot.
        - var_used (list[str]): The list of variables used for the PCA.
        - standardization (bool, optional): Indicates if the data should be standardized. 
            By default, True.
        - individus (list, optional): The list of the names of the individuals to display. By default, [].
        - var_coloration (str, optional): The name of the variable used for the coloration of the points. 
            By default, None.
    """
    pca = PCA()
    
    data_used = data.loc[:, var_used]
    
    if standardization == True:
        data_standardized = standardize(data_used)
        acp_result = pca.fit_transform(data_standardized)
    else:
        acp_result = pca.fit_transform(data_used)

    # Visualiser les résultats avec un biplot et les noms des individus
    plt.figure(1)
    
    if var_coloration == None:
        plt.scatter(acp_result[:, 0], acp_result[:, 1])
    else:
        #Colorier selon la variable de coloration:
        scatter = plt.scatter(acp_result[:, 0], acp_result[:, 1], c = data[var_coloration].cat.codes, cmap='turbo')
        # Ajouter une légende de couleur:
        plt.legend(handles=scatter.legend_elements()[0], labels=sorted(data[var_coloration].unique()), title=var_coloration)

        
    # Ajouter les noms des individus en label
    for i, txt in enumerate(individus):
        plt.annotate(txt, (acp_result[i, 0], acp_result[i, 1]), fontsize = 8)

    # Afficher les pourcentages d'information sur chaque axe:
    var_ratios = pca.explained_variance_ratio_

    # Ajouter des étiquettes aux axes:
    plt.title(titre)
    plt.xlabel(f"Axe 1 - {var_ratios[0]* 100:.2f}%")
    plt.ylabel(f"Axe 2 - {var_ratios[1]* 100:.2f}%")
    plt.grid(True)
    
    plt.figure(2)
    # Ajouter les vecteurs des variables dans l'espace des composantes principales (biplot)
    plt.title('Contribution des variables sur les axes')
    for i, (comp1, comp2) in enumerate(zip(pca.components_[0, :], pca.components_[1, :])):
        plt.arrow(0, 0, comp1, comp2, color='r', alpha=1)
        plt.text(comp1, comp2, data_used.columns[i], color='r', ha='right', va='bottom', fontsize=8)
    
    plt.show()


def correlation_plot(data: pd.DataFrame, titre: str = None):
    """
    Affiche graphiquement les valeurs des coefficients de corrélation du DataFrame donné.
    -
    Arguments:
        - data (pd.DataFrame): Le DataFrame contenant les données.
        - titre (str): Le titre du graphique.
    --------------------------------------------------
    Plot the correlation matrix of the variables in the given DataFrame.
    -
    Arguments:
        - data (pd.DataFrame): The DataFrame containing the data.
        - titre (str): The title of the plot.
    """
    correlations = data.corr()
    plt.figure()
    
    # Heatmap des coefficients de corrélation:
    heatmap = sns.heatmap(correlations, annot=True, annot_kws={"fontsize":6}, fmt='.2f', cmap='coolwarm', cbar=True, square=True, linewidths=0.5, linecolor='black')
    
    # Titre ou non:
    if titre is not None:
        plt.title(titre)

    # Gestion de l'axe x:
    heatmap.set_xticks(np.arange(len(correlations.columns)) + 0.5, minor=False)
    heatmap.set_xticklabels(correlations.columns, rotation = 90, ha = 'right', fontsize=8)
    
    # Gestions de l'axe y:
    heatmap.set_yticks(np.arange(len(correlations.columns)) + 0.5, minor=False)
    heatmap.set_yticklabels(correlations.columns, rotation = 0, ha = 'right', fontsize=8)

    plt.tight_layout()
    plt.show()


def pairs_plot(data):
    """
    Affiche graphiquement les paires de variables du DataFrame donné, ainsi que le coefficient de corrélation.
    -
    Arguments:
        - data (pd.DataFrame): Le DataFrame contenant les données.
    --------------------------------------------------
    Plot the pairs of variables in the given DataFrame, and the correlation coefficient.
    -
    Arguments:
        - data (pd.DataFrame): The DataFrame containing the data.
    """
    
    # Coefficient de corrélation (représentation matricielle):
    cor_matrix = data.corr().values

    # Graphe des paires de variables:
    pair_plot = sns.pairplot(data)
    cor_matrix = data.corr().values

    # Ajouter les coefficients de corrélation sur chaque subplot hors diagonale
    for i, (ax, corr) in enumerate(zip(pair_plot.axes.flat, cor_matrix.flatten())):
        if i % (len(data.columns) + 1) != 0:  # Ne pas ajouter sur la diagonale
            ax.annotate(f"Corr: {corr:.2f}", xy=(0.8, 0.95), xycoords='axes fraction', ha='center', va='center', fontsize=8)
    plt.tight_layout()

    plt.show()

if __name__ == '__main__':
    
    # Générer des données d'exemple
    np.random.seed(123)
    data = pd.DataFrame({
        'variable1': np.random.randn(100),
        'variable2': np.random.randn(100),
        'variable3': np.random.randn(100)
    })
    
    # Effectuer l'ACP avec scikit-learn
    pca = PCA()
    
    # Afficher les résultats
    "print(pd.DataFrame(acp_result, columns=[f'PC{i+1}' for i in range(data.shape[1])]))"
    
    acp_plot(data, 'Test')