"""Fichier Python contenant des fonctions utiles pour l'analyse de données.
- 
Fonctions implémentées:
    - `standardize`: Normalise un DataFrame donné.
    - `acp_plot`: Affiche graphiquement les résultats de l'Analyse en Composantes Principales (ACP).
    - `correlation_plot`: Affiche graphiquement les valeurs des coefficients de corrélation du DataFrame donné.
    - `plot_selected_pair`: Représente graphiquement une paire de variables sélectionnée d'un DataFrame.
    - `pairs_plot`: Affiche graphiquement les paires de variables d'un DataFrame.
--------------------------------------------------
Python file containing useful functions for data analysis.
-
Implemented functions:
    - `standardize`: Standardizes a given DataFrame.
    - `acp_plot`: Plots the results of a Principal Component Analysis (PCA).
    - `correlation_plot`: Plots the correlation matrix of the variables in the given DataFrame.
    - `plot_selected_pair`: Plots a selected pair of variables from a DataFrame.
    - `pairs_plot`: Plots the pairs of variables in the given DataFrame.
"""

# Importer les bibliothèques nécessaires:
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

class Ligne:
    """
    Représente une ligne à tracer sur un graphique.
    -
    Attributs:
        - xy: `str`: L'axe sur lequel la ligne sera tracée (`'x'` ou `'y'`).
        - value: `float`: La valeur à laquelle la ligne sera tracée.
        - color: `str` optionnel: La couleur de la ligne (par défaut `'black'`).
        - style `str`, optionnel: Le style de la ligne (par défaut `'-'`).
    --------------------------
    Represents a line to be plotted on a graph.
    -
    Attributes:
        - xy: `str`: The axis on which the line will be plotted (`'x'` or `'y'`).
        - value: `float`: The value at which the line will be plotted.
        - color: `str`, optional): The color of the line (default is `'black'`).
        - style: `str`, optional): The style of the line (default is `'-'`).
    """
    
    def __init__(self, xy: 'str', value: float, color: str = 'black', style: str = '-', label: str = None):
        """
        Initialise un objet Ligne.
        -
        Args:
            - xy: `str`: Les coordonnées du point de données.
            - value: `float`: La valeur du point de données.
            - color: `str`, optionnel: La couleur du point de données. Par défaut, `'black'`.
            - style `str`, optionnel: Le style du point de données. Par défaut, `'-'`.
        --------------------------
        Initialize a Ligne object.
        -
        Args:
            - xy: `str`: The coordinates of the data point.
            - value: `float`: The value of the data point.
            - color `str`, optional: The color of the data point. Defaults to `'black'`.
            - style `str`, optional: The style of the data point. Defaults to `'-'`.
        """
        self.xy = xy
        self.value = value
        self.color = color
        self.style = style
        self.label = label

    def plot(self):
        """
        Reproduit une ligne verticale ou horizontale sur le graphique actuel.
        -
        Si `xy` est défini sur 'x', une ligne verticale sera tracée à la valeur spécifiée.
        Si `xy` est défini sur 'y', une ligne horizontale sera tracée à la valeur spécifiée.
        
        Paramètres:
            - self: `object`: L'instance de la classe Ligne.
        --------------------------
        Plot a vertical or horizontal line on the current plot.
        - 
        If `xy` is set to 'x', a vertical line will be plotted at the specified `value`.
        If `xy` is set to 'y', a horizontal line will be plotted at the specified `value`.
        
        Parameters:
            - self: `object`: The instance of the class Ligne.
        """
        if self.xy == 'x':
            plt.axvline(self.value, color=self.color, label= self.label, linewidth=1, linestyle=self.style)
        elif self.xy == 'y':
            plt.axhline(self.value, color=self.color, label= self.label, linewidth=1, linestyle=self.style)


def standardize(data: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise le DataFrame donné.
    -
    Fait en soustrayant la moyenne et en divisant par l'écart-type.
    
    Argument:
        - data: `pd.DataFrame`: Le DataFrame d'entrée à normaliser.
    
    Renvoie:
        - `pd.DataFrame`: Le DataFrame normalisé.
    --------------------------------------------------
    Standardizes the given DataFrame. 
    - 
    Done by subtracting the mean and dividing by the standard deviation.

    Argument:
        - data : `pd.DataFrame`: The input DataFrame to be standardized.

    Returns:
        - `pd.DataFrame`: The standardized DataFrame.
    """
    return (data - data.mean()) / data.std()


def acp_plot(data: pd.DataFrame, titre: str, var_used :list[str], 
                  standardization = True, individus = [], var_coloration:str = None):
    """
    Affiche graphiquement les résultats de l'Analyse en Composantes Principales (ACP).
    -
    Arguments:
        - data: `pd.DataFrame`: Le dataframe contenant les données.
        - titre: `str`: Le titre du graphique.
        - var_used: `list[str]: La liste des variables utilisées pour l'ACP.
        - standardization: `bool`, optional: Indique si les données doivent être standardisées. 
            Par défaut: `True`.
        - individus `list[str]`, optional: La liste des noms des individus à afficher. Par défaut, `[]`.
        - var_coloration `str`, optional: Le nom de la variable utilisée pour la coloration des points. 
            Par défaut, `None`.
    --------------------------------------------------
    Plot the results of a Principal Component Analysis (PCA).
    -
    Arguments:
        - data: `pd.DataFrame`: The dataframe containing the data.
        - titre: `str`: The title of the plot.
        - var_used: `list[str]`: The list of variables used for the PCA.
        - standardization `bool`, optional: Indicates if the data should be standardized. 
            By default: `True`.
        - individus `list[str]`, optional): The list of the names of the individuals to display. By default, `[]`.
        - var_coloration `str`, optional: The name of the variable used for the coloration of the points. 
            By default, `None`.
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


def correlation_plot(data: pd.DataFrame, title: str = None):
    """
    Affiche graphiquement les valeurs des coefficients de corrélation du DataFrame donné.
    -
    Arguments:
        - data: `pd.DataFrame`: Le DataFrame contenant les données.
        - title: `str`: Le titre du graphique. Par défaut, `None`.
    --------------------------------------------------
    Plot the correlation matrix of the variables in the given DataFrame.
    -
    Arguments:
        - data: `pd.DataFrame`: The DataFrame containing the data.
        - title: `str`: The title of the plot. By default, `None`.
    """
    correlations = data.corr()
    plt.figure()
    
    # Heatmap des coefficients de corrélation:
    heatmap = sns.heatmap(correlations, annot=True, annot_kws={"fontsize":6}, fmt='.2f', cmap='coolwarm', cbar=True, square=True, linewidths=0.5, linecolor='black')
    
    # Titre ou non:
    if title is not None:
        plt.title(title)

    # Gestion de l'axe x:
    heatmap.set_xticks(np.arange(len(correlations.columns)) + 0.5, minor=False)
    heatmap.set_xticklabels(correlations.columns, rotation = 90, ha = 'right', fontsize=8)
    
    # Gestions de l'axe y:
    heatmap.set_yticks(np.arange(len(correlations.columns)) + 0.5, minor=False)
    heatmap.set_yticklabels(correlations.columns, rotation = 0, ha = 'right', fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_selected_pair(data: pd.DataFrame, var_x: str, var_y: str, style: str = 'scatter', color: str = 'blue',
                        var_coloration: str = None, title: str = None, lignes: list[Ligne] = []):
    """
    Represente graphiquement une paire de variables sélectionnée d'un DataFrame.
    -
    Paramètres:
        - data: `pd.DataFrame`: Le DataFrame contenant les données.
        - var_x: `str`: Le nom de la première variable à représenter.
        - var_y: `str`: Le nom de la deuxième variable à représenter.
        - style: `str`, optionnel: Le style du graphique: `'scatter'` ou `'bar'`. Par défaut, `'scatter'`.
        - color: `str`, optionnel: La couleur du graphique. Utile que si `var_coloration` vaut `None`. Par défaut, `'blue'`.
        - var_coloration: `str`, optionnel: La variable utilisée pour la coloration du graphique. Par défaut, `None`.
        - title: `str`, optionnel: Le titre du graphique. Par défaut, `None`.
        - lignes: `list[Ligne]`, optionnel: Une liste de lignes à ajouter au graphique. Par défaut, une liste vide `[]`.
    --------------------------------------------------
    Plot a selected pair of variables from a DataFrame.
    -
    Parameters:
    - data: `pd.DataFrame`: The DataFrame containing the data.
    - var_x: `str`: The name of the first variable to plot.
    - var_y: `str`: The name of the second variable to plot.
    - style: `str`, optional: The style of the plot: `'scatter'` or `'bar'`. Default is `'scatter'`.
    - color: `str`, optional: The color of the plot. Default is `'blue'`.
    - var_coloration: `str`, optional: The variable used for color coding the plot. Default is `None`.
    - title: `str`, optional: The title of the plot. Default is `None`.
    - lignes: `list[Ligne]`, optional: A list of lines to be added to the plot. Default is an empty list `[]`.
    """
    x = list(data[var_x])
    y = list(data[var_y])

    plt.figure()
    
    # Ajout de ligne si nécessaire:
    for ligne in lignes:
        ligne.plot()
    
    # Ajout de titre et labels:    
    if title is not None:
        plt.title(title)

    plt.xlabel(var_x)
    plt.ylabel(var_y)

    # Plot des données, selon le style scatter:
    if style == 'scatter':
        if var_coloration is None:
            plt.scatter(x, y, marker = 'o', color = color)

        # Gestion de la coloration catégorielle:
        if var_coloration is not None:
            scatter = plt.scatter(x, y, marker = 'o', c = data[var_coloration].cat.codes, cmap='plasma')
            plt.legend(handles=scatter.legend_elements()[0], labels=sorted(data[var_coloration].unique()), title=var_coloration)
    
    # Plot des données, selon le style bar:
    elif style == 'bar':
        """bars = plt.bar(x,y, color = color)
        
        # Gestion de la coloration catégorielle:
        unique_categories = sorted(data[var_coloration].unique())
        num_categories = len(unique_categories)
        color_map = plt.cm.get_cmap('plasma', num_categories)
        if var_coloration is not None:
            for i, bar in enumerate(bars):
                bar.set_color(color_map(data[var_coloration].cat.codes[i] / num_categories))  
            plt.legend(handles = bars, labels = unique_categories, title=var_coloration)

        # Affichage au dessus des barres:
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom', fontsize = 6)"""

        if var_coloration is None:
            bars = sns.barplot(x=var_x, y=var_y, data=data, color = color)
        elif var_coloration is not None:
            bars = sns.barplot(x=var_x, y=var_y, data=data, hue=var_coloration, palette='Set1')
            
         # Affichage au dessus des barres:
        for bar in bars.patches:
            bars.annotate(f'{bar.get_height():.0f}', (bar.get_x() + bar.get_width() / 2., bar.get_height()),
                        ha='center', va='center', xytext=(0, 6), fontsize = 8, textcoords='offset points')

    plt.show()


def pairs_plot(data):
    """
    Affiche graphiquement les paires de variables d'un DataFrame. 
    -
    Affiche graphiquement toutes les paires du DataFrame donné, ainsi que le coefficient de corrélation.
    
    Arguments:
        - data: `pd.DataFrame`: Le DataFrame contenant les données.
    --------------------------------------------------
    Plot the pairs of variables in the given DataFrame.
    -
    Plot the pairs of variables in the given DataFrame 
    and the correlation coefficient.
    Arguments:
        - data: `pd.DataFrame`: The DataFrame containing the data.
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
