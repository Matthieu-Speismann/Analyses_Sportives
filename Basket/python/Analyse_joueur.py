#################### Imports: ####################
from data_analysis_methods import *
import pandas as pd
import pandasql as pdsql
import seaborn as sns
import matplotlib.pyplot as plt

#################### Lecture des données brutes: ####################
path = "Basket/data/Victor_Wenbanyama.csv" 
data = pd.read_csv(path, sep = ';')

#################### Preprocessing: ####################

# Exclure ['Rk', 'GS', 'Date', 'Age', 'Tm'] variables inutles
data = data[data.columns.difference(['Rk', 'GS', 'Date', 'Age', 'Tm'])]

# Requete SQL pour exclure les matchs où Wemby ne jouait pas.
query = """                                                          
    SELECT * FROM data
    WHERE MP > 0
"""
data = pdsql.sqldf(query)

# Variable catégorielle:
data['Away'] = data['Away'].astype('category')

# Definition du data avec que les variables numériques:
data_numerical = data[data.columns.difference(['G', 'Away', 'Opp'])]

#################### Analyse: ####################
plot_selected_pair(data, 'G', 'Win', style = 'bar', var_coloration = 'Away',
                    lignes=[Ligne('y', 0)])

# correlation_plot(data_numerical)