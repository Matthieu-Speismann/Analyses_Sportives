#################### Imports: ####################
from data_analysis_methods import *
import pandas as pd
import pandasql as pdsql
import seaborn as sns
import matplotlib.pyplot as plt

#################### Lecture des données brutes: ####################
path = "Analyses_Sportives/Basket/data/NBA Player Stats 2022-2023 - Regular.csv"
data = pd.read_csv(path, sep = ';')

#################### Preprocessing: ####################

# Exclure 'Rk'
data = data[data.columns.difference(['Rk'])]
     
# Requete SQL pour sélectionner les individus, prenant en compte les transfert en cours d'année:   
query_anti_doublon = """
    SELECT * FROM data
    WHERE
        Team = 'TOT' AND Player IN (
            SELECT Player FROM data
            GROUP BY Player
            HAVING COUNT(DISTINCT TEAM) > 1)
        OR Player IN (
            SELECT Player FROM data
            GROUP BY Player
            HAVING COUNT(DISTINCT TEAM) = 1)
"""
data = pdsql.sqldf(query_anti_doublon)

# Liste des individus:
individus = data['Player']

# Data numérique:
data_numeric = data[data.columns.difference(['Player', 'Pos', 'Team'])]

#################### Analyse: ####################

# Requête SQL pour sélectionner les joueurs selon conditions d'étude:
query: str = """
    SELECT * FROM data
"""
data = pdsql.sqldf(query)
# Variable Catégorielle (à faire après chaque redéfinition de data):
data['Pos'] = data['Pos'].astype('category')

# Etude de la corrélation:
correlation_plot(data_numeric, "Saison 2022-2023")

# Selection des données d'intérêt:
var_used = ['TRB', 'AST', 'STL', 'BLK', 'PTS']

# ACP:
acp_plot(data, "Saison 2022-2023", var_used = var_used,
        individus = individus, var_coloration = 'Pos')