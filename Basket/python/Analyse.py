### Imports:
from data_analysis_methods import acp_plot, correlation_plot, pairs_plot
import pandas as pd
import pandasql as pdsql
import seaborn as sns
import matplotlib.pyplot as plt

### Lecture des données:
path = "Analyses_Sportives/Basket/data/NBA Player Stats 2022-2023 - Regular.csv"
data = pd.read_csv(path, sep = ';')

### Preprocessing des données: 

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

# Exécution de la requête;:
data = pdsql.sqldf(query_anti_doublon)

# Liste des individus:
individus = data['Player']

### Analyse:

# Requête SQL pour sélectionner les joueurs selon conditions:
query: str = """
    SELECT * FROM data
"""

# Exécution de la requête:
data = pdsql.sqldf(query)

# Variable Catégorielle:
data['Pos'] = data['Pos'].astype('category')

# Etude de la corrélation:
data_numeric = data[data.columns.difference(['Player', 'Pos', 'Team'])]
correlation_plot(data_numeric, "Saison 2022-2023")

# Selection des données d'intérêt:
var_used = ['TRB', 'AST', 'STL', 'BLK', 'PTS']

# Fonction finale:
acp_plot(data, "Saison 2022-2023", var_used = var_used,
        individus = individus, var_coloration = 'Pos')
