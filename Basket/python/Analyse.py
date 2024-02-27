# Imports:
from Exemple_ACP import affichage_acp
import pandas as pd
import pandasql as pdsql

# Lecture des données:
path = "Analyses Sportives/Perso/Basket/data/NBA Player Stats 2022-2023 - Regular.csv"
data = pd.read_csv(path, sep = ';')
     
# Requete SQL pour sélectionner les individus, prenant en compte les transfert en cours d'année:   
query = """
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

# Exécution de la requête - Permet de choisir les individus selon certains critères:
data = pdsql.sqldf(query)

# Variable Catégorielle:
data['Pos'] = data['Pos'].astype('category')

# Liste des individus:
individus = data['Player']

# Selection des données d'intérêt:
var_used = ['TRB', 'AST', 'STL', 'BLK', 'PTS']

# Fonction finale:
affichage_acp(data, "Saison 2022-2023", var_used = var_used,
               individus = individus, var_coloration= 'Pos')
