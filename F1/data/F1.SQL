CREATE DATABASE Formula1;
USE Formula1;

-- Créez une table pour le premier fichier CSV
CREATE TABLE Circuits (
    circuitID INT PRIMARY KEY,
    circuitRef TEXT,
    nom TEXT,
    localisation TEXT,
    pays TEXT,
    lat FLOAT,
    lng FLOAT,
    alt INT,
    url TEXT
);

-- Importez les données du premier fichier CSV
LOAD DATA LOCAL INFILE 'G:/Mon Drive/Fun/Test/F1/data/circuits.csv'
INTO TABLE Circuits
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
IGNORE 1 ROWS; -- Ignore la première ligne si elle contient des en-têtes


CREATE TABLE Constructors_Results(
    constructorResultsId INT PRIMARY KEY,
    raceId INT,
    constructorId INT,
    points FLOAT,
    status TEXT
);

LOAD DATA LOCAL INFILE 'G:/Mon Drive/Fun/Test/F1/data/constructor_results.csv'
INTO TABLE Constructors_Results
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

CREATE TABLE Constructors_Standings(
    constructorStandingsId INT PRIMARY KEY,
    raceId INT,
    constructorId INT,
    points FLOAT,
    position INT,
    positionText TEXT,
    wins INT
);

LOAD DATA LOCAL INFILE 'G:/Mon Drive/Fun/Test/F1/data/constructor_standings.csv'
INTO TABLE Constructors_Standings
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

CREATE TABLE Constructors(
    constructorId INT PRIMARY KEY,
    constructorRef TEXT,
    nom TEXT,
    nationalite TEXT,
    url TEXT
);

LOAD DATA LOCAL INFILE 'G:/Mon Drive/Fun/Test/F1/data/constructors.csv'
INTO TABLE Constructors
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

CREATE TABLE Drivers(
    driverId INT PRIMARY KEY,
    driverRef TEXT,
    number INT,
    code TEXT,
    prenom TEXT,
    nom TEXT,
    dateNaissance TEXT,
    nationalite TEXT,
    url TEXT
);

LOAD DATA LOCAL INFILE 'G:/Mon Drive/Fun/Test/F1/data/drivers.csv'
INTO TABLE Drivers
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

CREATE TABLE Drivers_Standings(
    driverStandingsId INT PRIMARY KEY,
    raceId INT,
    driverId INT,
    points FLOAT,
    position INT,
    positionText TEXT,
    wins INT
);

LOAD DATA LOCAL INFILE 'G:/Mon Drive/Fun/Test/F1/data/driver_standings.csv'
INTO TABLE Drivers_Standings
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

CREATE TABLE Lap_Times(
    raceId INT,
    driverId INT,
    tour INT,
    position INT,
    temps TEXT,
    tourEnMilli INT
);

LOAD DATA LOCAL INFILE 'G:/Mon Drive/Fun/Test/F1/data/lap_times.csv'
INTO TABLE Lap_Times
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

CREATE TABLE Pit_Stops(
    raceId INT,
    driverId INT,
    arret INT,
    tour INT,
    heure TEXT,
    duree TEXT,
    duree_ms INT
);

LOAD DATA LOCAL INFILE 'G:/Mon Drive/Fun/Test/F1/data/pit_stops.csv'
INTO TABLE Pit_Stops
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

CREATE TABLE Qualifying(
    qualifyId INT PRIMARY KEY,
    raceId INT,
    driverId INT,
    constructorId INT,
    number INT,
    position INT,
    temps_q1 TEXT,
    temps_q2 TEXT,
    temps_q3 TEXT
);

LOAD DATA LOCAL INFILE 'G:/Mon Drive/Fun/Test/F1/data/qualifying.csv'
INTO TABLE Qualifying
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

CREATE TABLE Races(
    raceId INT PRIMARY KEY,
    year INT,
    round INT,
    circuitId INT,
    nom TEXT,
    date TEXT,
    heure TEXT,
    url TEXT,
    fp1_date TEXT,
    fp1_heure TEXT,
    fp2_date TEXT,
    fp2_heure TEXT,
    fp3_date TEXT,
    fp3_heure TEXT,
    qualif_date TEXT,
    qualif_heure TEXT,
    sprint_date TEXT,
    sprint_heure TEXT
);

LOAD DATA LOCAL INFILE 'G:/Mon Drive/Fun/Test/F1/data/races.csv'
INTO TABLE Races
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

CREATE TABLE Results(
    resultId INT PRIMARY KEY,
    raceId INT,
    driverId INT,
    constructorId INT,
    number INT,
    grid INT,
    position INT,
    positionText TEXT,
    positionOrder INT,
    points FLOAT,
    laps INT,
    tempsTotal TEXT,
    temps_ms INT,
    fastestLap INT,
    rang INT,
    fastestLapTime TEXT,
    fastestLapSpeed FLOAT,
    statusId INT
);

LOAD DATA LOCAL INFILE 'G:/Mon Drive/Fun/Test/F1/data/results.csv'
INTO TABLE Results
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

CREATE TABLE Seasons(
    year INT PRIMARY KEY,
    url TEXT
);

LOAD DATA LOCAL INFILE 'G:/Mon Drive/Fun/Test/F1/data/seasons.csv'
INTO TABLE Seasons
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

CREATE TABLE Status(
    statusId INT PRIMARY KEY,
    status TEXT
);

LOAD DATA LOCAL INFILE 'G:/Mon Drive/Fun/Test/F1/data/status.csv'
INTO TABLE Status
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

CREATE TABLE Sprint_Results(
    raceId INT,
    driverId INT,
    constructorId INT,
    number INT,
    grid INT,
    position INT,
    positionText TEXT,
    positionOrder INT,
    points FLOAT,
    laps INT,
    tempsTotal TIME,
    temps_ms INT,
    fastestLap INT,
    fastestLapTime TEXT,
    statusId INT
);

LOAD DATA LOCAL INFILE 'G:/Mon Drive/Fun/Test/F1/data/sprint_results.csv'
INTO TABLE Status
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;