USE formula1;

SELECT D.prenom, D.nom, C.nom AS Ecurie, R.year,  DS.points
FROM drivers AS D
JOIN drivers_standings AS DS ON
    DS.driverID = D.driverId 
JOIN races AS R ON
    R.raceId = DS.raceId
JOIN qualifying AS Q ON
    Q.raceId = R.raceId AND Q.driverId = D.driverId
JOIN constructors AS C ON
    C.constructorId = Q.constructorId
WHERE 
    DS.position = 1 and 
    R.round = 
        -- Sous requête pour récupérer le dernier round de chaque année:
        (SELECT MAX(round)
        FROM Races
        WHERE year = R.year)
ORDER BY R.year DESC;
