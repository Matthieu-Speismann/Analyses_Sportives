setwd("G:/Mon Drive/Fun/Stat NBA/Perso")

library(FactoMineR)
library(corrplot)
library(Factoshiny)


nba <- read.csv("2022-2023 NBA Player Stats - Regular.csv",
                sep = ";", header = TRUE)

nba[, "Team"] <- factor(nba[, "Team"])
nba[, "Pos"] <- factor(nba[, "Pos"])
nba[, "Player"] <- factor(nba[, "Player"])

summary(nba)

C <- cor(nba[7:30])
corrplot(C, method = "circle")

res_pca <- PCA(nba,
               quali.sup = c(2, 3, 5),
               quanti.sup = c(1, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21),
               graph = FALSE)

plot.PCA(res_pca, choix = "var", )
plot.PCA(res_pca, invisible = c("ind.sup"),
         cex = 0.5, cex.main = 0.5, cex.axis = 0.5,
         col.quali = "#0D00FF", label = c("quali"),
         max.overlaps = 1000)



