# Données Astronomiques
library(caret)
library(ggplot2)
library(Hmisc)
set.seed(1729)
Astronomy <- read.csv("./Données du TP 10-20191217/astronomy_train.csv", header = TRUE)
Astronomy$class <- as.factor(Astronomy$class)

Astronomy1 <- Astronomy

# vérifier s'il y a des variables unknown
describe(Astronomy1) # 0 unknown

summary(Astronomy1$class)

# Correlation
panel.cor <- function(x, y){
  usr <- par("usr"); on.exit(par(usr))
  par(usr = c(0, 1, 0, 1))
  r <- round(cor(x, y), digits=2)
  txt <- paste0("R = ", r)
  text(0.5, 0.5, txt)
}

# Traitement de données



# LDA
