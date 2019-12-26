# Données Astronomiques
# Multinomial logistic regression, knn, bayes, svm
library(caret)
library(Hmisc)
library(corrplot)
library(nnet)


set.seed(1729)

# 1. load dataset
Astronomy <- read.csv("./Données du TP 10-20191217/astronomy_train.csv", header = TRUE)
Astronomy$class <- as.factor(Astronomy$class)
Astronomy1 <- Astronomy

# 2. Exploring and pre-treatment

## 2.1 verify the value unknown
describe(Astronomy1) # 0 unknown
summary(Astronomy)

## 2.2 Traitement de données

###  2.2.1 Delete the preditors invariants 
# objid = 1.238e+18, rerun = 301
# reorgnaize the dataframe by placing column class to the last                                                                   

zerovar <- nearZeroVar(Astronomy1)
cols <- c(2:9, 11:13, 15:18, 14)
Astronomy1 <- Astronomy1[,cols]
Astronomy1

###  2.2.2 Delete the params having strong correlation with other param

descrCorr <- cor(Astronomy1[,-16])
corrplot(descrCorr, type="upper", order="hclust", tl.col="black", tl.srt=45)
highCorr = findCorrelation(descrCorr, 0.8) # high Correlation
colnames(Astronomy1[,highCorr]) # "g", "i", "z", "specobjid", "plate"
Astronomy1 <- Astronomy1[,-highCorr]
str(Astronomy1)

###  2.2.3 Scaling/standalize data
pre <- preProcess(Astronomy1[,-11], method = c("center", "scale"))
data <- predict(pre, Astronomy1) # data of Astronomy1 after scaling 
summary(data)

### 2.2.4 Division de training set et Test set (80% - 20%)
index <- createDataPartition(data$class, p = 0.80, list = FALSE) # first generation after set.seed 
train <- data[index,] # we use this "train" data to fit model
test <- data[-index,] # we use this "test" data just to test the final model


# 3. Fitting with model

## 3.1 setting up k-fold 
folds <- createFolds(y=train[,11],k=10)

## 3.1 Logistic regression

cv.errors.logistic <- matrix(NA,10,1)

for (j in 1:10) {
  model <- nnet::multinom(class~., data = train[-folds[[j]],])
  predict <- predict(model, newdata = train[folds[[j]],], type = "class")
  matrix_conf <- table(train[folds[[j]],]$class, predict)
  cv.errors.logistic[j,1] <- mean(1-sum(diag(matrix_conf))/length(folds[[j]]))
}

mean(cv.errors.logistic) # 0.01024317

model_logistic <- nnet::multinom(class~., data = train)
pred_logistic <- predict(model_logistic, newdata = test, type = "class")
matrix_conf <- table(test$class, pred_logistic)
mean(1-sum(diag(matrix_conf))/nrow(test))






