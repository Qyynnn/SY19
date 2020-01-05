# Données Astronomiques
# Multinomial logistic regression, knn, bayes, svm
library(caret)
library(Hmisc) # function decribes
library(corrplot)
library(nnet) # logistic regression
library(FNN) # knn
library(e1071) #svm


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

## 3.2 Logistic regression

cv.errors.logistic <- matrix(NA,10,1)

for (i in 1:10) {
  model <- nnet::multinom(class~., data = train[-folds[[i]],])
  predict <- predict(model, newdata = train[folds[[i]],], type = "class")
  matrix_conf <- table(train[folds[[i]],]$class, predict)
  cv.errors.logistic[i,1] <- mean(1-sum(diag(matrix_conf))/length(folds[[i]]))
}
mean(cv.errors.logistic) # 0.01024317

model_logistic <- nnet::multinom(class~., data = train)
pred_logistic <- predict(model_logistic, newdata = test, type = "class")
matrix_conf <- table(test$class, pred_logistic)
mean(1-sum(diag(matrix_conf))/nrow(test))

# 3.3 knn
cv.errors.knn <- matrix(NA,10,size_k)
size_k <- 20
for (i in 1:10) {
  for (j in 1:size_k) {
    model <- FNN::knn(train = train[-folds[[i]],-11], 
                      test = train[folds[[i]],-11], 
                      cl = train[-folds[[i]],11],
                      k = j)
    matrix_conf <- table(train[folds[[i]],11], model)
    cv.errors.knn[i,j] = mean(1-sum(diag(matrix_conf))/length(folds[[i]]))
  }
}
cv.errors.knn
colMeans(cv.errors.knn)
plot(1:size_k, colMeans(cv.errors.knn), type='b', col='blue',
     xlab='k', ylab='MSE', lty = 1, pch = 1)
boxplot(cv.errors.knn, xlab = "k", ylab = "MSE")
## result abnormal???

cv.errors <- matrix(NA,1,size_k)
for (i in 1:size_k) {
  model <- FNN::knn(train = train[,-11], 
                    test = test[,-11], 
                    cl = train[,11],
                    k = i)
  matrix_conf <- table(test[,11], model)
  cv.errors[1,i] = mean(1-sum(diag(matrix_conf))/nrow(test))
}

lines(1:size_k, cv.errors, type='b', col='red',
     xlab='k', ylab='MSE', lty = 1, pch = 1)
legend("topright", inset = 0.01, legend=c("boxplot - training error", "test error"),
       col=c("black", "red"), lty = 2:1, pch = 0:1)

# 3.4 SVM

cv.errors.svm <- matrix(NA,10,1)

for (i in 1:10) {
  model <- svm(formula = class~., data = train[-folds[[i]],])
  predict <- predict(object = model, newdata = train[folds[[i]],], type = "class")
  matrix_conf <- table(train[folds[[i]],]$class, predict)
  cv.errors.svm[i,1] <- mean(1-sum(diag(matrix_conf))/length(folds[[i]]))
}
mean(cv.errors.svm) # 0.09794987 training error

model_svm <- svm(formula = class~., data = train)
pred_logistic <- predict(object = model_svm, newdata = test, type = "class")
matrix_conf <- table(test$class, pred_logistic)
mean(1-sum(diag(matrix_conf))/nrow(test)) # 0.08416834



