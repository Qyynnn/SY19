# TP7 Tree based and ensemble method
credit <- read.csv('./TableF7-3.csv', sep=",",header=TRUE)
head(credit)
n <- nrow(credit)
ntrain <- 10000
ntest <- n - ntrain
idx.train <- sample(n, ntrain)

# Exercice 1

# 1.2
library(rpart) # 
credit$CARDHLDR <- as.factor(credit$CARDHLDR)
fit <- rpart(CARDHLDR ~ .- DEFAULT - SPENDING - LOGSPEND - EXP_INC,
                  data = credit, subset = idx.train, method = "class",
                  control = rpart.control(xval = 10, minbucket = 10, cp = 0.00))
# cp = alpha, alpha = 0时为全树

plot(fit, margin = 0.005)
text(fit, minlength=0.5, cex=0.2, splits=TRUE)

library(rpart.plot)
rpart.plot(fit, box.palette="RdBu", shadow.col="gray", fallen.leaves=FALSE)

yhat <- predict(fit, newdata = credit[-idx.train,], type = "class")
ytest <- credit[-idx.train,]$CARDHLDR
matrix.conf.tree <- table(ytest, yhat)
matrix.conf.tree
err.tree <- 1- sum(diag(matrix.conf.tree))/ntest
err.tree

# 1.3 pruning
printcp(fit)
plotcp(fit, minline = TRUE)
i.min <- which.min(fit$cptable[,4])
cp.opt <- fit$cptable[i.min,1]
prune_tree <- prune(tree = fit, cp = cp.opt)
rpart.plot(prune_tree,box.palette="RdBu", shadow.col="gray")
yhat <- predict(prune_tree,newdata=credit[-idx.train,],type='class')
matrix.conf.pruned <- table(ytest,yhat)
matrix.conf.pruned
err.pruned <- 1-mean(ytest==yhat)
err.pruned

# 1.4 ROC curve
library(pROC)
prob <- predict(prune_tree, newdata = credit[-idx.train,], type = "prob")
roc_tree_pruned <- roc(ytest, prob[,1])
plot(roc_tree_pruned)
TPR <- matrix.conf.pruned[2,2]/rowSums(matrix.conf.pruned)[2]
FPR <- matrix.conf.pruned[1,2]/rowSums(matrix.conf.pruned)[1]
points(1-FPR, TPR, col="red", pch = 19)
legend("bottomright", lty = 1, legend=c("Tree pruned"), col=c("black"))

# 1.5 bagging decision tree and random forest decision tree
# Bagging
library(randomForest)
credit.new <- credit[,-c(2,12,13,14)]
p <- ncol(credit.new)-1 # number of predictors
fit.bagged <- randomForest(CARDHLDR~., data = credit.new, subset = idx.train, mtry = p)

yhat <- predict(fit.bagged, newdata = credit.new[-idx.train,], type = "response")
matrix.conf.bag <- table(ytest, yhat)
err.bagged <- 1-mean(ytest == yhat)
err.bagged

# Random Forest
fit.rf <- randomForest(CARDHLDR~., data = credit.new, subset = idx.train)
yhat <- predict(fit.rf, newdata = credit.new[-idx.train,], type = "response")
matrix.conf.rf <- table(ytest, yhat)
err.rf <- 1-mean(ytest == yhat)
err.rf
summary(fit.rf)


# 1.6 Logistic regression and GAM comparaison
# LR
fit.lr <- glm(CARDHLDR~., data = credit.new, subset = idx.train, family = "binomial")
summary(fit.lr)
pred.lr <- predict(fit.lr, newdata=credit.new[-idx.train,], type="response")
pred.lr
yhat <- pred.lr > 0.5
matrix.conf.lr <- table(ytest, yhat)
err.lr <- 1 - sum(diag(matrix.conf.lr))/ntest
err.lr

# GAM
library(gam)
fit.gam<-gam(CARDHLDR ~ s(AGE)+s(ACADMOS)+s(ADEPCNT)+s(MAJORDRG)+
               s(MINORDRG)+OWNRENT+s(INCOME)+SELFEMPL+s(INCPER),
             data=credit.new, subset=idx.train, family='binomial', trace=TRUE)
par(mfrow=c(3,3))
plot(fit.gam, residuals=TRUE, all.terms = TRUE)
