source('./TP6 corrigé/src/dataset.R')
n1 <- 100
n2 <- 100
r1 <- 3
r2 <- 10
X <- sample.donut(n1, r1, n2, r2)
plot(X[, 1:2], col = X$y)

# 2
X.train <- sample.donut(n1, r1, n2, r2)
X.test <- sample.donut(10*n1, r1, 10*n2, r2)
library(MASS)
fit <- lda(y ~ ., data = X.train)
pred <- predict(fit, newdata = X.test)$class
mean(pred == X.test$y)

# 3
fit <- lda(y ~ poly(X1, degree = 3) + poly(X2, degree = 3), data = X.train)
pred <- predict(fit, newdata = X.test)$class
mean(pred == X.test$y)

# 4 spline
library(mgcv)
fit <- gam(y ~ s(X1) + s(X2), data = X.train, family = binomial)
pred <- factor(as.numeric(predict(fit, newdata = X.test) > 0) + 1)
mean(pred == X.test$y)
summary(fit)
vis.gam(fit, type = "response", plot.type = "contour")
points(X.train[, 1:2], col = X.train$y)

plot(fit, select = 1)
plot(fit, select = 2)
