set.seed(43)
library(MASS)
attach(Boston)
data = Boston

# Exercise 1
# 1.1
plot(lstat, medv)
fit.poly3 <- lm(medv~poly(lstat, 3), data = Boston)
fit.poly5 <- lm(medv~poly(lstat, 5), data = Boston)
fit.linear <- lm(medv~lstat, data = Boston)
summary(fit1)

ypred.poly3 <- predict(fit.poly3, newdata = Boston)
ypred.linear <- predict(fit.linear, newdata = Boston)
ypred.poly5 <- predict(fit.poly5, newdata = Boston)

idx <- sort(Boston$lstat, index.return = T) # 默认从小到大

lines(lstat[idx$ix], ypred.poly3[idx$ix], col = "red", lty = 3, lwd = 3)
lines(lstat[idx$ix], ypred.poly5[idx$ix], col = "blue", lty = 2, lwd = 3)
lines(lstat[idx$ix], ypred.linear[idx$ix], col = "green", lty = 1, lwd = 3)

# 1.2 déterminer optm)imal value of p
K <- 10
n <- nrow(Boston)
folds <- sample(1:K, n, replace = TRUE)

# function of creating plot for cross validation error
plot.cv.error <- function(data, x.title = "x"){
  ic.error.bar <- function(x, lower, upper, length = 0.1){
    arrows(x, upper, x, lower, angle = 90, code = 3, length = length, col = 'red')
  }
  stderr <- function(x) sd(x)/sqrt(length(x))
  # calculer les erreurs moyennes et l'erreur tpe (standard error)
  means.errs <- colMeans(data)
  std.errs <- apply(data, 2, stderr)
  # plotting 
  x.values <- 1:ncol(data)
  plot(x.values, means.errs, type = "b",
       ylim = range(means.errs + 1.6*std.errs, means.errs - 1.6*std.errs),
       xlab = x.title, ylab = "CV error")
  ic.error.bar(x.values, means.errs - 1.6*std.errs, means.errs + 1.6*std.errs)
}

degrees <- 1:10
cv.err.poly <- matrix(0, ncol = length(degrees), nrow = K)
for(i in (1:length(degrees))){
  for(k in 1:K){
    fit <- lm(medv~poly(lstat, degrees[i]), data = Boston[folds!=k,])
    pred <- predict(fit, newdata = Boston[folds==k,])
    cv.err.poly[k,i] <- mean((Boston$medv[folds==k]-pred)^2)
  }
}

plot.cv.error(cv.err.poly, x.title = 'Degrees')
best.degree.poly <- degrees[which.min(colMeans(cv.err.poly))]
fit.poly.cv <- lm(medv~poly(lstat, best.degree.poly), data = Boston)
pred.poly.cv <- predict(fit.poly.cv, newdata = Boston)
par(mfrow = c(1,2))
plot(lstat, medv)
lines(lstat[idx$ix], pred.poly.cv[idx$ix], col = "red", lwd = 3)    

# 1.3
library(splines)
fit.ns <- lm(medv ~ ns(lstat, df = 7), data = Boston)
ypred.ns <- predict(fit.ns, newdata = Boston)
plot(lstat,medv)
lines(lstat[idx$ix], ypred.ns[idx$ix], col = "red", lwd = 3)
knots <- attr(ns(Boston$lstat, df = 7), "knots")

abline(v = knots, col = "black", lty = 2)

DF <- 1:10
cv.err.splines <- matrix(0, ncol = length(DF), nrow = K)
for(i in (1:length(DF))){
  for(k in 1:K){
    fit <- lm(medv~ns(lstat, DF[i]), data = Boston[folds!=k,])
    pred <- predict(fit, newdata = Boston[folds==k,])
    cv.err.splines[k,i] <- mean((Boston$medv[folds==k]-pred)^2)
  }
}
plot.cv.error(cv.err.splines, x.title = 'Degree of freedom')

best.df.splines <- DF[which.min(colMeans(cv.err.splines))]
fit.ns <- lm(medv ~ns(lstat, df = best.df.splines), data = Boston)
ypred.ns.cv <- predict(fit.ns, newdata = Boston)
plot(lstat, medv)
lines(lstat[idx$ix], pred.poly.cv[idx$ix], col = "red", lwd = 3)
lines(lstat[idx$ix], ypred.ns.cv[idx$ix], col = "blue", lwd = 3)

# 1.4

fit.smooth.cv <- smooth.spline(Boston$lstat, Boston$medv, cv = TRUE)
dfopt <- fit.smooth.cv$df
fit.smooth.splines <- smooth.spline(Boston$lstat, Boston$medv, df = dfopt)
lines(fit.smooth.splines$x, fit.smooth.splines$y, col = "green", lwd = 3)

# Exercice 2
library(gam)
par(mfrow = c(2,2))
fit.gam <- gam(medv ~ s(crim)+ s(lstat) + s(dis) + s(nox), data = Boston, trace = TRUE)
plot(fit.gam, residuals = TRUE, all.terms = TRUE)

# 2.1 
mse.gam <- function(par, folds){
  K <- max(folds)
  par <- abs(par)
  ERR <- 0
  j0<<-par[1]
  k0<<-par[2]
  l0<<-par[3]
  q0<<-par[4]
  for(i in 1:K){
    fit <- gam(medv ~ s(crim, j0)+ s(lstat, k0) + s(dis,l0) + s(nox,q0), 
               data = Boston[folds!=i,], trace = FALSE)
    pred <- predict(fit, newdata = Boston[folds==i,], type = 'response')
    ERR <- ERR + sum((Boston$medv[folds==i]-pred)^2)
  }
  return(ERR/nrow(Boston))
}

# 2.2
opt <- optim(par = c(4,4,4,4), fn = mse.gam, folds = folds, control = list(trace = 1))

# 2.3

