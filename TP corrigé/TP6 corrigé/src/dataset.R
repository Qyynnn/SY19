sample.donut <- function(n1, r1, n2, r2) {
    p <- 2

    R1 <- rnorm(n1, mean = r1)
    angle1 <- runif(n1, 0, 2 * pi)
    X1 <- data.frame(X1 = R1 * cos(angle1), X2 = R1 * sin(angle1))

    R2 <- rnorm(n2, mean = r2)
    angle2 <- runif(n2, 0, 2 * pi)
    X2 <- data.frame(X1 = R2 * cos(angle2), X2 = R2 * sin(angle2))

    cbind(rbind(X1, X2), y = factor(c(rep(1, n1), rep(2, n2))))
}
