
# Introduction ------------------------------------------------------------

#' while pievewise linear smoothers reasonably represent smooth functions, using spline bases we can obtain substantially reduced funcion approximation error for a given dimension of smoothing basis. 
#' 


# Smoothing splines -------------------------------------------------------

#' the smoothing spline is a result of minimizing a cost function with a lambda value that penalizes wiggliness to result in a smoother representation of the resulting spline.
#' the numberr of free parameters is O(n) to fit the cubic spline so that's OK but when adding more covariates this gets computationally expensive.
#' 


# Penalized regression splines --------------------------------------------

#' penalized splines are a good balance of spline effects and compuational efficiency.
#' At its simplest, it involves constructing a spline basis for a much smaller dataset than the one to be analysed and then using the basis (plus penalities) to model the original dataset. 
#' 


# Some one-dimensional smoothers ------------------------------------------

# The cubic spline basis doesn't require any rescaling of the predictor variables.

bspline <- function(x,k,i,m=2){
    # evaluate ith B-spline basis function of order m at the
    # values in x, given knot locations in k
    if(m==-1){ #base of the recursion
        res <- as.numeric(x<k[i+1]&x>=k[i])
    }else{ # construct from call to lower order basis
        z0 <- (x-k[i])/(k[i+m+1]-k[i])
        z1 <- (k[i+m+2]-x)/(k[i+m+2]-k[i+1])
        res <- z0*bspline(x,k,i,m-1) + z1*bspline(x,k,i+1,m-1)
    }
    res
}


k <- 6 # example basis dimension
P <- diff(diag(k),difference=1) # sqrt of penalty matrix- higher order penalties are imposed by increasing the difference parameter
S <- t(P)%*%P # penalty matrix

library(mgcv)
x = sin(seq(0,2*pi,length.out=100)) + runif(100)
y = sin(seq(0,2*pi,length.out=100))
ssp <- s(x,bs="ps",k=k); ssp$mono <- 1
sm <- smoothCon(ssp,data.frame(x))[[1]]
X <- sm$X; XX <- crossprod(X);sp <- .5
gamma <- rep(0,k); S <- sm$S[[1]]
for(i in 1:20){
    gt <- c(gamma[1],exp(gamma[2:k]))
    dg <- c(1,gt[2:k])
    g <- -dg*(t(X)%*%(y-X%*%gt)) + sp*S%*%gamma
    H <- dg*t(dg*XX)
    gamma <- gamma - solve(H+sp*S,g)
}


# isotropic smoothing -----------------------------------------------------


# tensor product smooth interactions --------------------------------------


# Smooths, random fields, and random effects ------------------------------


# choosing the basis dimension --------------------------------------------


# generalized smoothing splines -------------------------------------------




