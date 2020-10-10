# Chapter 1 Introduction to searching forc patterns in data

- Generalization: the ability of a model to correctly categorize examples that different from the training/learning phase
- Density estimation: determining the distribution of the data within the input space.
- Model coefficient's magnitudes may correspond with overrfitting - a general property of maximum likelihood
- Regularization of neural networks uses weight decay
- Elementary rules of probability: the sum and product rules (Bayes theorem is the product rule divided by the sum rule essentially)
- When deciding what to do for a patient xi within the infered P(x,t), we can either decide based on the posterior P(t|xi), based on a decision region with the fewest misclassifications, minimizing the expected loss of a specified criteria, not deciding based on a posterior threshold value, or jointly modeling the inference-decision process (discrimination) 
- Information theory: how much information is received when we observe a specific value for this variable. The amount of information can be viewed as the ‘degree of surprise’ on learning the value of x.
- the mutual information can be thought of as the reduction in the uncertainty about x by virtue of being told the value of y
 
# Chapter 2 Probability Distributions

- Distrributions are building blocks for pattern recognition models
- Density estimatation, modeling the probability distribution p(x) of a random variable x, given a finite set x1,..., xN of observations, is fundamentally illposed because an infinite amount of distributions could have generated the data
- Conjugate priors are useful in bayesian inference because they give the same functional form of the prior for the posterior. Multinomial > Direchlet, Gaussian -> Gaussian
- In nonparametric density estimation, the parameters for the distribution control the model complexity rather than the form of the distribution. Example methods use histograms (ni/N\*binwidth), nearest neighbor (parameter K), and kernels (parameter h). 
- Sequential methods make use of observations one at a time, or in small batches, and then discard them before the next observations are used. They can be used, for example, in real-time learning scenarios where a steady stream of data is arriving, and predictions must be made before all of the data is seen. Because they do not require the whole data set to be stored or loaded into memory, sequential methods are also useful for large data sets. Maximum likelihood methods can also be cast into a sequential framework.
- a statistic is sufficient with respect to a statistical model and its associated unknown parameter if "no other statistic that can be calculated from the same sample provides any additional information as to the value of the parameter". It depends only on the data. 
- The central limit theorem (due to Laplace) tells us that, subject to certain mild conditions, the sum of a set of random variables, which is of course itself a random variable, has a distribution that becomes increas- ingly Gaussian as the number of terms in the sum increases
- Student’s t-distribution is obtained by adding up an infinite number of Gaussian distributions having the same mean but different preci- sions. This can be interpreted as an infinite mixture of Gaussians - this distribution is more robust to outliers because of this. 
- a noninformative prior is intended to have as little influence on the posterior distribution as possible

# Chapter 3 Linear Models for Regression

- in supervised learning, one can model the target variable with linear combinations of a set of linear or nonlinear functions (using basis functions that can lead to splines and other possibly different transformations across an input space) of the input variables
- including a regularization term in SVD, which can solve the least squares problem, can avoid singularity of the data fitting
- sequential or online learning can be done with SGD to update the parameters as more data is seen. Regularizers are also called weight decay taken from sequential learning. 
- the overfitting problem goes away in the bayesian setting because of marginalizing over parameters. 
- ML frameworks have the tradeoff between bias (the extent to which the average prediction over all data sets differs from the desired regression function) and variance (the extent to which the solutions for individual data sets vary around their average) - the balance is achieved at optimal prediction
- a broad prior distribution over weights to derive the posterior is equivalent to the maximum likelihood of the weights in linear regression
- the predictive distribution at a point x is a linear combination of its target variables and is formulated with an equivalent kernel (from the use of basis functions) which is just matrix algebra with the weight and covariance martices, also depending on the input data. The effective kernel can define the weights by which the training set target variables are combined in order to make a new prediction, and these weights sum to 1. This leads to the gaussian process framework.
- in comparing bayesian models, a validation set isn't need (a hold out set is good practice). Only need to marginalize over the parameters used  on the training data. We can compare model preference with the posterior distributions encoded by the ratio of bayes factors. One can pick a single model (model selection) or a collection of models (mixture distribution). 
- Implicit in the Bayesian model comparison framework is the assumption that the true distribution from which the data are generated is contained within the set of models under consideration. The Kullback Leiber quantity gives the average bayes factor (the bayes factor could be higher for an incorrect model (on average) and so integrating the ratio of model probabilities given the true input space gives an average bayes factor).
- Empirical bayes - hyperparameters set by MLE over the parameters. 
- linear combinations of fixed basis functions have drawbacks that methods like NNs can alleviate. Neural network models, which use adaptive basis functions having sigmoidal nonlinearities, can adapt the parameters so that the regions of input space over which the basis functions vary corresponds to the data manifold. The second property is that target variables may have significant dependence on only a small number of possible directions within the data manifold. Neural networks can exploit this property by choosing the directions in input space to which the basis functions respond.

# Chapter 4 Linear Models for Classification

## Discrimant functions

- takes data x and assigns it to a class C.
- Parameters can be learned by least squares (not robust to outliers), the fisher criterian (optimizing interclass difference and minimizing intraclass variance), and the perceptron learning algorithm. 

## Probabilistic generative models

- relies on bayes theorem to quantify uncertainty in class probability distribution given the data.

## Probabilistic discriminitive models

- maximizing likelihood and class condidtional distributions and then finding weight parameters instead of the other way around. 

## The Laplace transformation

- Finds a gaussian approximation for the posterior disribution

## Bayesian logistic regression

- Exact bayesian inference of logistic regression is intractable. 
- The laplace approximation is used for estimating the weights. 

# Chapter 5 Neural Networks

- need for adaptive basis functions
- SVM centers basis functions on training data and convex optimization fits the model
- MLP: fixed basis functions with parameters addapting during training. Not convex :/ learn parameters with MLE framework (nonlinear optimization problem)

## Feed Forward Network Functions

- linear combinations of parametric nonlinear basis functions for addjusting parameters

## Network Training

- analagous to polynomial curve fitting minimizing a sum of squares error function using partial derivatives or gradient descent optimization

## Error backpropogation

- finding a way for efficient evaluation the gradient of the error function
- two steps: calculate backwards the derivates and then optimize forward using calculated derivates

## The Hessian Matrix

- useful to derive second derivatives for faster retraining with new data and other model validations

## Regularization in Neural Networks

- Can control number of hidden units or add regularizer term to error function (weight decay)
- early stopping before the training error minimum is reached

## Mixture Density Networks

- adaptive approach for modeling conditional probabilities for which there isn't a unique solution
- the conditional mean poorly predicts a multimodal distribution, the conditional mode can be derived numerical to provide better predictions. 

## Bayesian Neural Networks

- quantifying the marginal distributions to get the posterior can be done by variational inference (posterior has local minima)
- the posterior distribution is gaussian using a laplace approximation
- variational inference adds variance to calculating the posterior distribution and results in a gaussian!
- marginalization makes the predictions effectively less predictive

