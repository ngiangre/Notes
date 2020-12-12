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

# Chapter 6 Kernel Methods

- Usually the training data is tossed away after learning the parameters, but there's approaches that still use the training data or a subset
- These memory based methods may construct functions based on some training data, fast to train but slower to predict
- Could be nonlinear feature space - often symmetric. Linear kernel is the inner product between two feature vectors.

## Dual Representations

- comparing and fitting two vectors or represeentations

## Constrtucting Kernels

- corresponding to a scalar product

## Radial basis function networks

- each basis function is based on a distance from a center - great for interpolation

## Gaussian processes

- extension to probabilistic discriminitive models introducing the bayesian viewpoint to kernel methods

# Chapter 7 Sparse Kernel Machines

- Different from other kernel methods in that only a subset of the training is needed to make predictions.
- SVM is an example

## Maximum Margin Classifiers

- Found by computational learning theory
- reduces to convex optimization
- single class support vector machines are an unsupervised alternative - aim to find a smooth boundary for enclosing high density areas
- but output are decisions and not posteriors

## Relevance Vector Machines

- Bayesian, faster performance estimation, similar sparseness
- training is longer because it's a nonconvex optimization problem
- makes probabilistic predictions

# Chapter 8 Graphical Models

## Bayesian Networks

- aka directed graphical models
- the joint distribution can be written as a product of conditional distributions, one for each of the variables. This is called factorization
- can surround multiple nodes in a plate for easier graphical representation
- shading nodes symbolizes observed variables and unshaded nodes are hidden/latent. Solid circles are deterministic. 
- New predictions are then derived from the sum rule marginilizing over the weights.
- ancestral sampling - generating new input x from joint distributions

## Conditional Independence

- simplifies structures and amount of computation/factorization to be done
- can test with d-separation

## Markov Random Fields

- aka undirected graphical models
- simpler d-separation testing than DAGs
- Markov blanket - set of nodes for testing conditional independence to another set of nodes
- use cliques to define sets of nodes - but need an explicit normalization constant but can use boltzmann distributions in inexact potential functions instead


## Inference in Graphical Models

- useful to converts graphical models to factor graphs
- idea is to pass local messages around
- exact-inference using a tree like structure
- belief propogation and the junction tree algorithm gives exact inference, variational methods give approximations as well as sampling/monte carlo methods for nondeterministic

# Chapter 9 Mixture Models and EM

## K means Clustering

- Also used for data compression and image segmentation

## Mixtures of Gaussians

- a method for finding maximum likelihood estimates for models with latent variables is EM

## An Alternative View of EM

- Inching towards the parameters via converge 

## The EM algorithm in general

- Forms the foundation for the variational inference framework. 
- Two stage iterative optimization technique

# Chapter 10 Approximate Inference

## Variational inference

- Uses a unknown covariate Z to "suck up" the latent variables
- Said to give a good approximation of the posterior distrtibution 

## Illustration: variational mixture of gaussians

- good illustration of the application of variational methods and will also demonstrate how a Bayesian treatment elegantly resolves many of the difficulties associated with the maximum likelihood approach
- the optimization of the variational posterior distribution involves cycling
between two stages analogous to the E and M steps of the maximum likelihood EM algorithm.
- if we consider the limit N →∞then the Bayesian treatment converges to the maximum likelihood EM algorithm

## Variational linear regression

- the variational approach gives precisely the same expression as that obtained by maximizing the evidence function using EM except that the point estimate for α is replaced by its expected value

## Exponentional family distributions

## Local variational messages

- An alternative ‘local’ approach involves finding bounds on functions over individual variables or groups of variables within a model.

## Variational logistic regression

- the greater flexibility of the variational approximation leads to improved accuracy compared to the Laplace method.

## Expectation propogation

- an alternative form of deterministic approx- imate inference
- based on the minimization of a Kullback-Leibler divergence but now of the reverse form, which gives the approximation rather different properties
- Expectation propagation is based on an approximation to the posterior distribu-
tion which is also given by a product of factors
- Expectation propagation makes a much better approximation by optimizing each
factor in turn in the context of all of the remaining factors
- One disadvantage of expectation propagation is that there is no guarantee that
the iterations will converge.

# Chapter 11 Sampling Methods

- Allows for approximate versus deterministic inference
- As long as samples are drawn from the same probability distribution then the expectation  of f hat is the expectatio of f.
 
## Basic sampling methods

- The rejection sampling framework allows us to sample from relatively complex
distributions, subject to certain constraints.
- The technique of importance sampling provides a framework for approximating expectations di- rectly but does not itself provide a mechanism for drawing samples from distribution p(z)
- MC sampling can replace the E step of Expectation-Maximization

## Markov Chain Monte Carlo

- Good sampling approach for high dimmensional data
- The sample state is stored and determmines the next sampled state
- The probability of transition is the same from the t state to the t+1 state

## Gibbs Sampling

- Gibbs sampling (Geman and Geman, 1984) is a simple and widely applicable Markov chain Monte Carlo algorithm and can be seen as a special case of the Metropolis- Hastings algorithm. Consider

## Slice Sampling

- defines step size to captture distribution characteristics when sampling

## The Hybrid Monte Carlo Algorithm

- based on physical systems
- the Hamiltonian dynamical approach involves alternating between a series of leapfrog updates and a resampling of the momentum variables from their marginal distribution.

## Estimating the partition function


