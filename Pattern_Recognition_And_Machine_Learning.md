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
