# The Titanic

Please note that the code that accompanies this is all in one file, and as such will take an extremely long time to run. It is intended that anyone who wishes to use it comment out any code they do not wish to run before executing the file. 

### Step One
'Define the problem/understand the business'

In this project, I will try to create a model that will be able to accurately predict the likelihood of a person surviving the Titanic disaster, given the records of them taken at the time of boarding the ship.\
The efficacy of the model will be determined by the test set kept by the Kaggle website. 

### Step Two
'Locate and acquire relevant datasets'

I have downloaded the training dataset - including both feature and target data, the test dataset - including only target data, and a submission example file - to ensure I know the format in which to supply my results to Kaggle.\

### Step Three 
'Perform an EDA of the datasets - including some preliminary cleaning and perhaps feature engineering - to understand it's value and quality. Make sure the dataset is in some form that can be fed into a mathematical, algorithmically trained model'

To begin with, I will look at the description of the datasets that has been provided to me by Kaggle: 

###### Data Dictionary

| Variable | Definition | Key |
| --- | --- | --- |
| survival | Survival | 0 = No, 1 = Yes |
| pclass | Ticket class | 1 = 1st, 2 = 2nd, 3 = 3rd |
| sex | Sex |
| Age | Age in years |
| sibsp | # of siblings / spouses aboard the Titanic | 
| parch | # of parents / children aboard the Titanic | 
| ticket | Ticket number |
| fare | Passenger fare |
| cabin | Cabin number |
| embarked | Port of Embarkation | C = Cherbourg, Q = Queenstown, S = Southampton |

##### Variable Notes

**pclass**: A proxy for socio-economic status (SES)
1st = Upper
2nd = Middle
3rd = Lower

**age**: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5

**sibsp**: The dataset defines family relations in this way...
Sibling = brother, sister, stepbrother, stepsister
Spouse = husband, wife (mistresses and fiancés were ignored)

**parch**: The dataset defines family relations in this way...
Parent = mother, father
Child = daughter, son, stepdaughter, stepson
Some children travelled only with a nanny, therefore parch=0 for them.

This describes 10 different features, of which 4 are categorical, and 6 are continuous (or who's range of categories is so large as to be equivalent to continuous data, e.g. cabin number).\
I will load the data and look at it for myself to see if this limited description is accurate.

The training data as loaded has 12 features: 
- PassengerId
- Survived
- Pclass
- Name
- Sex
- Age
- SibSp
- Parch
- Ticket
- Fare
- Cabin
- Embarked

Which is very similar to what was described in the data dictionary, with the addition of PassengerId, which I agree needs no introduction and has no realistic use in determining survivability, and Survived, which is obviously not a training feature, but a training target that, again, requires no explanation.\
The training data has 891 samples, which should be plenty.\
The test data has all the same features with the exception of Survived, and has 418 samples. 

I think now is a good time to look at each feature and think about what it means and how useful it's going to be.

First up is _PassengerId_, a predicatively useless feature that was added to the dataset to enumerate the data samples. I will remove this prior to training.

Second is _Survived_. This is the binary target variable and will be used to inform the model if it's right or wrong during training.

Third is _Pclass_, a feature that describes the ticket class that the passenger purchased. According to the data description provided by Kaggle, this can be used to infer socioeconomic class, which makes intuitive sense to me, so I will agree. How this affects survival is something I could speculate on, but probably has to do with access to survival resources (lifejackets, crew assistance, lifeboat boarding priority, etc) 

Fourth is _Name_. Since names are _independent_ of each other, and do nothing to describe a person, this feature is useless and I will not be using it for training.

Fifth is _Sex_. Given the varied and extensive inter-sex social dynamics that are pervasive to all humans, this is likely to be a good predictive feature, due (in my opinion) to the likelihood of behaviour affecting survivial.

Sixth is _Age_. I believe this will be a useful feature, for similar reasons to the _Sex_ feature. The young and old are often favoured in escape survival situations (that is, they are according to my understanding). Conversely, those who are too young or too old (infirm) to get themselves to safety, may be more likely to die.

Seventh is _SibSp_. I think this could be very relevant, given that group cooperation is a major force for success in human history, and humans are more likely to cooperation with family members.

Eighth is _Parch_. This will likely be relevant for the same reason _SibSp_ is, though may require some combination with age and possibly sex to be fully useful. Having a parent to look after you is an advantage in survival situations, and having a child may give you a better argument for getting onto a lifeboat.

Ninth is _Ticket_. This is tell us nothing of any use, given that we already know what class they were travelling in. It is just another arbitrary coding system like _PassengerId_ that has no predictive worth.

Tenth is _Fare_. How much they paid for their ticket could be of use, if wealth is a useful indicator. It would add more nuance to the socioeconomic information provided by _Pclass_.

Eleventh is _Cabin_. I don't know if this will be of much use. It could be, in that some part of the cabin designation could indicate which part of the ship the passenger was in, and therefore relate to their distance from lifesaving equipment and escape vessels, however, many people may not have been in their cabins at the time, so it could just add noise. I will include it, and allow the models to determine it's usefulness for themselves.

Twelfth is _Embarked_. Where they got on the ship shouldn't be relevant, as it's not related to the sinking in any way (everyone embarked beforehand, and I know of no grouping by embarkation whatsoever). I will not use this for training.

So, the features I intend to use (for now, who knows what will come later) are: 
- Survived (as the target)
- Pclass
- Sex
- Age
- SibSp
- Parch
- Fare
- Cabin

Some statistical information and comparison of the selected variables should be useful at this point. A straight forward `pandas.describe()` function call provides this information: 

| Statistical Attribute | PassengerId | Survived | Pclass | Age | SibSp | Parch | Fare |
| --- | --- | --- | --- | --- | --- | --- | --- |
| count | 891.000000 | 891.000000 | 891.000000 | 714.000000 | 891.000000 | 891.000000 | 891.000000 |
| mean | 446.000000 | 0.383838 | 2.308642 | 29.699118 | 0.523008 | 0.381594 | 32.204208 |
| std | 257.353842 | 0.486592 | 0.836071 | 14.526497 | 1.102743 | 0.806057 | 49.693429 |
| min | 1.000000 | 0.000000 | 1.000000 | 0.420000 | 0.000000 | 0.000000 | 0.000000 |
| 25% | 223.500000 | 0.000000 | 2.000000 | 20.125000 | 0.000000 | 0.000000 | 7.910400 |
| 50% | 446.000000 | 0.000000 | 3.000000 | 28.000000 | 0.000000 | 0.000000 | 14.454200 |
| 75% | 668.500000 | 1.000000 | 3.000000 | 38.000000 | 1.000000 | 0.000000 | 31.000000 |
| max | 891.000000 | 1.000000 | 3.000000 | 80.000000 | 8.000000 | 6.000000 | 512.329200 |

Is any of this useful? I suppose the huge range of means and standard deviations does tell me that some standardisation is going to be useful for several model types.\
What it does tell me is that several of the features are definitely not numeric data. 5 of them, to be exact. The only feature I want to use that isn't numeric is _cabin_. 

For the sake of it, I will see what `scipy.describe(data)` on the numerical features tells me:
 
| Statistical Attribute | Survived | Pclass | Age | SibSp | Parch | Fare |
| --- | --- | --- | --- | --- | --- | --- |
| min | 0 | 1 | nan | 0 | 0 | 0 |
| max | 1 | 3 | nan | 8 | 6 | 512.3292 |
| mean | 0.38383838 | 2.30864198 | nan | 0.52300786 | 0.38159371 | 32.20420797 |
| variance | 2.36772217e-01 | 6.99015120e-01 | nan | 1.21604308e+00 | 6.49728244e-01 | 2.46943685e+03 |
| skewness | 0.47771747 | -0.62948588 | nan | 3.68912768 | 2.74448674 | 4.77925329 |
| kurtosis | -1.77178602 | -1.27956968 | nan | 17.77351175 | 9.71661324 | 33.20428925 |

I found two interesting things out while getting this statistical data: 
1. Cabin is so sparse, and of such a meaningless format, that is is useless, and so I will leave it out of the analysis. 
2. Age is missing a lot of values (177 out of 891. 19.9%, to be exact).  

Therefore, I extracted age, removed the NaNs completely, and ran the analysis again on age:
 
| Statistical Attribute | Age |
| --- | --- |
| min | 0.41999999999999998 | 
| max | 80.0 |
| mean | 29.69911764705882 |
| variance | 211.01912474630805 |
| skewness | 0.3882898514698657 |
| kurtosis | 0.16863657224286044 |

Altogether, this shows me that there is not an extreme amount of skewedness, nor is there much of a kurtosis issue except in two cases: _SibSp_; and _Fare_.

Taking a graphical look at _SibSp_ and _Fare_:\
Graph 1: SibSp Histogram\
![SibSp Histogram](https://raw.githubusercontent.com/chorhatarahuduketuri/MachineLearningProjects/master/Kaggle/secondAttempt/titanic/images/SibSp_histogram.png "SibSp Histogram")\
Obviously the tail shown on this graph is relatively long, given that most of the data (>600 of 981 points) is 0. This explains the kurtosis.\
Graph 2: Fare Histogram\
![Fare Histogram](https://raw.githubusercontent.com/chorhatarahuduketuri/MachineLearningProjects/master/Kaggle/secondAttempt/titanic/images/Fare_histogram.png "Fare Histogram")\
The same with this distribution, where the tail is even longer as where the mean is 32.2, some people paid 512.3. 

###### Filling in the blanks 
I can see that the _Sex_ data needs to be converted into something numerical so that the computer can read it. I can also see that there are a lot of missing data points for _Age_.\
Making text categorical data into numerical data is easy: there's a function call for that. Dealing with the missing _Age_ data is more difficult. Not how to do something - that's just more code. The difficult part is deciding what to do.

There are two options here: either set it to 0, or set it to the average (but which average; mean, median, or mode?)\
Lets look at the _Age_ data first: 
Graph 2: Age Histogram\
![Age Histogram](https://raw.githubusercontent.com/chorhatarahuduketuri/MachineLearningProjects/master/Kaggle/secondAttempt/titanic/images/Age_histogram.png "Age Histogram")\
The lack of outlier data in this graph, it's low skew and kurtosis, combined with it's _mostly_ normal distribution appearance, leads me to think that mean substitution isn't a bad idea, except for the pollution of the dataset that it causes. However, 0 substitution is also valid, because there are no other 0 values in _Age_, which means it isn't polluting. This also allows a model to hypothetically discount the _Age_ feature if it is 0. I won't remove the rows with no _Age_ value because they make up nearly 20% of the dataset. I could always put -1 in? 

If I'm going to take this seriously, and I'm unable (or inexperienced such that I'm unable) to determine which of mean substitution and 0 substitution is best, then I'll do both, and run all the learning models on both, and submit prediction results for testing from both approaches, and learn which way which is best in this case.

###### Getting the data ready for training
Having decided on what data to use and how to make it usable, I should get on and do that. 

There will be 2 datasets, one called training_data_0 and one called training_data_mean. training_data_0 will have its age NaNs substituted with 0s, training_data_mean will have its age NaNs substituted with the mean of the non-NaN values of the age feature. \
Testing data will have to be prepared in the same way. I checked and the training_age_mean is 29.699, while the test_age_mean is 30.273. \
Now that the data is prepared, lets move on to step 4. 

### Step Four 
'Consider what models might be appropriate to train on the datasets, as well as understand which sorts of algorithms will work and which will not. Feature engineering should also be undertaken at this stage, in the case of any selected model that would benefit from or require it.'

By default, I will mean normalise my datasets.\
From previous work, I am interested in using logistic regression (because it will give the best results), and neural networks (because they're cool). \
I'm also interested in 2nd order polynomial feature engineering. 

I've split each of the training sets into training/validation sets using `sklearn.model_selection.test_train_split`, so that I can check the effectiveness of each approach. \
I'm also mean normalising the training data, then creating 2nd order polynomials from the _original, non-standardised_ training data, and _then_ mean normalising that. 

### Step Five
'Design, create, and/or train the model'

#### Logistic Regression
I will create 4 logistic regression models for each of the original/polynomial and 0/mean substitution training datasets.\
I made 4 identical logistic regression models as per the previous line. They had `max_iter` set to 500, and `solver` set to lbfgs.\
They've been trained using the datasets prepared. 

#### Artificial Neural Network
I will create 4 neural network models for each of the original/polynomial and 0/mean substitution training datasets.\
I made 4 identical neural network models as per the previous line. They had `hidden_layer_sizes` set to `100, 50, 25, 12'` (I expect I'll have to update these later), and `activation` set to logistic.\
I trained them with the same datasets as the logistic regression models. 

### Step Six - model evaluation
'Evaluation of model on validation set'

To be able to evaluate the performance of my models, I'm using my trained models to make predictions on the validation set I set aside earlier.\
To calculate their performance I'm using `sklearn.metrics.accuracy_score`, which tells me the proportion of the validation set that is accurately predicted, and `sklearn.metrics.classification_report`, which tells me the precision, recall, f1 score, and support for the validation set predictions. I don't actually know what support is. Apparently it's the 'number of samples of the true response that lie in that class'. Which is not super clear at this point.\
Oh OK having actually looked at my own results I now get it. The support is the number of times that class was predicted correctly.

##### Interpretation
So, the accuracy for all 8 classifiers is ~0.6~0.7, with occasional variations outside of that, but not by much.\
The f1-scores are ~0.5~0.8, which in itself is awfully low, and is also spread across a worryingly large range.\
Clearly I need to do some improvement.
 
* **Question**: Is this result really bad?\
**Answer**: _Yes why isn't its f1-score 0.95+?_ 
* **Question**: Can I get data from the models on how they learned?\
**Answer**: _I don't know - I'll have to look into that._
* **Question**: Can I make a graph of that data?\
**Answer**: _I most certainly hope so!_
* **Question**: What can I do to improve the models performance?\
**Answer**: _Well that depends on what's wrong with it. Although I suspect that lots of extra neurons will be part of helping the ANNs._

What to do, what to do. 
Well, I've already scaled and normalised my features. 
The class imbalance is not too great (only about 2:1, I would consider it imbalanced enough to require intervention at 10:1). 
All the scores are rubbish, so I can't just swap one for another (plus, the main objective is accuracy).
I could use something like grid search to tune the hyperparameters of my models. That sounds like it could have some positive effect, plus 'hyperparameters' is a cool word. 

### Step Seven - hyperparameter tuning
'7. Hyperparameter turning (improving model performance).' 

OK, so, there are a lot of hyperparameters and I'm not doing all this by hand. What options are there? 
Well firstly, lets look at what hyperparameters there are:\
Logistic Regression:  
* penalty 
* stopping criteria tolerance (tol)
* C
* solver
* maximum iterations (max_iter)

Neural Networks:
* hidden layers (hidden_layer_sizes) 
* activation
* solver
* alpha
* learning rate (learning_rate)
* learning rate initial value (learning_rate_init)
* stopping criteria tolerance (tol)

That's a lot of stuff. I am _definitely_ not doing that by hand. \
Cue, `GridSearchCV`. A magical bit of code, and part of, `sklearn.model_selection`. It takes an estimator, and some parameters, and runs the estimator with all the combinations of the provided parameters. Neat.

Which parameters to have `GridSearchCV` search over...  why not all? Because I don't have enough RAM is why not all. So lets cut that down to something more interesting anyway.\
Logistic Regression:
* `C`; in steps of 0.1, from 0.1 to 4.0
* `solvers`; liblinear and lbfgs
* `tol`; in steps of 0.00001, from 0.00001 to 0.004

Neural Networks;
* `hidden_layer_sizes`; (100, 50, 25, 12) and (50, 40, 30, 20, 10, 5)
* `alpha`; in steps of 0.00001, from 0.00001 to 0.0004
* `learning_rate`; adaptive and constant
* `learning_rate_init`; in steps of 0.0001, from 0.0001 to 0.004

All of these I get, and know are relevant to improving learning. The only one I don't really know a lot about is `hidden_layer_sizes`, since I'm no expert on ANN topology, except that MOAR IS BETTA. This is a test to see if a larger number of neurons arranged in a smaller number of larger layers, is better than a smaller number of neurons, arranged in a larger number of smaller layers.

Currently neither of my computers have finished calculating any of this, and some have been at it for what we will politely term a while. The one I'm writing this on (faster CPU, less RAM) suffered a segfault near the end of the last run. I'm trying again and we'll see what happens.

OK, so, they finished, after some time. One took 255 minutes, another took 268, and a reduced form took 90. \

The results of the two full runs are as follows:

##### Logistic Regression
| Dataset | Score | Tol | Solver | C |
| --- | --- | --- | --- | --- |
| Run 1 - 0 1 | 0.6838 | 0.00001 | liblinear | 0.1 |
| Run 2 - 0 1 | 0.7047 | 0.00001 | liblinear | 0.7 |
| Run 1 - mean 1 | 0.7030 | 0.00001 | liblinear | 0.8 |
| Run 2 - mean 1 | 0.7030 | 0.00001 | lbfgs | 0.1 |
| Run 1 - 0 poly | 0.7030 | 0.00001 | lbfgs | 3.8 |
| Run 2 - 0 poly | 0.7287 | 0.00001 | lbfgs | 3.6 |
| Run 1 - mean poly | 0.7287 | 0.00001 | liblinear | 0.5 |
| Run 2 - mean poly | 0.7175 | 0.00001 | liblinear | 0.4 |

##### Artificial Neural Networks
| Dataset | Score | Learning Rate | Learning Rate Init | Alpha | Architecture | 
| --- | --- | --- | --- | --- | --- |
| Run 1 - 0 1 | 0.6501 | constant | 0.0001 | 0.00024 | (100, 50, 25, 12) |
| Run 2 - 0 1 | 0.6613 | constant | 0.00130 | 0.00004 | (100, 50, 25, 12) |
| Run 1 - mean 1 | 0.6597 | constant | 0.00301 | 0.00017 | (100, 50, 25, 12) |
| Run 2 - mean 1 | 0.6437 | constant | 0.0015 | 0.00011 | (100, 50, 25, 12) |
| Run 1 - 0 poly | 0.6661 | adaptive | 0.00390 | 0.00027 | (100, 50, 25, 12) |
| Run 2 - 0 poly | 0.6597 | adaptive | 0.0016 | 0.00015 | (100, 50, 25, 12) |
| Run 1 - mean poly | 0.6709 | constant | 0.0019 | 0.00001 | (100, 50, 25, 12) |
| Run 2 - mean poly | 0.6613 | constant | 0.0013 | 0.00003 | (100, 50, 25, 12) |

Wow that was pointless. Basically, logistic regression is better than neural networks regardless of the dataset. In fact, the _worst_ logistic regression model is better than the _best_ neural network model.\
Enough of this I'm trying some other model type. 

##### Other models
 * Linear Support Vector Classifier (SVC)
 * KNeighbors Classifier
 * Support Vector Classifier
 * Ensemble Classifiers

Before we go getting extreme and using ensemble methods, lets just think about this a minute. \
For starters, this data is obviously not cleanly linearly separable, so linear SVC is straight out the window. KNeighbors classifier sounds interesting, as does the ever-useless SVC (with grid search to configure it properly). Ensemble classifiers can be dragged out if the others fail to get >90% accuracy and F1-Scores. Yea, that sounds like a reasonable plan.

#### KNeighbors Classifier
`sklearn.neighbors.KNeighborsClassifier` has only one meaningful parameter for the grid search: n_neighbors. The other parameters are all relevant to much larger datasets that take a lot more computation time. 

| Dataset | Score | n_neighbors |
| --- | --- | --- |
| Run 1 - 0 1 | 0.7191 | 23 |
| Run 2 - 0 1 | 0.7239 | 46 |
| Run 3 - 0 1 | 0.7303 | 20 |
| Run 4 - 0 1 | 0.7303 | 22 |
| Run 1 - mean 1 | 0.7175 | 33 |
| Run 2 - mean 1 | 0.7191 | 18 |
| Run 3 - mean 1 | 0.7191 | 24 |
| Run 4 - mean 1 | 0.7271 | 9 |
| Run 1 - 0 poly | 0.7239 | 29 |
| Run 2 - 0 poly | 0.7191 | 12 |
| Run 3 - 0 poly | 0.7287 | 12 |
| Run 4 - 0 poly | 0.7191 | 21 |
| Run 1 - mean poly | 0.7207 | 35 |
| Run 2 - mean poly | 0.7271 | 45 |
| Run 3 - mean poly | 0.7207 | 32 |
| Run 4 - mean poly | 0.7319 | 27 |

Well that's not much better than linear regression. Never mind onto SVC: 

#### Support Vector Classifier
`sklearn.svm.SVC` has 5 meaningful parameters: C, kernel, degree (when kernel='poly', shrinking, tol.\
So the original range of values on the parameters resulted in 35636436 fits, of which 59032 were done in 3.2 minutes. So that's ((35636436/59032)*(60*3.2)) = 115906 seconds, which is 32.2 hours. I'll try another arrangement of parameters. 

| Dataset | Score | C | Kernel | Tol |
| --- | --- | --- | --- | --- |
| Run 1 - 0 1 | 0.6886 | 2.565 | rbf | 0.0001 |
| Run 2 - 0 1 | 0.6886 | 2.565 | rbf | 0.0001 |
| Run 1 - mean 1 | 0.7352 | 0.766 | rbf | 0.0001 |
| Run 2 - mean 1 | 0.7352 | 0.766 | rbf | 0.0001 |
| Run 1 - 0 poly | 0.6918 | 0.669 | rbf | 0.0001 |
| Run 2 - 0 poly | 0.6918 | 0.669 | rbf | 0.0001 |
| Run 1 - mean poly | 0.7287 | 0.285 | rbf | 0.0001 |
| Run 2 - mean poly | 0.7287 | 0.285 | rbf | 0.0001 |

Both runs took just over 17 minutes, and both got exceedingly similar results. 

At this point it's becoming annoying that I've got nothing better than 0.7352, though it is interesting that the least linear thing I've applied (SVM with RBF kernel) is the most accurate.\
I'm going to look at the approaches of others to see what they've achieved.\
(Note that I'm just blithely trying algorithm after algorithm and pretending grid searches of parameters are actually a proper analysis procedure. I'm fairly sure it's not. I'll end this soon and do something proper next.)


#### Decision Tree Classifier
OK so the next library function I'm gonna throw at this is `sklearn.tree.DecisionTreeClassifier`, because it looks interesting, and also because basically I looked at the leaderboard on Kaggle for a bit and found that tree models seemed to do well (ensemble models did the best).\
Tree models are essentially the computer making a '20 questions' game out of classifying samples, which is cool because a human can actually understand a 20 questions tree (unlike a neural network which is incomprehensible), and it can deal with categorical data in addition to numerical data.\
I'm going to leave most of the parameters on default for this, and concentrate more on the data preparation, as it wants something different to what the numerical-only algorithms wanted. 

In a single run that took 3 minutes, the two DTCs run came back with 75.6019% and 74.9599% accuracy. This looks promising. 

OK, so after messing around with code for a while, looking at the results, and then hand-rolling something in the cli (`splitter='random', max_depth=6, min_samples_split=10, min_samples_leaf=10, max_features=5`), I got something that I locally validated as 75.0% accurate. On the Kaggle scoreboard this came out as 0.69856, which is 69.856% accurate, which is rubbish. Basically I'm onto ensemble now.\
But first, I'm going to take a look at the picture I made, which displays some interesting things: 

Decision Tree Classifier: \
![SibSp Histogram](https://raw.githubusercontent.com/chorhatarahuduketuri/MachineLearningProjects/master/Kaggle/secondAttempt/titanic/images/sub_mean.png "Decision Tree Classifier")\
This is the first bit of learning that I've gotten a machine to do that you can actually see. And the first thing you see is that almost everyone in 3rd class dies. Actually only 10 of them survive. That's awful.\
After that, it's parents & children - if you're in that group you all survive (providing you were in 1st or 2nd class, that is). Take a look and see what you can determine from it. \

