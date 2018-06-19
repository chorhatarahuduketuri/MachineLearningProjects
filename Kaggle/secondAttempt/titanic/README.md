# The Titanic

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