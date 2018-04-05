There are 6 immediately interesting datasets in sklearn.datasets (based on my prior familiarity with it, or based on its name): 
1. Iris
2. Boston
3. Breast Cancer
4. Diabetes
5. Digits
6. Wine

For each, this is the initial interpretation of how to analyse them: 
* Iris: Supervised, Classification
* Boston: Supervised, Regression
* Breast Cancer: Supervised, Classification, ?Linearly inseparable?
* Diabetes: Supervised, Regression
* Digits: Supervised, Classification
* Wine: Supervised, Classification

Since classification is the most common, I will begin with the classification datasets; Iris, Breast Cancer, Digits, Wine. Arbitrarily, since I have written them out in this order, I will perform the analyses in this order.\
After the classifications are done, I will start on the regressions; Boston, Diabetes. Again, I will undertake these analyses in the order I happen to have written them down. 

For each of these, I will create some visualisations to describe the original datasets, then clean the data and train a model with it. After this, I will assess the accuracy of the model using a reserved test set of data, and attempt to visualise some aspect of the analysis that is useful for understanding what the model shows. 

**SciKit Learn**\
The two main things in scikit learn are _datasets_ and _estimators_. 
Datasets are data that you load, visualise, and clean & engineer them. 
Estimators are models that you train, test the accuracy of, and make predictions with. 


### Iris
##### Step one
There are nine steps (as I have identified so far) in any given machine learning project, the first being to 'define the problem/understand the business'. In this case, the problem is to identify different species of flowers based on measurements of their physical attributes. This problem is one of classification.\
##### Step two
The second step is to 'locate and acquire relevant datasets'. This is already done, the dataset is the famous Iris dataset.\
##### Step three
The third step is to 'perform an EDA (Exploratory Data Analysis) of the dataset'. I have this further defined as _'including some preliminary cleaning and perhpas feature engineering - to understand it's value and quality. Make sure the dataset is in some form that can be fed into a mathematical, algorithmically trained model'_. To that end, I will look at the description of the dataset that comes with it, load it into memory and look at what form the data comes in, and I will determine some statistical information about it, along with create some visualisations if I think they show something of interest.
###### EDA
Firstly, since this dataset is so famous, it has a wikipedia page which I'll read: https://en.wikipedia.org/wiki/Iris_flower_data_set\
From that, I've learnt that this multivariate dataset has three classes in it, with 50 samples each, for a total of 150 data points. The samples were mostly collected in the same place, at the same time, by the same person, using the same methods, which to me seems like a good way to minimise bias from sensors. There were four variables collected (besides the class/species), which were the length and width of the petals and and sepals of each flower. Other interesting/useful to know things are that it is not linearly separable, and that basic cluster analysis has poor performance on it, due to there only being two linearly separable clusters between the three classes. On the upside, this means it's a good test case for SVMs (Support Vector Machines).\
Now, I'm going to load it into memory and take a look at the data itself. 
The data is in the order of Sepal Length, Sepal Width, Petal Length and Petal Width along four columns, ordered with the first 50 being class zero (Setosa), the second fifty being class one (Versicolour), and the last 50 being class two (Virginica).



  