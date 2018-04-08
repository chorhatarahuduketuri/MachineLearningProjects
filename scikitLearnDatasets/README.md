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

Using the `sklearn.stats` package, I ascertained the following:

thing | Sepal length (cm) | Sepal width (cm) | Petal length (cm) | Petal width (cm)
 --- | --- | --- | --- | --- 
Minimum | 4.3 | 2.0 | 1.0 | 0.1
Maximum | 7.9 | 4.4 | 6.9 | 2.5
Mean | 5.84 | 3.05 | 3.76 | 1.20
Variance | 0.69 | 0.19 |  3.11 |  0.58
Skewness | 0.31 | 0.33 | -0.27 | -0.10
Kurtosis | 0.57 | 0.24 | -1.40 | -1.34

The min/max/mean information is not really very interesting, as they're all within a small range of orders of magnitude, so shouldn't cause any problems for the algorithms. \
Skewness is a measurement of which side of the mean most of the data points lie. If most of the data points are higher than the mean, then it has a negative skew. Conversly, if most of the data points are lower than the mean, it has a positive skew. Skewness is calculated by subtracting the median from the mean, then dividing by the standard deviation. \
Kurtosis describes the tailedness of a distribution of data points, where a positive value indicates that a lot of the data points exist in the tails of a distribution, whereas a negative value describes a distribution with the minority of the data points in the tails. High kurtosis scores can indicate a problem with outliers. Kurtosis is calculated in a variety of complicated ways, none of which I'm going to go into here, since I'm more interested in knowing if it means I have to do something to get a good model, rather than exactly how it works. \
Since these two things basically describe how normal a distribution of data points is, and most algorithms work better with more normal distributions, I will come back to this if I fail to get good results later, otherwise I'm not really interested in them at this point.

In an effort to better understand the data, I will graph all three classes of each of the four features on histograms, to better compare the data relative to each other and spot any obvious features of the dataset.\
Graph 1: Sepal length\    
![Sepal Length Histogram](https://raw.githubusercontent.com/chorhatarahuduketuri/MachineLearningProjects/master/scikitLearnDatasets/sepalLength.png "Sepal Length Histogram")
The sepal lengths graph shows that all three species of flower have sepals of similar lengths, and I can see why a classification algorithm might give so-so results based on this feature alone. \
Graph 2: Sepal width\
![Sepal Width Histogram](https://raw.githubusercontent.com/chorhatarahuduketuri/MachineLearningProjects/master/scikitLearnDatasets/sepalWidth.png "Sepal Width Histogram")
The sepal widths graph shows that Versicolor and Virginica are very similar, and shows why a clustering algorithm might not be able to easily separate them based on this feature. \
Graph 3: Petal length\
![Petal Length Histogram](https://raw.githubusercontent.com/chorhatarahuduketuri/MachineLearningProjects/master/scikitLearnDatasets/petalLength.png "Petal Length Histogram")
The petal length graph is much easier to tell the three species apart on, especially the Setosa which stands distinctly apart from the other two, which overlap slightly. \
Graph 4: Petal width\
![Petal Width Histogram](https://raw.githubusercontent.com/chorhatarahuduketuri/MachineLearningProjects/master/scikitLearnDatasets/petalWidth.png "Petal Width Histogram")
The petal width graph is much like the petal length histogram, in that the Versicolor and Viriginica species overlap slightly, but the Setosa is cleanly separated to the lower end of the range.

