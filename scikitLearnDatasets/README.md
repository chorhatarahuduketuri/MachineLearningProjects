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

Using the `scipy.stats` package, I ascertained the following:

Statistical Attribute | Sepal length (cm) | Sepal width (cm) | Petal length (cm) | Petal width (cm)
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
![Sepal Length Histogram](https://raw.githubusercontent.com/chorhatarahuduketuri/MachineLearningProjects/master/scikitLearnDatasets/irisImages/sepalLength.png "Sepal Length Histogram")
The sepal lengths graph shows that all three species of flower have sepals of similar lengths, and I can see why a classification algorithm might give so-so results based on this feature alone. \
Graph 2: Sepal width\
![Sepal Width Histogram](https://raw.githubusercontent.com/chorhatarahuduketuri/MachineLearningProjects/master/scikitLearnDatasets/irisImages/sepalWidth.png "Sepal Width Histogram")
The sepal widths graph shows that Versicolor and Virginica are very similar, and shows why a clustering algorithm might not be able to easily separate them based on this feature. \
Graph 3: Petal length\
![Petal Length Histogram](https://raw.githubusercontent.com/chorhatarahuduketuri/MachineLearningProjects/master/scikitLearnDatasets/irisImages/petalLength.png "Petal Length Histogram")
The petal length graph is much easier to tell the three species apart on, especially the Setosa which stands distinctly apart from the other two, which overlap slightly. \
Graph 4: Petal width\
![Petal Width Histogram](https://raw.githubusercontent.com/chorhatarahuduketuri/MachineLearningProjects/master/scikitLearnDatasets/irisImages/petalWidth.png "Petal Width Histogram")
The petal width graph is much like the petal length histogram, in that the Versicolor and Viriginica species overlap slightly, but the Setosa is cleanly separated to the lower end of the range.

##### Step four
The fourth step is to 'consider what sorts of models would be appropriate, as well as understand which sorts of algorithms will work and which will not. Feature engineering should also be undertaken at this stage, in the case of any selected model that would benefit from or require it.'\
Obviously a supervised classification algorithm will be necessary, since the objective is classification and there is target data. Naively, I would assume logistic regression would be suitable, perhaps with polynomial features being necessary, however from reading the Wikipedia page of its history, I know that it is an excellent example of a non-linearly separable dataset, so the performance of logistic regression, even with polynomial feature engineering, may be poor. As a result of this, in addition to the use of logistic regression, I will use support vector machines to see how that model compares.\
As part of step four, I will standardise the data, as well as generate several polynomial features in case they are necessary for any of the algorithms I end up using.\
I will use training feature sets containing up to 1st, 2nd, and 3rd degree polynomial features. 

##### Step five 
'Design, create, and/or train the model'
###### Logistic Regression
After creating three separate logistic regression models, all of which use the L-BFGS algorithm, a one-versus-rest multiclass strategy, and train for up to 500 iterations, I trained all three models on the 1st, 2nd, and 3rd degree polynomial training sets.\

###### Support Vector Machines
All the data preparation having already been done for logistic regression, implementing straight forward support vector machines was simple.\

##### Step six
###### Logistic Regression 
All three models attain accuracy scores in the 0.9 - 1.0 range, with f1 scores in the same range. This suggests that linear regression is very good for this particular application.

###### Support Vector Machines
The accuracy of 1st degree polynomial features (basically, the original data, unaltered), is near perfect. Accuracy with 2nd and 3rd degree polynomial feature engineering, ~~is very poor. As bad as as 0.51 in some cases.~~ is, now I've corrected the code to use the mean normalised training data, actually really good, around 0.02 better than without any mean normalisation. The F1 scores are similarly improved. These numbers are all still around 0.90 to 0.95, but it's a lot better than when the code was wrong. 

##### Step seven
'Hyperparameter turning (improving model performance). Algorithm tuning, ensemble methods.'\
This is unnecessary, as accuracy with both linear regression and SVM is near perfect. 

##### Step eight
'Prediction: make actual predictions on actual data and test it's real world performance.'\
Since this is only for training, and I have used the entire dataset, this step seems unnecessary and unfeasible.\
'Presentation: present to the stakeholder/business the results of the work so far and explain the future worth (or lack thereof).'
This writeup is my presentation of my work and what I have achieved. It has been instructive and encouraging, in that actually having a framework and a step-by-step plan, even one as generalistic as this one, is extremely helpful in getting the project completed without getting lost among the seemingly infinite details and options of machine learning.

##### Step nine
'Deployment to a production environment.'\
As with the first part of step eight, I deem this step unnecessary, due to this being a training project with no useful real world application that I can put it to at this time. Hopefully, after my first six training projects, I will be able to attempt to install some trained model in some real world system of some kind, even if contrived and not really practically useful, just for the experience of having done so. 

### Breast Cancer
##### Step one
'Define the problem/understand the business'\
In this case, the business problem is accurately predicting weather or not a tumour found in the breast of a human is malignant or benign. Target data is given. The dataset given is a series of numerical measurements about the tumour itself. I will state here that I have very little to no idea what most of them mean, as I am not medically trained past First Aid at Work (in an office environment). \
This is a supervised classification problem. 

##### Step two
'Locate and acquire relevant datasets'\
This dataset is available via the import of `sklearn.datasets`. 

##### Step three
'Perform an EDA of the datasets - including some preliminary cleaning and perhaps feature engineering - to understand it's value and quality. Make sure the dataset is in some form that can be fed into a mathematical, algorithmically trained model'

###### EDA
Wikipedia doesn't have an entry on this dataset like it did for the Iris dataset, but the description that comes with it in the code is interesting.\
There are 569 data points, each with 30 numeric, predictive attributes, and one target attribute that describes the class, where 0 indicates a malignant tumour, and 1 indicates a benign tumour.\
There were only ten measurements made about each tumour: 
- radius (mean of distances from center to points on the perimeter)
- texture (standard deviation of gray-scale values)
- perimeter
- area
- smoothness (local variation in radius lengths)
- compactness (perimeter^2 / area - 1.0)
- concavity (severity of concave portions of the contour)
- concave points (number of concave portions of the contour)
- symmetry
- fractal dimension ("coastline approximation" - 1)

I had naively assumed that these features described aspects of the tumours themselves, but they in fact describe characteristics of the cell nuclei present in the image. They were computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. I have for years (I first encountered this during my academic education at least 6 years ago) lived with the false belief that this dataset described tumour characteristics, not the characteristics of nuclei from cells in the tumour. \
Due to these readings being taken from 2D image data (a medical scan image), and those images not being perfect themselves, nor of perfectly uniform, symmetrical objects, 30 features were derived from statistics calculated from measurements of those 2D images.\
For example, the images of the nuclei aren't perfect circles, so the radius is measured several times at different orientations, and the mean is taken. The standard error of those readings is also included, along with the 'worst' reading, which is actually the mean of the largest three taken for that feature for that data point.\
The first ten features are all the means, the second ten are the standard errors, and the third ten features are the worst.\
I have no idea what unit of measurement these readings were taken in. 

| Statistical Attribute | Mean Radius | Mean Texture | Mean Perimeter | Mean Area | Mean Smoothness | Mean Compactness | Mean Concavity | Mean Concave Points | Mean Symmetry | Mean Fractal Dimension | Radius Error | Texture Error | Perimeter Error | Area Error | Smoothness Error | Compactness Error | Concavity Error | Concave Points Error | Symmetry Error | Fractal Dimension Error | Worst Radius | Worst Texture | Worst Perimeter | Worst Area | Worst Smoothness | Worst Compactness | Worst Concavity | Worst Concave Points | Worst Symmetry | Worst Fractal Dimension |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Min | 6.98100000e+00 |   9.71000000e+00 |   4.37900000e+01 |         1.43500000e+02 |   5.26300000e-02 |   1.93800000e-02 |         0.00000000e+00 |   0.00000000e+00 |   1.06000000e-01 |         4.99600000e-02 |   1.11500000e-01 |   3.60200000e-01 |         7.57000000e-01 |   6.80200000e+00 |   1.71300000e-03 |         2.25200000e-03 |   0.00000000e+00 |   0.00000000e+00 |         7.88200000e-03 |   8.94800000e-04 |   7.93000000e+00 |         1.20200000e+01 |   5.04100000e+01 |   1.85200000e+02 |         7.11700000e-02 |   2.72900000e-02 |   0.00000000e+00 |         0.00000000e+00 |   1.56500000e-01 |   5.50400000e-02 |
| Max | 2.81100000e+01 |   3.92800000e+01 |   1.88500000e+02 |         2.50100000e+03 |   1.63400000e-01 |   3.45400000e-01 |         4.26800000e-01 |   2.01200000e-01 |   3.04000000e-01 |         9.74400000e-02 |   2.87300000e+00 |   4.88500000e+00 |         2.19800000e+01 |   5.42200000e+02 |   3.11300000e-02 |         1.35400000e-01 |   3.96000000e-01 |   5.27900000e-02 |         7.89500000e-02 |   2.98400000e-02 |   3.60400000e+01 |         4.95400000e+01 |   2.51200000e+02 |   4.25400000e+03 |         2.22600000e-01 |   1.05800000e+00 |   1.25200000e+00 |         2.91000000e-01 |   6.63800000e-01 |   2.07500000e-01 |
| Mean |  1.41272917e+01 |   1.92896485e+01 |   9.19690334e+01 |         6.54889104e+02 |   9.63602812e-02 |   1.04340984e-01 |         8.87993158e-02 |   4.89191459e-02 |   1.81161863e-01 |         6.27976098e-02 |   4.05172056e-01 |   1.21685343e+00 |         2.86605923e+00 |   4.03370791e+01 |   7.04097891e-03 |         2.54781388e-02 |   3.18937163e-02 |   1.17961371e-02 |         2.05422988e-02 |   3.79490387e-03 |   1.62691898e+01 |         2.56772232e+01 |   1.07261213e+02 |   8.80583128e+02 |         1.32368594e-01 |   2.54265044e-01 |   2.72188483e-01 |         1.14606223e-01 |   2.90075571e-01 |   8.39458172e-02 |
| Variance |  1.24189201e+01 |   1.84989087e+01 |   5.90440480e+02 |         1.23843554e+05 |   1.97799700e-04 |   2.78918740e-03 |         6.35524790e-03 |   1.50566077e-03 |   7.51542821e-04 |         4.98487228e-05 |   7.69023519e-02 |   3.04315949e-01 |         4.08789584e+00 |   2.06943158e+03 |   9.01511400e-06 |         3.20702887e-04 |   9.11198238e-04 |   3.80724191e-05 |         6.83328983e-05 |   7.00169156e-06 |   2.33602242e+01 |         3.77764828e+01 |   1.12913085e+03 |   3.24167385e+05 |         5.21319833e-04 |   2.47547707e-02 |   4.35240905e-02 |         4.32074068e-03 |   3.82758354e-03 |   3.26209378e-04 |
| Skewness | 0.93989345 |  0.64873357 |  0.98803695 |  1.64139051 |  0.45511992 |        1.18698332 |  1.39748324 |  1.16809035 |  0.72369472 |  1.30104739 |        3.08046399 |  1.64210026 |  3.43453047 |  5.43281586 |  2.30834422 |        1.89720239 |  5.09698095 |  1.44086689 |  2.18934184 |  3.91361665 |        1.10020504 |  0.49700667 |  1.12518762 |  1.85446799 |  0.41433005 |        1.46966746 |  1.14720234 |  0.49131594 |  1.43014487 |  1.65819316 |
| Kurtosis |  0.82758367 |   0.74114542 |   0.95316505 |   3.60976126 |         0.83794535 |   1.62513952 |   1.97059165 |   1.04668022 |         1.26611697 |   2.9690169  |  17.52116219 |   5.29175289 |        21.20377481 |  48.76719561 |  10.3675372  |   5.05096602 |        48.42256209 |   5.07083973 |   7.81638799 |  26.03994977 |         0.9252876  |   0.21180938 |   1.05024268 |   4.3473308  |         0.50275975 |   3.00212021 |   1.59056807 |  -0.54136707 |         4.39507329 |   5.18811128 |

The Skewness and Kurtosis measurements range from very small values (0.21) to very large values (48.77), and this suggests that the data should be mean normalised, to alter its 'shape' into something closer to a normal distribution. The min/max and mean differences also vary widely, by several orders of magnitude in some cases, which further suggests that most algorithms (certainly any that I'm going to use here) would benefit from the data being rearranged into a more symmetrical, normal shape. 

To further explore the data, I will create some graphs to see if there is anything of particular interest to be found by doing so. \
The first thing I've done is create straight forward histograms of each feature over the entire dataset, to view the skewness and kurtosis. The following selection of graphs show some of of the distributions I found interesting. 

###### Mean graphs
Graph 1: Mean area\
![Mean Area](https://raw.githubusercontent.com/chorhatarahuduketuri/MachineLearningProjects/master/scikitLearnDatasets/breastCancerImages/hist_mean_area.png "Mean Area Histogram")
As you can see from this histogram of the mean area of the nuclei, the data is somewhat negatively skewed. I think it is a better way to put it to say, that the _mean_ of the data is negatively skewed, in that the mean is lower than the median.\
Graph 2: Mean concave points\
![Mean Concave Points](https://raw.githubusercontent.com/chorhatarahuduketuri/MachineLearningProjects/master/scikitLearnDatasets/breastCancerImages/hist_mean_concave_points.png "Mean Concave Points Histogram")
The mean concave points are even more negatively skewed than the mean area.\
Graph 3: Mean smoothness\
![Mean Smoothness](https://raw.githubusercontent.com/chorhatarahuduketuri/MachineLearningProjects/master/scikitLearnDatasets/breastCancerImages/hist_mean_smoothness.png "Mean Smoothness Histogram")
The mean smoothness has much less skew, and so looks more similar to a normal distribution. It has a very similar kurtosis score to the mean area, which indicates that as similar proportions of the data points lie in the centre of the bell curve, as lie in the tails, in both distributions. \
Remember that a higher kurtosis score indicates that a greater proportion of the data is in the centre of the bell curve, and a lower score indicates the opposite, such that the height of the bell would be lower, and the tails would be longer. 

###### Error graphs
Graph 4: Area error\
![Area Error Histogram](https://raw.githubusercontent.com/chorhatarahuduketuri/MachineLearningProjects/master/scikitLearnDatasets/breastCancerImages/hist_area_error.png "Area Error Histogram")
As you can see from the graph, the area error not only has a strongly negative skew, but also a very high kurtosis score (48.77).\
Graph 5: Concave points error\
![Concave Points Error Histogram](https://raw.githubusercontent.com/chorhatarahuduketuri/MachineLearningProjects/master/scikitLearnDatasets/breastCancerImages/hist_concave_points_error.png "Concave Points Error Histogram")
The concave points error data, while having some of the lowest scores of the error features, still has significant skew and kurtosis scores. 

###### Worst graphs
Graph 6: Worst area\
![Worst Area Histogram](https://raw.githubusercontent.com/chorhatarahuduketuri/MachineLearningProjects/master/scikitLearnDatasets/breastCancerImages/hist_worst_area.png "Worst Area Histogram")
The worst area data shows significant skew and high kurtosis (though still much lower than area error), and learning algorithms could benefit from this data being mean normalised before being fed into them. \
Graph 7: Worst smoothness\
![Worst Smoothness Histogram](https://raw.githubusercontent.com/chorhatarahuduketuri/MachineLearningProjects/master/scikitLearnDatasets/breastCancerImages/hist_worst_smoothness.png "Worst Smoothness Histogram")
Having some of the lowest scores of the worst data features, the smoothness has both low skew and low kurtosis. 

I have considered further graphical exploration or other analysis of the data prior to model selection, and I will not do any more. This is for two reasons. The first is that I wish to keep these analyses simple and short, so that I gain experience in starting ML projects, carrying them out, and _finishing_ them. The second is that I do not believe any further analysis will cause me to change anything I think I am likely to do going forward during this project. 

##### Step four
The fourth step is to 'consider what sorts of models would be appropriate, as well as understand which sorts of algorithms will work and which will not. Feature engineering should also be undertaken at this stage, in the case of any selected model that would benefit from or require it.'\
Given that this is a classification problem with 596 labelled data points I should use linear SVC, according to scikit learn's algorithm selection cheat-sheet.\
I will use linear SVC. However, due to the ease of implementation, as in the Iris project I will use logistic regression as well.\
I will also mean normalise the training data now, and split it into 70% training data, and 30% testing data.
As with the Iris project, I will also generate polynomial features, although only up to the 2nd degree (due to number of features), to see if this increases accuracy. 

##### Step five 
'Design, create, and/or train the model'
###### Logistic Regression
I created two logistic regression models, one for the unmodified data, and one for the 2nd degree polynomial data. 

###### Support Vector Machines
For the SVMs, I used the SVC model instead of the LinearSVC model. This is because I wanted to compare the effectiveness of the radial basis function 'RBF' kernel against that of the linear kernel. The RBF kernel is the default, and is the one used in the Iris project. SVC(kernel='linear') is equivalent to LinearSVC, so it simplifies the code to only import and use SVC, then specify the desired kernel in the parameters.

##### Step six
'Evaluation of model on Validation set'\
In both cases I used the `accuracy_score` and `classification_report` functions from the `sklearn.metrics` package. 
###### Logistic Regression 
The accuracy of logistic regression with 1st order features varied around the 0.97 mark, which is gratifyingly high. With 2nd order features, it was around 0.96.\
F1 scores are numerically similar to the accuracy scores, which is a good sign. 

###### Support Vector Machines
The accuracy of SVMs with linear kernels using 1st order features was consistently similar to logistic regression with 1st order features. With 2nd order features however, it varied between the same as logistic regression, and up to 0.1 lower.\
F1 scores are numerically similar to the accuracy scores, which is good, though they are always a little lower for 2nd order features. 
With the RBF kernel, the accuracy with 1st order features was usually as good as the linear classifiers with the same data, but I haven't seen it outperform them. With 2nd order features, it is consistently worse, by as much as 0.1.\
F1 scores are similar to the accuracy scores for 1st order features, but usually even lower than the accuracy scores for 2nd order features.

##### Step seven
'Hyperparameter turning (improving model performance). Algorithm tuning, ensemble methods.'\
I find this to be an unnecessary step in this project, as none of the classifiers used had particularly complex parameters specified.\
Additionally, since a satisfactorily high level of accuracy has shown to be achievable using logistic regression, I don't feel the need to improve on what has already been done.

##### Step eight
'Prediction: make actual predictions on actual data and test it's real world performance.'\
'Presentation: present to the stakeholder/business the results of the work so far and explain the future worth (or lack thereof).'\
This writeup is my presentation of my work and what has been achieved. It has been proven to be a positive progression from the Iris project, showing that my methodology is effective for getting projects _done_.

##### Step nine
'Deployment to a production environment.'\
I do not need to deploy the trained models from this to any production environment. Pushing this writeup to github could be considered to be deployment to production, if you wanted to stretch definitions somewhat.


### Digits
##### Step one
'Define the problem/understand the business'\
In this case, the business problem is the correct identification of which handwritten, numerical digit the system has been provided with a 64 (8x8) pixel image of.\
This is the first image recognition system I have worked on outside the Coursera Machine Learning course, which I used for theory revision and as an introduction to practice, last year. This is one of the most exciting of the 6 datasets I have chosen to work with for the first phase of my portfolio-proper.

##### Step two
'Locate and acquire relevant datasets'\
This dataset is available via the SciKit-Learn package `sklearn.datasets`. 

##### Step three
'Perform an EDA of the datasets - including some preliminary cleaning and perhaps feature engineering - to understand it's value and quality. Make sure the dataset is in some form that can be fed into a mathematical, algorithmically trained model'
I have no internet in the location in which I am currently working, so I have only what is included with the dataset in sklearns packages to go on.\
Reading the `.DESCR` of this dataset, I have learned that there are 5620 data samples in this dataset, where each has 64 attributes. Each attribute is a pixel in an 8-by-8 pixel greyscale image, where each pixel has an intensity in the range 0-16.\
These 8x8 pixel images are of handwritten numerical digits, from 0-9. Each data sample is the reduction of a 32x32 pixel image, where each nonoverlapping 4x4 square of pixels was reduced to one, which has the intensity that is the average of the 4 it is replacing. This was done to reduce dimensionality and reduce the effects of small distortions.\
The data was gathered from 43 people - the data from 30 form the training set, and data from the remaining 13 forms the test set. This means that it will be the first of the 6 datasets to have separate sources for the training and test data.

Looking at the data itself, something very different becomes apparent. There are only 1797 data samples, each with 64 features. There are also 1797 target labels. This is in contrast to the datasets own `.DESCR` description. How interesting. I shall have to be wary of inconsistent descriptions of datasets from now on.

###### EDA
I am unable to access Wikipedia, or any other external source of information, on this dataset at this time.\
There are 1797 data samples, each with 64 features that vary between 0 and 16 (inclusive). The total number of training data points is 115008, and there are 1797 target labels, which vary from 0 to 9. This brings the total number of data to 116805.

In this case, the majority of the data created by the statistical summary function is of no use. Additionally, the table is far too large to be of any manual use.\
Of use, is that skewness between different features (in this case, a feature is a pixel from the same location in different images), varies by as much as 3 orders of magnitude, kurtosis by up to 4. This suggest that mean normalisation is a good idea. 


###### Graphs
While I would normally (is the trend of normal established after only 2 previous data points?) create graphs to represent features of the dataset that are relevant to choices I have yet to make, in this case I do not believe that any are necessary, or of any use.\
Additionally, since I am not connected to the internet, I am unable to look up how to render the images themselves so that you could more intuitively understand what I am using as input data. 


##### Step four
The fourth step is to 'consider what sorts of models would be appropriate, as well as understand which sorts of algorithms will work and which will not. Feature engineering should also be undertaken at this stage, in the case of any selected model that would benefit from or require it.'\
Due to having all numerical, categorically labelled data, from which I want to be able to predict which distinct digit has been written, this is a supervised classification problem.\
As for appropriate models, I am going to start with the ever-reliable logistic regression. This I will run with 1st, 2nd, and 3rd degree generated polynomial training data.\
Since it has been close in its performance so far, and from a practical standpoint so easy to implement alongside logistic regression, I will also be using SVMs. Due to the exciting nature of this as an image recognition task, I will be using more different types of kernels, including linear, which I do not expect to perform best, RBF, which I do not feel confident predicting the effectiveness of, and Sigmoid, which as a matter of cognitive bias (due to my familiarity with it, rather than its suitability), I predict will perform best of the SVM kernels.\
Additionally, and since I used them on this dataset in the Coursera Machine Learning course, and because they are so famous for being better at this sort of thing, I will also be using the ever so exciting _Artificial Neural Network_. Imported though - I can't be bothered to write them from scratch again. Of the available activation functions, I will try out relu, because it is default, and logistic, again because of my familiarity with it (it is what I have hand coded before now). \
For the neural network, I will be also be inputting 1st, 2nd, and 3rd degree polynomial training data. I will be creating 1, 8, and 64 layer networks (I would create all sizes in between as well but this laptop is only so powerful).


While undertaking this step, I noticed that I had made an error of rigorousness in the breast cancer project, and the iris project as well. Upon creating standardised versions of my training and test data, I then utterly failed to use them. I will avoid making this mistake here. I will also go back and fix that mistake after I have finished this project. 

##### Step five
'Design, create, and/or train the model'
###### Logistic Regression
As planned, I created 3 logistic regression models. All 3 were the same apart from which dataset they took in.\

###### Support Vector Machines
As planned, I created 9 support vector machine classifiers.\
3 used linear kernels, 3 used RBF kernels, and 3 used sigmoid kernels. For each kernel type, one classifier was trained on each of the pre-processed training datasets.\

###### Artificial Neural Networks
As planned, I created 6 ANNs, named Multi-layer perceptrons in scikit learn. 3 for the relu activation function and 3 for logistic. 1 of each of those was trained on the 1st, 2nd, and 3rd degree polynomial pre-processed training datasets.\
Not yet being an expert, I decided to just use the default network architecture for now. The default architecture is 1 layer of 100 neurons. Technically, this is not deep learning, which requires 2 or more hidden layers, but I don't think that sort of thing will be necessary here.\

##### Step six
'Evaluation of model on Validation set'\
###### Logistic Regression 
The accuracy of logistic regression was above 0.95 for 1st order features, and identically above 0.98 for 2nd and 3rd order features.\
I am impressed and disappointed that logistic regression performed the best. I was hoping to see ANNs be magically better, although I'm not sure that it means so much with these tiny differences on such a small dataset.\
The F1 scores are the same as the accuracy scores for each degree of engineered training data, which is impressive and pleasing.

###### Support Vector Machines
The accuracy of SVMs was universally above 0.9, but there was significant variation within the remaining window.\
Linear kernels actually performed the best, with 1st order features reaching 0.97, and 2nd & 3rd order features reaching 0.989 and 0.985 respectively.\
RBF kernels had the most variation: 1st got 0.97, 2nd got 0.95, 3rd got 0.92.\
Sigmoid kernels were similarly dissapointing: 1st got 0.94, 2nd got 0.96, and 3rd got 0.94.\
The F1 scores for linear kernels are all 0.98, which is as good as the accuracy.\
The F1 scores for RBF kernels are also almost the same as the accuracy, and just as disappointing.\
The F1 scores for sigmoid kernels are actually 0.01 or 0.02 better than the accuracy, which doesn't make me like the performance any more than I already don't.

###### Artificial Neural Networks
These were particularly interesting, but unfortunately no better than linear regression.\
relu activation: 1st got 0.972, 2nd got 0.969, 3rd got 0.956.\
Sigmoid activation: 1st got 0.97, 2nd got 0.92, 3rd got 0.98 (0.97963).\
The F1 scores for relu activation are the same as the accuracy scores.\
The F1 scores for logistic activation are, again, the same as the accuracy scores, or close enough, which I count as a good thing.


##### Step seven
'Hyperparameter turning (improving model performance). Algorithm tuning, ensemble methods.'\
After twice now saying that I won't do any hyperparameter tuning because the models I've used haven't had any changed from their defaults, I've decided instead that I will not do anything for step 7 until after these 6 initial projects are complete. The purpose of these first 6 is to get 6 small scale projects complete, and done with, so that I have experience in completing projects, regardless of how artificial and small that experience is.

##### Step eight
'Prediction: make actual predictions on actual data and test it's real world performance.'\
Since this is only for training, and I have used the entire dataset, this step seems unnecessary and unfeasible.\
'Presentation: present to the stakeholder/business the results of the work so far and explain the future worth (or lack thereof).'\
This writeup is my presentation of my work and what I have achieved. This project in particular has been exciting because it dealt with image data (regardless of how small and boring it is, and how we aren't even getting to see them).\
It has also made very clear - to me at least - just how easy it is to implement a wide variety of machine learning algorithms. 3 lines of code; instantiate, fit, predict. That's all it takes. And once you've prepared the data beforehand, coding multiple models is more copy and paste than complex design. This makes plain the power of this field - you can repeatedly try all the models you want, and the only cost is waiting for the computer to finish it's computations before you adjust the model and run it again, and again, and again, until you get what you want. In fact, given that it's a computer we've got here, _and a general purpose programming language no less_, I could probably have the computer automatically try multiple models with multiple hyperparameter configurations, on multiple differently preprocessed training datasets, and have it just tell me which was best. You know what I'm sticking that on the todo list.\
Incidentally, I was wrong in my prediction about sigmoid SVMs being the best, because linear activation functions on polynomially engineered training data was far superior. I speculate that this is due to the greater amount of relation between different data points that is made available to the model and its learning algorithm in the multiplication of all the different pixels, as oppose to just if they were a particular shade of grey. This also leads me to be surprised by the ANNs not doing better. 

##### Step nine
'Deployment to a production environment.'\
As with the first part of step eight, I deem this step unnecessary, due to this being a training project with no useful real world application that I can put it to at this time.

### Wine
We begin again.
##### Step one
'Define the problem/understand the business'.\
The problem here is to identify different types of wine, given some measurements from a chemical analysis of the liquid. It is a supervised classification problem.\
##### Step two
The second step is to 'locate and acquire relevant datasets'.\
The dataset is available from the SciKit Learn code library, via a code import.\
##### Step three
The third step is to 'perform an EDA (Exploratory Data Analysis) of the dataset'. I have this further defined as _'including some preliminary cleaning and perhpas feature engineering - to understand it's value and quality. Make sure the dataset is in some form that can be fed into a mathematical, algorithmically trained model'_.\

From the `.DESCR` of the dataset, I have learned that in the 'Wine Data Database', there are 178 data samples, with 50 examples of each class. Each has 13 numeric features and a target class, of which there are 3; 0,1,2. The features are all chemical measurements such as alcohol content, magnesium content, and colo(u)r intensity. One is called 'flavanoids', which I used to think was a made up word from 80's movies with mad scientists or something.\
Some summary statistical information about each feature is provided (min, max, mean, SD), which shows that they're all mostly within a couple of orders of magnitude of each other. It is stated that there are no missing values. The exact numbers of each class are then provided; class 0: 59; class 1: 71; class 2: 48.\
All 3 wines in this data were grown in the same region of Italy by 3 different cultivators.\
I will not attempt any meaningful understanding of the particulars of these features, in order to better engineer them, as I lack a chemistry BSc, or any real understanding of wine types.

###### EDA

Firstly, a look at the features of the training data: 
- alcohol
- malic_acid
- ash
- alcalinity_of_ash
- magnesium
- total_phenols
- flavanoids
- nonflavanoid_phenols
- proanthocyanins
- color_intensity
- hue
- od280/od315_of_diluted_wines
- proline

I have no concrete idea what most of these are, with the obvious exception of alcohol (I assume it's content by volume). I also assume that color_intensity and hue are related to the colour and visible appearance of the wine. I also assume that the rest of the measurements concern the presence of some substance by volume or by parts per million (or other number).\
Fortunately for me, there are no null values, and all of these features are provided in numerical form, so it will be easy to provide them to a machine learning model in a form it will work well with.

A look at the statistics of the data:

| Statistical Attribute | alcohol | malic_acid | ash | alcalinity_of_ash | magnesium | total_phenols | flavanoids | nonflavanoid_phenols | proanthocyanins | color_intensity | hue | od280/od315_of_diluted_wines | proline |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| min | 1.10300000e+01 | 7.40000000e-01 | 1.36000000e+00 | 1.06000000e+01 | 7.00000000e+01 | 9.80000000e-01 | 3.40000000e-01 | 1.30000000e-01 | 4.10000000e-01 | 1.28000000e+00 | 4.80000000e-01 | 1.27000000e+00 | 2.78000000e+02 |
| max | 1.48300000e+01 | 5.80000000e+00 | 3.23000000e+00 | 3.00000000e+01 | 1.62000000e+02 | 3.88000000e+00 | 5.08000000e+00 | 6.60000000e-01 | 3.58000000e+00 | 1.30000000e+01 | 1.71000000e+00 | 4.00000000e+00 | 1.68000000e+03 |
| mean | 1.30006180e+01 | 2.33634831e+00 | 2.36651685e+00 | 1.94949438e+01 | 9.97415730e+01 | 2.29511236e+00 | 2.02926966e+00 | 3.61853933e-01 | 1.59089888e+00 | 5.05808988e+00 | 9.57449438e-01 | 2.61168539e+00 | 7.46893258e+02 |
| variance | 6.59062328e-01 | 1.24801540e+00 | 7.52646353e-02 | 1.11526862e+01 | 2.03989335e+02 | 3.91689535e-01 | 9.97718673e-01 | 1.54886339e-02 | 3.27594668e-01 | 5.37444938e+00 | 5.22449607e-02 | 5.04086409e-01 | 9.91667174e+04 |
| skewness | -0.05104747 | 1.0308695  | -0.17520678 | 0.21124733 | 1.08891489 | 0.08590677 | 0.02512948 | 0.44634901 | 0.51276903 | 0.86124805 | 0.02091312 | -0.30468993 | 0.76133617 |
| kurtosis | -0.8622601  | 0.25734829 | 1.07857613 | 0.44082314 | 2.01280601 | -0.8458573  | -0.88936501 | -0.65295221 | 0.50567101 | 0.33736976 | -0.3680254  | -1.08967533 | -0.27499971 |

Looking at the min/max differences, I see that the greatest differences are only 4 orders of magnitude, and that nowhere are the skewness or kurtosis extreme. In only 5 of 26 instances do either have magnitudes >=1. The interesting example here is magnesium, which is the only feature to have both its skewness and kurtosis measurements magnitudes >=1.\
Lets see what that looks like: 

###### Magnesium graph
Graph 1: Magnesium histogram\
![Magnesium Histogram](https://raw.githubusercontent.com/chorhatarahuduketuri/MachineLearningProjects/master/scikitLearnDatasets/wine/magnesium.png "Magnesium Histogram")
You can see from this graph that the data for magnesium in the wines is negatively skewed, in that the mean is lower than the median. The relatively high kurtosis score compared to the rest of the data is shown in how much of the data appears to lie further away from the centre of the bell curve (if you imagine one drawn on). Neither of these is a problem that mean normalisation can't help with. 

From looking at the data, and based on my experience so far (in the last three projects, at least), I have determined that I don't really need to do much else to prepare this data for whatever generic ML models I decide to use.\
So, onto the cool part. 

##### Step four
The fourth step is to 'consider what sorts of models would be appropriate, as well as understand which sorts of algorithms will work and which will not. Feature engineering should also be undertaken at this stage, in the case of any selected model that would benefit from or require it.'\
As previously determined, this is a supervised multi-class classification problem, with non-null numerical feature data. More logistic regression and SVMs are on the table.\
As for feature engineering, I'll make up 2nd order polynomials, but not 3rd order, as I think they're basically overkill in this case.\
Of course, all this can be modified as needed based on the accuracy of the models that the machine learns. 

##### Step five
'Design, create, and/or train the model'
###### Logistic Regression
For the logistic regression models, I created two that had maximum iterations set to 500 - an increase from the default of 100. I also set the solver to L-BFGS, partly because I've used it before and partly because the documentation recommends it for multi-class problems.\
Both logistic regression models are the same, one is for the normal dataset, one is for the polynomial dataset. 

###### Support Vector Machines
I created two SVMs, one for each of the datasets I'll be using (original and 2nd order polynomial). Both have default parameters set. The default solver is RBF and the default iteration limit is unlimited. 

##### Step six
'Evaluation of model on Validation set'
###### Logistic Regression 
The logistic regression models repeatedly perform very well. \
The accuracy scores are always above 0.95, usually higher, sometimes even 1.0, which seems like the kind of result you'd only get on a small dataset. \
F1 scores are likewise excellent. \
There is no noticeable difference between the 1st and 2nd order training data input. 

###### Support Vector Machines
The SVM model also repeatedly perform well, though with greater variation. \
The accuracy scores for 1st order training data are consistently 0.95-1.0. For 2nd order training data they vary 0.88-1.0, and never higher than for 1st order training data alone. \
The F1 scores again follow the accuracy scores for each set of training data. 

##### Step seven
'Hyperparameter turning (improving model performance). Algorithm tuning, ensemble methods.'\
As per usual, especially when using default model configurations, I will not be engaging in hyperparameter tuning. 

##### Step eight
'Prediction: make actual predictions on actual data and test it's real world performance.'\
Since this is only for training, and I have used the entire dataset, this step seems unnecessary and unfeasible. \
'Presentation: present to the stakeholder/business the results of the work so far and explain the future worth (or lack thereof).'\
This writeup is my presentation of my work and what I have achieved. \ 

##### Step nine
'Deployment to a production environment.'\
As with the first part of step eight, I deem this step unnecessary, due to this being a training project with no useful real world application that I can put it to at this time.

### Boston House Prices
Finally, _linear_ regression. 
##### Step one
'Define the problem/understand the business'.\
The problem here is, given 13 (numerical **and** categorical) features, to predict the median house price of owner-occupied houses to the closest $1000. It is a supervised regression problem.

##### Step two
The second step is to 'locate and acquire relevant datasets'.\
The dataset is available from the SciKit Learn code library, via a code import.

##### Step three
The third step is to 'perform an EDA (Exploratory Data Analysis) of the dataset'. I have this further defined as _'including some preliminary cleaning and perhaps feature engineering - to understand it's value and quality. Make sure the dataset is in some form that can be fed into a mathematical, algorithmically trained model'_.\

From the `.DESCR` of the dataset, I have learned that there are 506 data samples, each with 14 features. \
The 14th feature, which is the median value of owner-occupied houses (in $1000 increments), is used as the target variable, and the import from scikit learn presents it as such, giving the other 13 features as the training dataset, and the median owner-occupied house value as the target. \
I have also learnt that the training data isn't all numerical this time, huzzah! Something categorical to deal with! Of course, since the categories are provided as integer values, this actually makes no difference between the numerical and categorical features, as to the practical way in which I will handle the data and provide it to the algorithms. Or will it? Does it still require mean normalisation? I will have to check. At the appropriate step - let's not get distracted here. \
There are no missing attributes. Easy. 

###### EDA

Firstly, a look at the features of the training data as described in the `.DESCR`: 
- CRIM - per capita crime rate by town
- ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
- INDUS - proportion of non-retail business acres per town
- CHAS - Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
- NOX - nitric oxides concentration (parts per 10 million)
- RM - average number of rooms per dwelling
- AGE - proportion of owner-occupied units built prior to 1940
- DIS - weighted distances to five Boston employment centres
- RAD - index of accessibility to radial highways
- TAX - full-value property-tax rate per $10,000
- PTRATIO - pupil-teacher ratio by town
- B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
- LSTAT - % lower status of the population
- MEDV - Median value of owner-occupied homes in $1000's

Comments on each feature: 
- CRIM - Criminal activity is generally something most people don't want to be close to. 
- ZN - Larger lots would be worth more themselves, and would accommodate larger houses. 
- INDUS - This indicates a greater variety of employment opportunities (with potentially higher wages) nearby to the house. 
- CHAS - Either because houses near the river are more aesthetically pleasing to homeowners, or because the river floods and who needs that. 
- NOX - Air quality is important, not just to how the air tastes and how it feels to breathe, but also for long term health. 
- RM - More is more, after all. 
- AGE - Older houses aren't as good as newer houses. 
- DIS - Makes it easier to get a job. 
- RAD - Transport access is important, especially when it comes to cars. 
- TAX - On-going costs are important to consider, and lower on-going costs are sometimes worth higher startup costs (be careful to do the maths on this though). 
- PTRATIO - Very important for those who have children. 
- B - Very important for racist people, and people interested in studying the effects racism has on society (in this case, economic). Is it racist to include this? For academic analysis I suppose not. For deciding how much _you're_ going to pay for the house I think it is. 
- LSTAT - I don't actually know what this means. A google search brings back nothing concrete, but suggest that it's to do with income and socioeconomic status. Basically, LSTAT % is the % of people who have low wages, relative to others. 
- MEDV - The target variable. 

Let's see if there are any particular statistical basics that stick out: 

| Statistical attribute | CRIM | ZN | INDUS | CHAS | NOX | RM | AGE | DIS | RAD | TAX | PTRATIO | B | LSTAT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| min | 6.32000000e-03 |0.00000000e+00 |4.60000000e-01 |0.00000000e+00 |3.85000000e-01 |3.56100000e+00 |2.90000000e+00 |1.12960000e+00 |1.00000000e+00 |1.87000000e+02 |1.26000000e+01 |3.20000000e-01 |1.73000000e+00 |
| max | 88.9762 |100.|27.74|1.|0.871|8.78|100.|12.1265 |24.|711.|22.|396.9|37.97 |
| mean | 3.59376071e+00 |1.13636364e+01 |1.11367787e+01 |6.91699605e-02 |5.54695059e-01 |6.28463439e+00 |6.85749012e+01 |3.79504269e+00 |9.54940711e+00 |4.08237154e+02 |1.84555336e+01 |3.56674032e+02 |1.26530632e+01 |
| variance | 7.39046710e+01 |5.43936814e+02 |4.70644425e+01 |6.45129730e-02 |1.34276357e-02 |4.93670850e-01 |7.92358399e+02 |4.43401514e+00 |7.58163660e+01 |2.84047595e+04 |4.68698912e+00 |8.33475226e+03 |5.09947595e+01 |
| skewness | 5.22203907 |2.21906306 |0.29414628 |3.39579929 |0.72714416 |0.40241467 |-0.59718559 |1.00877876 |1.00183349 |0.66796827 |-0.79994453 |-2.88179835 |0.90377074 |
| kurtosis | 36.88811011 |3.97994877 |-1.23321847 |9.53145284 |-0.07586422 |1.86102697 |-0.97001393 |0.47129857 |-0.8705205|-1.14298488 |-0.29411638 |7.14376929 |0.47654476 |

The min/max of all the features have a total range of 5 orders of magnitude, which is very large. Mean normalisation will be very useful here.\
The skewness and kurtosis is generally fairly moderate, below 10 in magnitude in most cases, except for the glaring kurtoisi of 36.89 for the CRIM feature. Lets take a look to see what that looks like: 

What an unusual looking graph (for what I've done so far): 
###### Magnesium graph
Graph 1: CRIM histogram\
![CRIM Histogram](https://raw.githubusercontent.com/chorhatarahuduketuri/MachineLearningProjects/master/scikitLearnDatasets/boston/CRIM.png "CRIM Histogram")
 
