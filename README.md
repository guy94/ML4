# ML4

## Project Goal
Compare 5 different feature selection algorithms on high dimensionality data sets

## Description
10 data sets were chosen and all are suffering from the "Curse of Dimensinality" - many feature with low amount of samples. For each data set, the best cross-validation method was selected (affected by samples zise). Each split was preprocessed (missing values imputation, discretization, variance threshold removal, etc.) prior to mdoel feature selection and training. 
In total 5 feature selection methods were tested along with 4 ML algorithms and 4 metrics methods.


## Project Resultts
After collecting all the predictions results, run time of each method and more, a data report was analyzed by the friedman test for distribution consistency of each feature selection method and the ROC AUC metric. If the null hypothesis was rejected (a difference between the methods was found - p-value < 0.05) we continued to a post-hoc test to determine which algorithm is superior.

## The Post-Hoc Test Matrix
![image](https://user-images.githubusercontent.com/62709275/177503509-3731ab3e-9715-4959-8792-99855f56b4ab.png)

Each individual cell (representing two feature selection methods) containing a value < 0.05 means that one algorithm is significantly better than the other. In order to determine which is better of the two, we calculated the ranks of the algorithms based on th average ROC AUC score achieved. Low rank - better algorithm.

### Ranks
![image](https://user-images.githubusercontent.com/62709275/177504659-e398845a-c5ce-4310-a48c-3e9afd2fb7bc.png)

## Mean Run Time Chart
![image](https://user-images.githubusercontent.com/62709275/177505099-4b97de22-f2c7-4831-9865-aafb06cf1151.png)

## Conclusions
RFE was determined to be the superior fearure selection algorithm due to best scoring and low run time.



