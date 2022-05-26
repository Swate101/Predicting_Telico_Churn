# Predicting  Telico_Churn number 2
Telecom Customer Churn Prediction

What is Churn? and how do we stop it?

Churn denotes that an existing customer of customers and clients is terminating his/her services with the organization and not using us any more for our services.

Is Churn Important? and, why is churn important?

The objective of a Telico are any profit producing buisness is too...

bring new customers, and keep those customers.For the telico corporation.
When the existing customer leaves the Telico, then it is a loss to the company and also creates a smear on the companys reputation. it also cost more money to acquire new customers then to keep old ones, once again loosing us profit.

The Churn Prediction models helps us at Telico in identifiying, predicting and analizing all the groups of customers are about to terminate the companys services.The pridictive modeling that we do will helps us reach and fix the most at risk customers to churn.

Dataset

The dataset is a fictional dataset of customer churn from the telecom industry. and the telico data file csv is in this github repositiery.

Objective

The object as a data scientist is to complete the models and data aquisition, clean it and then hyperfit it to see the best working results

Machine Learning Approaches Used

Logistic Regression
LinearSVC
Kernal SVC
KNN
Random Forest
Adaboost
Model Evaluation Metric

The dataset is incomplete and dirty, so pin point accurecy is not a suitable measure in this problem. My main objective is to find all the customers who churn True Positive.

True Positive is more important. In this case, the False Negative ( Actually Churn but Predicted as Not Churn) is more costly(important) than False Positive. So the metric chosen is F2-Score.

Output:

The Logistic Regression model is finalized as it has obtained the highest F2-Score of 57.30%. This model has obtained an accuracy of 73.00.
