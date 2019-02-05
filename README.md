General points to remember; <br>
# 1. Variables

    - x = independent variable = explanatory = predictor
    - y = dependent variable = response = target

# 2. Data Types
The type of data is essential as it determines what kind of tests can be applied to it.

- **i.Continuous:** Also known as quantitative. Unlimited number of values
- **ii.Categorical:** Also known as discrete or qualitative. Fixed number of values or categories

# 3. Bias-Variance Tradeoff
The best predictive algorithm is one that has **good Generalization Ability**. With that, it will be able to give accurate predictions to new and previously unseen data.

- **High Bias** results from *Underfitting the model*. This usually results from erroneous assumptions, and cause the model to be too general.

- **High Variance** results from *Overfitting the model*, and it will predict the training dataset very accurately, but not with unseen new datasets. This is because it will fit even the slightless noise in the dataset.

The best model with the highest accuarcy is the middle ground between the two.

![e1](https://user-images.githubusercontent.com/42317258/52269846-34748d00-2965-11e9-89fc-8e8d6f48c72b.PNG)

# 4. Steps to Build a Predictive Model
## 4.1. Feature Selection & Preprocessing
- i.Remove features that have too many NAN or fill NAN with another value
- ii.Remove features that will introduce data leakage
-iii.Encode categorical features into integers

## 4.2. Normalise the Features
With the exception of **Decision Trees and Naive Bayes**, other machine learning techniques like Neural Networks, KNN, Support Vector Machines should have their features scaled.

## 4.3. Train Test Split
Split the dataset into Train and Test datasets. By default, sklearn assigns 75% to train & 25% to test randomly. A random state (seed) can be selected to fixed the randomisation.

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(predictor, target, test_size=0.25, random_state=0)
    
 ## 4.4. Create Model
Choose model and set model parameters (if any).

        from sklearn.tree import DecisionTreeClassifier
        
        clf = DecisionTreeClassifier(random_state=0)
        
## 4.5. Fit Model
Fit the model using the training dataset.

        model = clf.fit(X_train, y_train)
        >>> print model
        DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=None, splitter='best')
            
## 4.6. Test Model
Test the model by predicting identity of unseen data using the testing dataset.

        y_predict = model.predict(X_test)
        
## 4.7. Score Model
Use a confusion matrix and…

    print sklearn.metrics.confusion_matrix(y_test, predictions)
     [[14  0  0]
     [ 0 13  0]
     [ 0  1 10]]
     
accuarcy percentage score to obtain the predictive accuarcy.

    import sklearn.metrics
    print sklearn.metrics.accuracy_score(y_test, y_predict)*100, '%'
    >>> 97.3684210526 %
    
## 4.8. Cross Validation

When all code is working fine, remove the train-test portion and use Grid Search Cross Validation to compute the best parameters with cross validation.

# ->I. Exploratory Data Analysis

Exploratory data analysis (EDA) is an essential step to understand the data better; in order to engineer and select features before modelling. This often requires skills in visualisation to better interpret the data.

## i. Distribution Plots
When plotting distributions, it is important to compare the distribution of both train and test sets. If the test set very specific to certain features, the model will underfit and have a low accuarcy.

## ii. Count Plots
For **categorical features**, you may want to see if they have enough sample size for each category.

## iii. Box Plots
Using the 50 percentile to compare among different classes, it is easy to find feature that can have high prediction importance if they do not overlap. Also can be use for outlier detection. **Features have to be continuous.**
