
# Cardiovascular Diseases Prediction using Machine Learning 




## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Algorithm Definition](#algorithm-definition)
4. [Data Cleaning](#data-cleaning)
5. [Exploratory Data Analysis](#exploratory-data-analysis)
6. [Feature Engineering](#feature-engineering)
7. [Modeling](#modeling)
8. [Performance Evaluation](#performance-evaluation)
9. [Random Forest Performance](#random-forest-performance)
10. [The K-Nearest Neighbors Performance](#the-k-nearest-neighbors-performance)
11. [Logistic Regression Performance](#logistic-regression-performance)
12. [Discussion](#discussion)


   




## Introduction

Cardiovascular diseases (CVDs) are the most frequent reason for death nowadays. According to WHO, an estimated 17.9 million people died from CVDs in 2019, representing 32% of all global deaths. Of these deaths, 85% were due to heart attack and stroke. Cardiovascular diseases are conditions that affect the function of your heart and blood vessels including coronary heart disease, cerebrovascular disease, rheumatic heart disease, and other conditions. Most of the CVD deaths are due to heart attacks and strokes, and one-third of these deaths occur in people who are under 70 years old. Most cardiovascular diseases are related to behavioral risk factors such as tobacco use, unhealthy diet, obesity, physical inactivity, and harmful use of alcohol. In this project, three different machine learning techniques,Random Forest, KNN, and Logistic Regression have been performed for CVD detection using a dataset from Kaggle, a public data repository. Outliers and unrelated observations have been removed to increase the model performance. Based on the results, we obtained 73 % accuracy in detection using the KNN model. All the analysis has been performed in Python 3.


## Dataset 

In this project, a dataset of 70000 patients has been analyzed which includes 11 attributes and the target variable. The dataset is obtained from Kaggle, a public data repository for datasets. We investigated this dataset to predict the presence or absence of Cardiovascular diseases based on the provided features. The attributes are Age, Gender, Height, Weight, Systolic blood pressure, Diastolic blood pressure, Cholesterol (1: normal, 2: above normal, 3: well above normal), Glucose (1: normal, 2: above normal, 3: well above normal), Smoking, Alcohol intake, Physical activity. Our target variable is whether a person has CVD or does not.

```python
data = pd.read_excel('/content/Data.xlsx')
data.head()
data.columns = ['id', 'age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
       'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio']
data.drop(['id'], axis=1, inplace=True)

data.info()
```
<img width="553" alt="image" src="https://github.com/Masoumeh89/Machine-Learning-Project-1/assets/74910834/158f4473-4a00-41a3-94a5-d3e17b577b97">


## Algorithm Definition

The required steps for model building to predict CVD are as follows:

**Step 1:** Data cleaning and pre-processing have been implemented by eliminating improper values and removing outliers from the observation. 

**Step 2:** EDA (Exploratory Data Analysis) has been performed to analyze and summarize data sets to gain insights into the underlying patterns, distributions, and relationships between variables.

**Step 3:** Feature engineering and feature selection have been performed to choose the proper attributes that are more helpful in CVD prediction.

**Step 4:** Random forest, KNN, and Logistic Regression algorithms are chosen to classify the selected features.

**Step 5:** Performance measures are evaluated to gain proper results from the models.



## Data Cleaning

- By looking at the statistical description of the data, we can say that there are no missing values in the dataset and the “age” variable is in days that have been changed to years. Besides, by looking at the statistical description of numerical variables, we recognized that Systolic blood pressure(ap_hi) and Diastolic blood pressure (ap_lo) have negative values which are not acceptable. So, we filtered out the unrealistic data of Systolic blood pressure and Diastolic blood pressure from the dataset.
- These two variables cannot be negative, so we removed the negative values too.
- According to the American Heart Association, if the Systolic blood pressure and Diastolic blood pressure values exceed 180/120 mm Hg, it will be a hypertensive crisis. So, for safety, we considered the values greater than 250 for Systolic blood pressure and 200 for Diastolic blood pressure as outliers and they need to be removed.
- I checked the number of duplicate observations, we removed 24 duplicated observations from the dataset.


```python
num_cols = ['age','height','weight','ap_hi','ap_lo']
plt.figure(figsize=(18,9))
data[num_cols].boxplot()
plt.title("Numerical variables in the dataset", fontsize=20)
plt.show()
```


- The next step is to investigate the outliers, we have visualized the numerical quantities in the dataset as boxplots, to have a better sense of the outliers. The boxplot of numerical variables like “age”, “height”,” weight”,”ap_hi” and “ap_lo” is as below:

![image](https://github.com/Masoumeh89/Machine-Learning-Project-1/assets/74910834/8dfc4095-0122-431c-9145-503421633058)

There are too many outliers and if we remove all the outliers, then almost 10% of the data will be missed. So, we decided to use another method and replace the outliers with the first and third quartiles. 

**Clamping** is a statistical process used to replace extreme values or outliers with more representative values. One method of clamping involves using the first and third quartile of a data set to identify potential outliers. The first quartile is the value below which 25% of the data falls, while the third quartile is the value below which 75% of the data falls. Any value outside of the interquartile range (IQR), which is defined as the difference between the third and first quartiles, multiplied by a factor (typically 1.5 or 3), is considered an outlier. To clamp outliers using the quartile method, values outside of the IQR are replaced with the closest value within the range of the IQR. This process helps to reduce the influence of extreme values on statistical analysis and modeling. After removing unnecessary observations and fixing the outliers, data, we recognized that 1.86% of the observations had been missed. Now, we have 68699 observations in the dataset. The boxplot after fixing the outliers is as below:

```python

outliers = len(data[(data["ap_hi"]>=250)])+len(data[(data["ap_hi"]<0)])+len(data[(data["ap_lo"]>=200)])+len(data[(data["ap_lo"]<0)])
print(f'percent missing: {round(outliers/len(data)*100,1)}%')

```


```python

#Filtering out the unrealistic data of Systolic blood pressure and Diastolic blood pressure
data = data[ (data['ap_lo'] >= 0) & (data['ap_hi'] >= 0) ]  #remove negative values
data = data[ (data['ap_lo'] < 200) & (data['ap_hi'] < 250) ]  #remove fishy data points
data = data[ (data['ap_lo'] < data['ap_hi']) ]  #remove systolic higher than diastolic
```
```python
num_cols = ['age','height','weight','ap_hi','ap_lo']
plt.figure(figsize=(18,9))
data[num_cols].boxplot()
plt.title("Numerical variables in the dataset", fontsize=20)
plt.show()
```

![image](https://github.com/Masoumeh89/Machine-Learning-Project-1/assets/74910834/a1500c64-9983-4cef-9624-6ed2aa3605a3)


After clamping outliers using the quartile method, it is important to check that the resulting data set is still representative of the underlying population and that the clamping did not introduce any unintended biases. One way to check this is by examining the distribution of the data before and after clamping.

### Distribution before fixing the outliers:

![image](https://github.com/Masoumeh89/Machine-Learning-Project-1/assets/74910834/20550b4a-28ba-45b4-bdd3-796c2bc0bbb0)

```python
fig = make_subplots(rows=2, cols=2, subplot_titles=("Height Distribution", "Weight Distribution","ap_hi","ap_lo"))

trace0 = go.Histogram(x=data['height'], name = 'Height')
trace1 = go.Histogram(x=data['weight'], name = 'Weight')
trace2 = go.Histogram(x=data['ap_hi'], name = 'ap_hi')
trace3 = go.Histogram(x=data['ap_lo'], name = 'ap_lo')

fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)
fig.append_trace(trace2, 2, 1)
fig.append_trace(trace3, 2, 2)

fig.update_xaxes(title_text="Height", row=1, col=1)
fig.update_yaxes(title_text="Total Count", row=1, col=1)

fig.update_xaxes(title_text="Weight", row=1, col=2)
fig.update_yaxes(title_text="Total Count", row=1, col=2)

fig.update_xaxes(title_text="ap_hi", row=2, col=1)
fig.update_yaxes(title_text="Total Count", row=2, col=1)

fig.update_xaxes(title_text="ap_lo", row=2, col=2)
fig.update_yaxes(title_text="Total Count", row=2, col=2)

fig.update_layout(title_text="Histograph", height=700)

fig.show()
```
### Distribution after fixing the outliers:

![image](https://github.com/Masoumeh89/Machine-Learning-Project-1/assets/74910834/75a70115-9495-4848-8933-72ace73b3879)


So, we can see that fixing the outliers with the clamping method did not change the distribution and statistical description of the data. Meanwhile, We have plotted the bar chart of the target variable, and as can be seen; the data set was balanced as 33989 patients had CVD and 34710 patients had not CVD. 

#Dataset after cleaning

```python
print(f'Number of rows of cardio dataset after data preprocessing: {len(data)}')
print(f'How much percent missing: {round((70000-len(data))/70000*100,2)}%')

cardio = data['cardio'].value_counts()
plt.figure(figsize=(7, 6))
ax = cardio.plot(kind='bar', rot=0, color=['#ADD8E6','#7BC8F6'])
ax.set_title("Cardiovascular Heart Disease Presense", y = 1)
ax.set_xlabel('cardio')
ax.set_ylabel('Frequency')
ax.set_xticklabels(('0', '1'))

counts = data['cardio'].value_counts()
print(counts)
```


![image](https://github.com/Masoumeh89/Machine-Learning-Project-1/assets/74910834/ff822e36-d1ad-4ae6-998a-92358a7b80b0)


## Exploratory Data Analysis


- First, we investigated the relationship of the age variable with the target variable. One can notice that individuals who are older than 55 years of age have a higher risk of being affected by cardiovascular disease (CVD). Individuals who belong to younger age groups have a reduced likelihood of developing cardiovascular disease (CVD). The plot indicates a decrease in the incidence of non-cardiovascular disease (CVD) and an increase in the incidence of CVD after reaching the highest point in the age group of 55. Individuals who belong to older age groups are more susceptible to developing cardiovascular disease (CVD).


![image](https://github.com/Masoumeh89/Machine-Learning-Project-1/assets/74910834/d069b9d9-0675-4595-b439-be057f3322e4)

- The relationship between categorical variables with target variable:

```python
  rcParams['figure.figsize'] = 18, 8
sns.set_palette("Paired")
sns.countplot(x='age', hue='cardio', data = data);


rcParams['figure.figsize'] = 11, 8
sns.set_palette("Paired")
sns.countplot(x='cholesterol', hue='cardio', data = data);

rcParams['figure.figsize'] = 11, 8
sns.set_palette("Paired")
sns.countplot(x='gluc', hue='cardio', data = data);


rcParams['figure.figsize'] = 11, 8
sns.set_palette("Paired")
sns.countplot(x='smoke', hue='cardio', data = data);

rcParams['figure.figsize'] = 11, 8
sns.set_palette("Paired")
sns.countplot(x='alco', hue='cardio', data = data);

rcParams['figure.figsize'] = 11, 8
sns.set_palette("Paired")
sns.countplot(x='active', hue='cardio', data = data);

rcParams['figure.figsize'] = 11, 8
sns.set_palette("Paired")
sns.countplot(x='gender', hue='cardio', data = data);
```

![image](https://github.com/Masoumeh89/Machine-Learning-Project-1/assets/74910834/98696764-be59-4814-bd7d-3895e9ef4786)
![image](https://github.com/Masoumeh89/Machine-Learning-Project-1/assets/74910834/2abb20b6-830b-437c-987d-d46e91b4cf8b)
![image](https://github.com/Masoumeh89/Machine-Learning-Project-1/assets/74910834/07913caa-1628-4bb0-9315-23ae1defad56)
![image](https://github.com/Masoumeh89/Machine-Learning-Project-1/assets/74910834/9ef58914-8429-433d-961a-e07efc7510d2)
![image](https://github.com/Masoumeh89/Machine-Learning-Project-1/assets/74910834/98efa332-8839-4f47-bd96-3b474f5437f1)
![image](https://github.com/Masoumeh89/Machine-Learning-Project-1/assets/74910834/da788c8f-d4d7-4859-b6d8-9236bacb0291)


Individuals who have cardiovascular disease (CVD) tend to exhibit elevated levels of cholesterol and blood glucose, as well as a lower level of physical activity overall. 

- In the dataset, 44742 are women and 23957 are men, and below is the crosstab presentation of how the target class is distributed among men and women;

```python
data.groupby('gender')['height'].mean()
data['gender'].value_counts()
data.groupby('gender')['alco'].sum()
pd.crosstab(data['cardio'],data['gender'],normalize=True)
```

<img width="286" alt="image" src="https://github.com/Masoumeh89/Machine-Learning-Project-1/assets/74910834/763f5143-dc50-4049-90dd-293646e46291">

- **Multivariate Analysis**: We can observe that Systolic blood pressure and Diastolic blood pressure are the most correlated variables with the target variable.

```python
  corr = data.corr()
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
```


![image](https://github.com/Masoumeh89/Machine-Learning-Project-1/assets/74910834/29e86035-ce04-4509-8ce9-47a7a2f22241)

  

## Feature Engineering

Whenever height and weight measurements are available, it is possible to compute the body mass index (BMI). It may be advantageous to create an additional feature for BMI, as it could potentially yield more valuable insights. The body mass index (BMI) is a frequently used measurement for assessing medical health and cardiovascular wellness. BMI can be calculated by the following: BMI = weight(kg) / height (cm) / height (cm) x 10,000
Pulse pressure is an additional indicator of cardiovascular wellbeing. Pulse Pressure can be calculated by the following: Pulse Pressure = systolic – diastolic. Usually, a pulse pressure that exceeds 60 can be a beneficial predictor of the likelihood of experiencing heart attacks or other cardiovascular ailments.

**Feature Selection**

```python

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap="YlGnBu", vmax=.3, center=0,annot = True,
            square=True, linewidths=.5, cbar_kws={"shrink": .5});

data['BMI'] = data['weight'] / data['height'] / data['height'] * 10000

data['pulse pressure'] = data['ap_hi'] - data['ap_lo']
```
```python
plt.rcParams['figure.figsize'] = (20, 15) 
sns.heatmap(data.corr(), annot = True, linewidths=.5, cmap="PuBu")
plt.title('Corelation Between Features', fontsize = 30)
plt.show()
```

![image](https://github.com/Masoumeh89/Machine-Learning-Project-1/assets/74910834/d21b6c19-7c38-40fd-be02-f4c6d8db0d03)


The plot of the correlation matrix indicates that the variable "ap_hi" has a positive correlation with the target output, with a coefficient of 0.44. This implies that the presence of "ap_hi" increases the likelihood of developing cardiovascular disease. Likewise, there are negative correlations between some variables, such as "active" and the target variable. The correlation coefficient of -0.037 suggests that if a person has a high level of physical activity, they are less likely to suffer from cardiovascular disease (CVD). It is preferable to choose those that have a strong positive correlation with the target variable. To streamline the data and improve accuracy, one of these features can be eliminated during the pre-processing stage. If two independent features have a high correlation, it means they are both trying to represent the same thing. So, if one of the features is dropped, there won't be a significant loss of quality data. The feature "ap_hi" is correlated with several other features, but it has the highest correlation with the target value, so it cannot be ignored. The same is true for "pulse" and "ap_lo". 

The feature alcohol has the lowest correlation with the target feature. Also, features such as 'height', 'smoke', and 'active' have relatively low correlation values with the target feature. To ensure the quality of our data, we will remove some features such as ‘height’ and 'alco'. Although 'age' and 'cholesterol' have a significant impact, their correlation with the target class is not very high. The feature 'ap_hi' has the highest correlation with the target value, indicating that it has a significant impact on the model. Similarly, 'ap_lo' also has a strong correlation with the target value and is important for the model. 

## Modeling 


The ratio of the training set to the test set can vary depending on the size and complexity of the dataset, but a common ratio is 80% for training and 20% for testing. By using a train-test split, we can avoid overfitting the model to the training data, which would result in poor performance on new data. In this scenario, we will be using the commonly used split ratio of 80:20, which means that 80% of the dataset will be used for training the model, and the remaining 20% of the dataset will be used for testing the model. In this project, we used random forest and KNN algorithms to model the data. 

```python

#Train-test-split for non-scaled data
X = data.drop(['cardio','height','alco'], axis=1) #features 
y = data['cardio']  #target feature

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle = True)
#Splitted Data
print('X_train shape is ' , X_train.shape)
print('X_test shape is ' , X_test.shape)
print('y_train shape is ' , y_train.shape)
print('y_test shape is ' , y_test.shape)
```


## Performance Evaluation 


Confusion Matrix Parameters



<img width="500" alt="image" src="https://github.com/Masoumeh89/Machine-Learning-Project-1/assets/74910834/0b253150-3e14-4124-964a-91d9998a0c4a">


Definition of performance metrics

<img width="454" alt="image" src="https://github.com/Masoumeh89/Machine-Learning-Project-1/assets/74910834/0b7bb1ec-64c2-4420-a04d-a794ea6f1e39">




## Random Forest Performance

```python

# Random Forest

rf = RandomForestClassifier(n_estimators=700, random_state=42)
rf.fit(X_train, y_train)

# Use the predict_proba() method to obtain the predicted probabilities
y_proba = rf.predict_proba(X_test)[:, 1]
```

```python

# Evaluate the performance of your model at different decision thresholds
for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
    y_pred = (y_proba > threshold).astype(int)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    print(f"Threshold: {threshold:.1f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1-score: {f1:.3f}, AUC: {auc:.3f}")
```

```python
# Choose the new decision threshold that optimizes your chosen metric
new_threshold = 0.5
# Use the new decision threshold to convert the predicted probabilities into binary predictions
y_pred = (y_proba > new_threshold).astype(int)

```

```python

# Random Forest Model Evaluation
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
```


<img width="415" alt="image" src="https://github.com/Masoumeh89/Machine-Learning-Project-1/assets/74910834/5668aa29-1ba4-4abd-92f1-10cfe3bbe55f">

```python
# Plot confusion matrix
fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=['Negative', 'Positive'],
       yticklabels=['Negative', 'Positive'],
       title='Confusion matrix',
       ylabel='True label',
       xlabel='Predicted label')
# Add labels to each cell
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")

fig.tight_layout()
plt.show()
```

![image](https://github.com/Masoumeh89/Machine-Learning-Project-1/assets/74910834/a3dfcfd4-0abc-4684-8cf3-8dbc7f820fd4)


Based on the accuracy, we can say that the model correctly predicted the outcome for 71% of the patient’s CVD in the test dataset. The precision says that of the patients that the model predicted to have CVD, 70% had CVD. This is a moderate precision score but it suggests that the model may be missing some patient’s CVD history. The Recall indicates that of the patients that had CVD, the model correctly identified 70%. Moreover, The F1-score is a balance between precision and recall and is 0.70 in this case. This indicates that the model is performing well overall, with a good balance between precision and recall. According to these performance metrics, we can conclude that the Random Forest model is performing well overall and is correctly identifying a high proportion of patient’s CVD. However, the model may need to be further tuned to improve its recall score and identify more patients who had CVD.

## The K-Nearest Neighbors Performance

```python
#we perform some Standardization
cardio_scaled=data.copy()

columns_to_scale = ['age', 'weight', 'ap_hi', 'ap_lo','cholesterol','gender','BMI','height','pulse pressure']

scaler = StandardScaler()
cardio_scaled[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])

cardio_scaled.head()

X_scaled = cardio_scaled.drop(['cardio'], axis=1) #features 
y_scaled = cardio_scaled['cardio']  #target feature

X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42, shuffle = True)

```

```python
#Splitted Data

print('X_train shape is ' , X_train.shape)
print('X_test shape is ' , X_test.shape)
print('y_train shape is ' , y_train.shape)
print('y_test shape is ' , y_test.shape)
```


```python
params = {'n_neighbors':list(range(0, 51)),
          'weights':['uniform', 'distance'],
          'p':[1,2]}

"""knn = KNeighborsClassifier()
knn_grid_cv = GridSearchCV(knn, param_grid=params, cv=10) 
knn_grid_cv.fit(X_train, y_train)
print("Best Hyper Parameters:\n",knn_grid_cv.best_params_)"""
```

```python
print("Best Hyper Parameters: {'n_neighbors': 50, 'p': 1, 'weights': 'uniform'}")
Best Hyper Parameters: {'n_neighbors': 50, 'p': 1, 'weights': 'uniform'}
from sklearn.neighbors import KNeighborsClassifier #KNN Model

knn = KNeighborsClassifier(n_neighbors=50, p=1, weights='uniform')
knn.fit(X_train_scaled, y_train_scaled)
```

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(knn, X_train_scaled, y_train_scaled, cv=10)
print('KNN Model gives an average accuracy of {0:.2f} % with minimun of {1:.2f} % and maximum of {2:.2f} % accuracy'.format(scores.mean() * 100, scores.min() * 100, scores.max() * 100))

Y_hat = knn.predict(X_test_scaled)
print(classification_report(y_test_scaled, Y_hat))

```



<img width="418" alt="image" src="https://github.com/Masoumeh89/Machine-Learning-Project-1/assets/74910834/57638864-c213-48a9-a482-42cb495504db">

```python
# Compute confusion matrix
cm = confusion_matrix(y_test_scaled, Y_hat)

```

```python

# Plot confusion matrix
fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=['Negative', 'Positive'],
       yticklabels=['Negative', 'Positive'],
       title='Confusion matrix',
       ylabel='True label',
       xlabel='Predicted label')

# Add labels to each cell
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")

fig.tight_layout()
plt.show()
```

![image](https://github.com/Masoumeh89/Machine-Learning-Project-1/assets/74910834/b4775205-dcbf-4815-8cf6-a6319d6e338a)


Based on the accuracy, we can say that the model correctly predicted the outcome for 73% of the patient’s CVD in the test dataset. The precision says that of the patients that the model predicted to have CVD, 75% had CVD. So, we can conclude that KNN provides better performance than random forest for this dataset.


## Logistic Regression Performance


<img width="413" alt="image" src="https://github.com/Masoumeh89/Machine-Learning-Project-1/assets/74910834/d1c53635-ee31-43f5-8bf6-14df70a1203e">

![image](https://github.com/Masoumeh89/Machine-Learning-Project-1/assets/74910834/766120f6-8b1a-47ef-b4ba-fb0256d4fb88)

As can be seen from the results, the accuracy of logistic regression is almost the same as the KNN algorithm.

## Discussion

In this project, we have investigated the behavior and patterns in a dataset of cardiovascular disease to provide a prediction of the presence of CVD among the patients. In this regard, we used Random Forest, KNN algorithms, and Logistic Regression to perform the prediction. While all these algorithms have their strengths and weaknesses, studies have shown that KNN can provide better accuracy. It is important to carefully consider the characteristics of the dataset and the requirements of the task when selecting an algorithm for classification.


