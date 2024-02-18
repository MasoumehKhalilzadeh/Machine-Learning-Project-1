Python Code:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import plotly.express as px

import plotly.graph_objects as go # Generate Graphs
from plotly.subplots import make_subplots #To Create Subplots

from sklearn import decomposition #pca
from sklearn.preprocessing import StandardScaler # Standardization ((X - X_mean)/X_std)

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import matplotlib
matplotlib.rc('figure', figsize = (20, 8))
matplotlib.rc('font', size = 14)
matplotlib.rc('axes.spines', top = False, right = False)
matplotlib.rc('axes', grid = False)
matplotlib.rc('axes', facecolor = 'white')

data = pd.read_excel('/content/Data.xlsx')
data.head()
data.columns = ['id', 'age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
       'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio']
data.drop(['id'], axis=1, inplace=True)

data.info()

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 70000 entries, 0 to 69999
Data columns (total 12 columns):
 #   Column       Non-Null Count  Dtype
---  ------       --------------  -----
 0   age          70000 non-null  int64
 1   gender       70000 non-null  int64
 2   height       70000 non-null  int64
 3   weight       70000 non-null  int64
 4   ap_hi        70000 non-null  int64
 5   ap_lo        70000 non-null  int64
 6   cholesterol  70000 non-null  int64
 7   gluc         70000 non-null  int64
 8   smoke        70000 non-null  int64
 9   alco         70000 non-null  int64
 10  active       70000 non-null  int64
 11  cardio       70000 non-null  int64
dtypes: int64(12)
memory usage: 6.4 MB
data.describe()
data.duplicated().sum()
data.drop_duplicates(inplace=True)
data['age'] = (data['age'] / 365).round().astype('int')


num_cols = ['age','height','weight','ap_hi','ap_lo']
plt.figure(figsize=(18,9))
data[num_cols].boxplot()
plt.title("Numerical variables in the dataset", fontsize=20)
plt.show()

outliers = len(data[(data["ap_hi"]>=250)])+len(data[(data["ap_hi"]<0)])+len(data[(data["ap_lo"]>=200)])+len(data[(data["ap_lo"]<0)])
print(f'percent missing: {round(outliers/len(data)*100,1)}%')
percent missing: 1.4%

#Filtering out the unrealistic data of Systolic blood pressure and Diastolic blood pressure
data = data[ (data['ap_lo'] >= 0) & (data['ap_hi'] >= 0) ]  #remove negative values
data = data[ (data['ap_lo'] < 200) & (data['ap_hi'] < 250) ]  #remove fishy data points
data = data[ (data['ap_lo'] < data['ap_hi']) ]  #remove systolic higher than diastolic

num_cols = ['age','height','weight','ap_hi','ap_lo']
plt.figure(figsize=(18,9))
data[num_cols].boxplot()
plt.title("Numerical variables in the dataset", fontsize=20)
plt.show()

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

outlier_df = pd.DataFrame()

feat = ['age','height','weight','ap_lo','ap_hi']
for ft in feat:
    o2 = pd.DataFrame()
    o2['feature'] = [ft]
    q1 = data[ft].quantile(.25, interpolation='midpoint')
    q3 = data[ft].quantile(.75, interpolation='midpoint')
    lb, ub = ( q1 - 1.5 * (q3 - q1) ), ( q3 + 1.5 * (q3 - q1) )
    o2['num of outliers'] = [len(data[data[ft] < lb]) + len(data[data[ft] > ub])]
    data[ft] = data[ft].clip(lb, ub)
    o2['outliers after clamping'] = [len(data[data[ft] < lb]) + len(data[data[ft] > ub])]
    outlier_df = pd.concat([outlier_df, o2], axis = 0, ignore_index = True)

outlier_df

num_cols = ['age','height','weight','ap_hi','ap_lo']
plt.figure(figsize=(18,9))
data[num_cols].boxplot()
plt.title("Numerical variables in the dataset", fontsize=20)
plt.show()


#Dataset after cleaning

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

data.groupby('gender')['height'].mean()
data['gender'].value_counts()
data.groupby('gender')['alco'].sum()
pd.crosstab(data['cardio'],data['gender'],normalize=True)


corr = data.corr()
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap="YlGnBu", vmax=.3, center=0,annot = True,
            square=True, linewidths=.5, cbar_kws={"shrink": .5});

data['BMI'] = data['weight'] / data['height'] / data['height'] * 10000

data['pulse pressure'] = data['ap_hi'] - data['ap_lo']


plt.rcParams['figure.figsize'] = (20, 15) 
sns.heatmap(data.corr(), annot = True, linewidths=.5, cmap="PuBu")
plt.title('Corelation Between Features', fontsize = 30)
plt.show()

#Train-test-split for non-scaled data
X = data.drop(['cardio','height','alco'], axis=1) #features 
y = data['cardio']  #target feature

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle = True)

#Splitted Data
print('X_train shape is ' , X_train.shape)
print('X_test shape is ' , X_test.shape)
print('y_train shape is ' , y_train.shape)
print('y_test shape is ' , y_test.shape)


# Random Forest

rf = RandomForestClassifier(n_estimators=700, random_state=42)
rf.fit(X_train, y_train)

# Use the predict_proba() method to obtain the predicted probabilities
y_proba = rf.predict_proba(X_test)[:, 1]

# Evaluate the performance of your model at different decision thresholds
for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
    y_pred = (y_proba > threshold).astype(int)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    print(f"Threshold: {threshold:.1f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1-score: {f1:.3f}, AUC: {auc:.3f}")
  
# Choose the new decision threshold that optimizes your chosen metric
new_threshold = 0.5
# Use the new decision threshold to convert the predicted probabilities into binary predictions
y_pred = (y_proba > new_threshold).astype(int)


# Random Forest Model Evaluation
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

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

#we perform some Standardization
cardio_scaled=data.copy()

columns_to_scale = ['age', 'weight', 'ap_hi', 'ap_lo','cholesterol','gender','BMI','height','pulse pressure']

scaler = StandardScaler()
cardio_scaled[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])

cardio_scaled.head()

X_scaled = cardio_scaled.drop(['cardio'], axis=1) #features 
y_scaled = cardio_scaled['cardio']  #target feature

X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42, shuffle = True)

#Splitted Data

print('X_train shape is ' , X_train.shape)
print('X_test shape is ' , X_test.shape)
print('y_train shape is ' , y_train.shape)
print('y_test shape is ' , y_test.shape)

params = {'n_neighbors':list(range(0, 51)),
          'weights':['uniform', 'distance'],
          'p':[1,2]}

"""knn = KNeighborsClassifier()
knn_grid_cv = GridSearchCV(knn, param_grid=params, cv=10) 
knn_grid_cv.fit(X_train, y_train)
print("Best Hyper Parameters:\n",knn_grid_cv.best_params_)"""

print("Best Hyper Parameters: {'n_neighbors': 50, 'p': 1, 'weights': 'uniform'}")
Best Hyper Parameters: {'n_neighbors': 50, 'p': 1, 'weights': 'uniform'}
from sklearn.neighbors import KNeighborsClassifier #KNN Model

knn = KNeighborsClassifier(n_neighbors=50, p=1, weights='uniform')
knn.fit(X_train_scaled, y_train_scaled) 

from sklearn.model_selection import cross_val_score

scores = cross_val_score(knn, X_train_scaled, y_train_scaled, cv=10)
print('KNN Model gives an average accuracy of {0:.2f} % with minimun of {1:.2f} % and maximum of {2:.2f} % accuracy'.format(scores.mean() * 100, scores.min() * 100, scores.max() * 100))

Y_hat = knn.predict(X_test_scaled)
print(classification_report(y_test_scaled, Y_hat))

# Compute confusion matrix
cm = confusion_matrix(y_test_scaled, Y_hat)

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
