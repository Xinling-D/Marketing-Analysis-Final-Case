# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
import seaborn as sns
from pandas_profiling import ProfileReport
from sklearn import metrics 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 

#Customer segmentation
subscribers = pd.read_csv('subscribers.csv')

customers = subscribers[['package_type','num_weekly_services_utilized', 'preferred_genre', 'intended_use', 'weekly_consumption_hour', 'num_ideal_streaming_services', 'age', 'male_TF', 'plan_type', 'monthly_price', 'discount_price', 'join_fee', 'current_sub_TF', 'payment_period', 'trial_completed']]

customers = pd.get_dummies(data=customers, columns=['package_type', 'preferred_genre', 'intended_use', 'male_TF', 'plan_type', 'current_sub_TF', 'trial_completed'])

customers.dropna(inplace=True)
customers = customers.drop(customers[customers['age']>100].index)
customers = customers.drop(customers[customers['weekly_consumption_hour']<0].index)
customers = customers.drop(customers[customers['num_ideal_streaming_services']<0].index)
customers = customers.drop(customers[customers['join_fee']<0].index)

profile = ProfileReport(customers)
profile.to_file("FP_Report.html")

x = customers.values 
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)

inertias = {}
for k in range(1, 20):
    kmeans = KMeans(n_clusters=k, random_state=2020)
    kmeans.fit(df)
    inertias[k] = kmeans.inertia_
print(inertias)


ax = plt.subplot()
ax.plot(list(inertias.keys()), list(inertias.values()), '-*')
ax.set_xticks(np.arange(1, 20))
ax.grid()
plt.show()

k = 3
kmeans = KMeans(n_clusters=k, random_state=2020)
y_pred = kmeans.fit_predict(df)
y_pred = list(y_pred)

customers['cluster'] = y_pred

customers.to_csv('customers.csv')

reduced_data = PCA(n_components=2).fit_transform(df)
results = pd.DataFrame(reduced_data, columns=['pca1', 'pca2'])

sns.scatterplot(x="pca1", y="pca2", hue=y_pred, data=results)
plt.title('K-means Clustering with 3 dimensions')
plt.show()

#Advertising channel
channel = pd.read_csv('channel_spend_graduate.csv')
channel.drop(['date'], axis=1, inplace=True)
converted = subscribers[['attribution_survey', 'attribution_technical']]

channel = channel.groupby(['channel']).sum()
converted_ways = converted.value_counts(converted.attribution_survey)
actual_ways = converted.value_counts(converted.attribution_technical)

CAC_youtube = 8730/734
CAC_search = 222500/22105
CAC_facebook = 113500/75068
CAC_bing = 10800/1238
CAC_display = 366/1259

#Churn Model
churn = subscribers[['package_type','num_weekly_services_utilized', 'preferred_genre', 'intended_use', 'weekly_consumption_hour', 'num_ideal_streaming_services', 'age', 'male_TF', 'plan_type', 'monthly_price', 'discount_price', 'join_fee', 'current_sub_TF', 'payment_period', 'trial_completed', 'cancel_date']]

churn.dropna(axis=0, subset=['package_type','num_weekly_services_utilized', 'preferred_genre', 'intended_use', 'weekly_consumption_hour', 'num_ideal_streaming_services', 'age', 'male_TF', 'plan_type', 'monthly_price', 'discount_price', 'join_fee', 'current_sub_TF', 'payment_period', 'trial_completed'], inplace = True)
churn = churn.drop(churn[churn['age']>100].index)
churn = churn.drop(churn[churn['weekly_consumption_hour']<0].index)
churn = churn.drop(churn[churn['num_ideal_streaming_services']<0].index)
churn = churn.drop(churn[churn['join_fee']<0].index)
churn = churn.drop(churn[churn['payment_period']>1].index)

churn = pd.get_dummies(data=churn, columns=['package_type', 'preferred_genre', 'intended_use', 'male_TF', 'plan_type', 'trial_completed'])

churn['churn'] = 0
churn['churn'][churn['cancel_date'].notna()] = 1
churn['churn'][churn['cancel_date'].isna().any() and churn['current_sub_TF'] == False] = 1

churn.drop('current_sub_TF',axis = 1, inplace = True)
churn.drop('cancel_date',axis = 1, inplace = True)

train, test = train_test_split(churn, test_size=0.3, random_state=42)

train_x = train.iloc[:,0:32]
train_y = train.iloc[:,32]
test_x = test.iloc[:,0:32]
test_y = test.iloc[:,32]

clf = DecisionTreeClassifier(max_depth=10)
clf.fit(train_x,train_y)
y_pred = clf.predict(test_x)
y_prob = clf.predict_proba(test_x)[:,1]

print("Accuracy:",metrics.accuracy_score(test_y, y_pred))

fpr, tpr, threshold = metrics.roc_curve(test_y, y_prob)
roc_auc = metrics.auc(fpr, tpr)

plt.title('ROC Curve')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()




