import streamlit as st
# Import Relevant Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np
from sklearn import preprocessing
import seaborn as sns
data=pd.read_csv('sampled_data.csv')
data.drop(columns=['user_id'], inplace=True)
# Replacing Categoric and Numerical Missing Value with a FOR LOOP
for column in data :
  if ((data[column].isna().sum())/len(data))*100 > 0 :
    if data[column].dtypes =='float64'  or data[column].dtype == 'int64':
      data[column].fillna(data[column].median(), inplace=True)
    else :
      data[column].fillna(data[column].mode()[0], inplace=True)
regions = set(data['REGION'])
region = st.selectbox('Select your region', regions)
top_packs = set(data['TOP_PACK'])
top_pack = st.selectbox('Select your top_pack', top_packs)

label_encoder = preprocessing.LabelEncoder()

data['REGION']= label_encoder.fit_transform(data['REGION'])
mapping_dict_Region = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
data['TENURE']=label_encoder.fit_transform(data['TENURE'])
mapping_dict_TENURE = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
data['MRG']=label_encoder.fit_transform(data['MRG'])
mapping_dict_MRG = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
data['TOP_PACK']=label_encoder.fit_transform(data['TOP_PACK'])
mapping_dict_TOP_PACK = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
x = data[['REGION','TOP_PACK']]
y = data['CHURN']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)
#splitting data with test size of 25%
logreg = LogisticRegression()   #build our logistic model
logreg.fit(x_train, y_train)
if region in regions: #hne zedna 3malna il map ala 5ater il client mosh bish y7out 0 w 1 etc ki bish ysami il bolden bish yda5alhom ala ases asemihom so nest7a9 il mapping to convert it again
      region_index = mapping_dict_Region[region]
if top_pack in top_packs:
      top_pack_index = mapping_dict_TOP_PACK[top_pack]
my_dict={'REGION':region_index,'TOP_PACK':top_pack_index }  #n7othom f dict bish najm nconverti il dataframe 5ater matnajamsh ta3ml predict ala 7aja mosh dataframe
df = pd.DataFrame([my_dict])
if (st.button('predict')):
      y_pred= logreg.predict(df)
      threshold = 0.2
      y_pred_binary = (y_pred > threshold).astype(int)
      if (y_pred_binary == 0):
            st.write('Not Churn')
      else:
            st.write('Churn')
if (st.button('Visualize')):
      st.bar_chart(data['REGION'])