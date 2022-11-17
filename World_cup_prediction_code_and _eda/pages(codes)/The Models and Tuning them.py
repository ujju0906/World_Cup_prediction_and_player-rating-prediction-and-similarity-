

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import streamlit as st
from sklearn.model_selection import train_test_split 
from sklearn.metrics import plot_confusion_matrix


# In[2]:


final_df = pd.read_csv('data/training.csv')
final_df.tail()


st.markdown(" GROUP STAGE MODELING")
st.markdown("XGBoost Model evaluation with tuned hyperparameters")
# ### Choosing a model

# In[3]:


# I save the original data frame in a flag to then train the final pipeline
pipe_DF = final_df
# Dummies for categorical columns
final_df = pd.get_dummies(final_df)


# I split the dataset into training, testing and validation.

# In[4]:


X = final_df.drop('Team1_Result',axis=1)
y = final_df['Team1_Result']
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
X_hold_test, X_test, y_hold_test, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42)


#  Scaling

# In[5]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_hold_test = scaler.transform(X_hold_test)


# Defining function to display the confusion matrix quickly.

# In[6]:


from sklearn.metrics import classification_report,ConfusionMatrixDisplay
def metrics_display(model):
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    st.text(classification_report(y_test,y_pred))
    
    plot_confusion_matrix(model, X_test, y_test)
    st.pyplot()
    

    
    

# # * **Random Forest**

# # In[7]:


from sklearn.ensemble import RandomForestClassifier
# metrics_display(RandomForestClassifier())


# # * **Ada Boost Classifier**

# # In[8]:


from sklearn.ensemble import AdaBoostClassifier
# metrics_display(AdaBoostClassifier())


# # * **XGB Boost**

# # In[9]:


# #get_ipython().system('pip3 install xgboost')


# # In[10]:


from xgboost import XGBClassifier
# metrics_display(XGBClassifier(use_label_encoder=False))


# # * **Neural network**
# # 
# # 

# # In[11]:


# # import keras
# # from keras import Sequential
# # from keras.layers import Dense,Dropout
# # from keras import Input

# # X_train.shape


# # # In[12]:


# # model = Sequential()
# # model.add(Input(shape=(404,)))
# # model.add(Dense(300,activation='relu'))
# # model.add(Dropout(0.3))
# # model.add(Dense(200,activation='relu'))
# # model.add(Dropout(0.3))
# # model.add(Dense(100,activation='relu'))
# # model.add(Dropout(0.3))
# # model.add(Dense(3,activation='softmax'))
# # model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# # model.fit(X_train,y_train,epochs=10,validation_split=0.2)

# # y_pred1 = model.predict(X_test)
# # y_pred1 = np.argmax(y_pred1,axis=1)
# # st.write(classification_report(y_test,y_pred1))
# # ConfusionMatrixDisplay.from_predictions(y_test,y_pred1)


# # The XGBoost model performs better than the others, so I will tune its hyperparameters and evaluate the performance based on the validation dataset.

# # ### XGB Boost - Tuning & Hold-out Validation

# # In[13]:


# from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# # # Make a dictionary of hyperparameter values to search
# search_space = {
#     "n_estimators" : [200,250,300,350,400,450,500],
#     "max_depth" : [3,4,5,6,7,8,9],
#     "gamma" : [0.001,0.01,0.1],
#     "learning_rate" : [0.001,0.01,0.1]
# }


# # # In[14]:


# # # make a GridSearchCV object
# from sklearn.model_selection import GridSearchCV
# GS = GridSearchCV(estimator = XGBClassifier(use_label_encoder=False),
#                   param_grid = search_space,
#                   scoring = 'accuracy',
#                   cv = 5,
#                   verbose = 4)


# # Uncomment the following line to enable the tuning. The best result I found was: gamma = 0.01, learning_rate = 0.01, n_estimators = 300, max_depth = 4

# # In[15]:


# #GS.fit(X_train,y_train)


# # To get only the best hyperparameter values

# # In[16]:


# #st.write(GS.best_params_) 


# # Initially, I validate the model with its default parameters, and then I will validate it with its tuned parameters.

# # * **Default Hyperparameters**

# # In[17]:


# model = XGBClassifier()
# model.fit(X_train,y_train)
# y_pred = model.predict(X_hold_test)
# st.text(classification_report(y_hold_test,y_pred))
# plot_confusion_matrix(model, X_hold_test, y_hold_test)
# st.pyplot()
    







# In[18]:

from xgboost import XGBClassifier
model = XGBClassifier(use_label_encoder = False, gamma = 0.01, learning_rate = 0.01, n_estimators = 300, max_depth = 4)
model.fit(X_train,y_train)
y_pred = model.predict(X_hold_test)
st.text(classification_report(y_hold_test,y_pred))
plot_confusion_matrix(model, X_hold_test, y_hold_test)

st.pyplot()


# The model improves a bit, so I will create a pipe to use the model later easily.

# ### Creating a pipeline for the XGB model

# In[19]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
column_trans = make_column_transformer(
    (OneHotEncoder(),['Team1', 'Team2']),remainder='passthrough')

pipe_X = pipe_DF.drop('Team1_Result',axis=1)
pipe_y = pipe_DF['Team1_Result']

from sklearn.pipeline import make_pipeline
pipe_League = make_pipeline(column_trans,StandardScaler(with_mean=False),XGBClassifier(use_label_encoder=False, gamma= 0.01, learning_rate= 0.01, n_estimators= 300, max_depth= 4))
pipe_League.fit(pipe_X,pipe_y)


# In[21]:


import joblib
joblib.dump(pipe_League,"models/groups_stage_prediction.pkl")


st.markdown("KNOCKOUT STAGE MODELING")

# ### Choosing the model 
# 
# Removing Draw status.

# In[22]:


knock_df = pipe_DF[pipe_DF['Team1_Result'] != 2]


# In[23]:


pipe_knock_df = knock_df
knock_df = pd.get_dummies(knock_df)
X = knock_df.drop('Team1_Result',axis=1)
y = knock_df['Team1_Result']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_hold_test, X_test, y_hold_test, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42)


# * **Ada Boost Classifier**

# In[24]:


# metrics_display(AdaBoostClassifier())


# *   **Random Forest**
# 
# 
# 

# In[25]:


# metrics_display(RandomForestClassifier())


# * **XGB Boost**

# In[26]:


# metrics_display(XGBClassifier(use_label_encoder=False))


# * **Neural network**

# In[27]:


# X_train.shape


# # In[28]:
# model = Sequential()
# model.add(Input(shape=(399,)))
# model.add(Dense(300,activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(200,activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(100,activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(2,activation='softmax'))
# model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(X_train,y_train,epochs=10,validation_split=0.2)

# y_pred1 = model.predict(X_test)
# y_pred1 = np.argmax(y_pred1,axis=1)
# st.write(classification_report(y_test,y_pred1))
# ConfusionMatrixDisplay.from_predictions(y_test,y_pred1)


# All models have very similar performance. Therefore I will tune the Random Forest model and the XGB Boost.

# ### Random Forest - Tuning & Hold-out Validation 

# In[29]:


search_space = {
    "max_depth" : [11,12,13,14,15,16],
    "max_leaf_nodes" : [170,180,190,200,210,220,230],
    "min_samples_leaf" : [3,4,5,6,7,8],
    "n_estimators" : [310,320,330,340,350]
}


# In[30]:


# GS = GridSearchCV(estimator = RandomForestClassifier(),
#                   param_grid = search_space,
#                   scoring = 'accuracy',
#                   cv = 5,
#                   verbose = 4)


# Uncomment the following lines to enable the tuning. The best result I found was: max_depth = 16, n_estimators = 320, max_leaf_nodes = 190, min_samples_leaf = 5

# In[31]:


#GS.fit(X_train,y_train)


# In[32]:


#st.write(GS.best_params_)


# * **Default Hyperparameters**

# In[33]:


# model = RandomForestClassifier()
# model.fit(X_train,y_train)
# y_pred = model.predict(X_hold_test)
# st.text(classification_report(y_hold_test,y_pred))

# plot_confusion_matrix(model, X_hold_test, y_hold_test)

# st.pyplot()


# * **Tuned Hyperparameters**

# In[34]:

st.markdown(" Random forest with tuned parameters")

model = RandomForestClassifier(max_depth= 16, n_estimators=320, max_leaf_nodes= 190, min_samples_leaf= 5)
model.fit(X_train,y_train)
y_pred = model.predict(X_hold_test)
st.text(classification_report(y_hold_test,y_pred))
plot_confusion_matrix(model, X_hold_test, y_hold_test)

st.pyplot()


# The Random Forest greatly improves performance with the tuned hyperparameters; let's see the XGB Boost model.

# ### XGB Boost - Tuning & Hold-out Validation

# In[35]:


search_space = {
    "n_estimators" : [300,350,400,450,500,550,600],
    "max_depth" : [3,4,5,6,7,8,9],
    "gamma" : [0.001,0.01,0.1],
    "learning_rate" : [0.001,0.01]
}


# In[36]:


# GS = GridSearchCV(estimator = XGBClassifier(use_label_encoder=False),
#                   param_grid = search_space,
#                   scoring = 'accuracy',
#                   cv = 5,
#                   verbose = 4)


# In[37]:


#GS.fit(X_train,y_train)


# In[38]:


#st.write(GS.best_params_) # to get only the best hyperparameter values that we searched for


# Uncomment the following lines to enable the tuning. The best result I found was: gamma = 0.01, learning_rate = 0.01, max_depth = 5, n_estimators = 500

# * **Default Hyperparameters**

# # In[39]:


# model = XGBClassifier()
# model.fit(X_train,y_train)
# y_pred = model.predict(X_hold_test)
# st.text(classification_report(y_hold_test,y_pred))
# plot_confusion_matrix(model, X_hold_test, y_hold_test)

# st.pyplot()


# * **Tuned Hyperparameters**

# In[40]:
st.markdown("XGBoost with tuned parameters")

model = XGBClassifier(gamma=0.01,learning_rate=0.01, max_depth=5, n_estimators=500)
model.fit(X_train,y_train)
y_pred = model.predict(X_hold_test)
st.text(classification_report(y_hold_test,y_pred))
plot_confusion_matrix(model, X_hold_test, y_hold_test)

st.pyplot()



# The model does not improve notably. However, it does improve compared to the Random Forest.

# ### Creating a pipeline for the XGB Boost model

# In[41]:


pipe_X = pipe_knock_df.drop('Team1_Result',axis=1)
pipe_y = pipe_knock_df['Team1_Result']
pipe_knock = make_pipeline(column_trans,StandardScaler(with_mean=False),XGBClassifier(gamma=0.01,learning_rate=0.01, max_depth=5, n_estimators=500))
pipe_knock.fit(pipe_X,pipe_y)


# In[42]:


joblib.dump(pipe_knock,"models/knockout_stage_prediction.pkl")


# In[ ]:




