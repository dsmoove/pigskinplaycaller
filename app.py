#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Load Modules
import pandas as pd
import glob
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn import model_selection
import streamlit as st
import pickle

# path = r'C:\Users\admin\Documents\Flatiron\capstone\pigskin\data\weekly' # use your path
# all_files = glob.glob(path + "/*.csv")

# li = []

# for filename in all_files:
#     df = pd.read_csv(filename, index_col=None, header=0)
#     li.append(df)

# df_weekly = pd.concat(li, axis=0, ignore_index=True)


# In[3]:


# load Pickled Model
with open('model.pkl', 'rb') as f:
    gbr = pickle.load(f)


# In[4]:


#Loading Data
players = pd.read_csv('data/players.csv')
df_games = pd.read_csv('data/weekly/week1.csv')
df_plays = pd.read_csv('data/plays.csv')
df_fo = pd.read_csv('data/2018fo.csv')
df_coverages = pd.read_csv('data/coverages_week1.csv')
df_fastr = data = pd.read_csv(
    'https://github.com/guga31bb/nflfastR-data/blob/master/data/play_by_play_' \
    + str(2018) + '.csv.gz?raw=True',compression='gzip', low_memory=False)


# In[5]:


#Scale EPA to 0-1 min max
df_plays['scaled_epa'] = (df_plays['epa'] - min(df_plays['epa'])) / (max(df_plays['epa']) - min(df_plays['epa']))


# In[6]:


#Merge tables
df_plays_coverage = pd.merge(df_plays,df_coverages,left_on=['gameId','playId'],right_on=['gameId','playId'])
df_plays_final = pd.merge(df_plays_coverage,df_games,left_on=['gameId','playId'],right_on=['gameId','playId'])
#df_plays_final = pd.merge(df_plays_coverage,df_games,on='Id')
#df_plays_final


# In[7]:


#Limit our data to only positioning at ball snap
df_plays_final = df_plays_final[df_plays_final.event == 'ball_snap']


# In[8]:


df_plays_final = df_plays_final.drop_duplicates(subset='playId', keep="first")


# In[9]:


df_plays_new = pd.merge(df_plays_final,df_fastr[['old_game_id','play_id','weather', 'roof', 'surface','temp','wind']], how = 'left',left_on=['gameId','playId'],right_on=['old_game_id','play_id'])


# In[10]:


df_plays_new.drop(['playDirection','gameId', 'playId', 'playDescription', 'quarter', 'possessionTeam', 'playType','yardlineSide','yardlineNumber','preSnapVisitorScore','preSnapHomeScore','gameClock','absoluteYardlineNumber','penaltyCodes','penaltyJerseyNumbers','passResult','offensePlayResult','playResult','epa','isDefensivePI','time','x','y','s','a','dis','o','dir','event','nflId','displayName','jerseyNumber','position','frameId','team','gameId','typeDropback','old_game_id','weather'], axis=1, inplace=True)


# In[11]:


# Title our App
st.write("""
# Pigskin Playcaller
This app will call your plays!
""")


# In[12]:


page_bg_img = '''
<style>
body {
background-image: url("https://images.fineartamerica.com/images-medium-large-5/1-yard-numbers-and-line-on-american-football-field-at-night-howard-sun.jpg");
background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)


# In[13]:


# Create sidebar and header
st.sidebar.header('User Input Parameters')


# In[14]:


# Define function that will allow user input into the app
def user_input_features():
    down = st.sidebar.selectbox('Current Down', df_plays_new['down'].unique())
    yardage = st.sidebar.slider('Yards Left to First Down', 0, 100)
    wind = st.sidebar.slider('Wind MPH', 0, 100)
    temp = st.sidebar.slider('Current Temperature', 0, 100)
    roof = st.sidebar.selectbox('Stadium', df_plays_new['roof'].unique())
    formation = st.sidebar.selectbox('Offensive Formation', df_plays_new['offenseFormation'].unique())
    personnelO = st.sidebar.selectbox('Offensive Personnel', df_plays_new['personnelO'].unique())
    surface = st.sidebar.selectbox('Surface', df_plays_new['surface'].unique())
    data = {'down': down,
            'yardsToGo' : yardage,
            'temp': temp,
            'roof': roof,
            'offenseFormation': formation,
            'personnelO': personnelO,
            'surface': surface}
    features = pd.DataFrame(data, index=[0])
    return features


# In[15]:


# Run our function
df = user_input_features()


# In[16]:


# Write our inputs
st.subheader('User Input parameters')
st.write(df)


# In[17]:


# Dummy our inputs
oform_dummies = pd.get_dummies(df['offenseFormation'])
roof_dummies = pd.get_dummies(df['roof'])
opersonnel_dummies = pd.get_dummies(df['personnelO'])
surface_dummies = pd.get_dummies(df['surface'])
df_model = pd.concat([df, oform_dummies,opersonnel_dummies,roof_dummies,surface_dummies], axis=1)


# In[18]:


# Define our columns
cols = ['down',
 'yardsToGo',
 'defendersInTheBox',
 'numberOfPassRushers',
 'temp',
 'wind',
 'I_FORM',
 'PISTOL',
 'SHOTGUN',
 'SINGLEBACK',
 'WILDCAT',
 '0 RB, 2 TE, 3 WR',
 '1 RB, 0 TE, 4 WR',
 '1 RB, 1 TE, 2 WR,1 DL',
 '1 RB, 1 TE, 3 WR',
 '1 RB, 2 TE, 2 WR',
 '1 RB, 3 TE, 1 WR',
 '2 QB, 0 RB, 1 TE, 3 WR',
 '2 QB, 1 RB, 1 TE, 2 WR',
 '2 RB, 0 TE, 3 WR',
 '2 RB, 1 TE, 2 WR',
 '2 RB, 2 TE, 1 WR',
 '2 RB, 3 TE, 0 WR',
 '3 RB, 1 TE, 1 WR',
 '6 OL, 1 RB, 1 TE, 2 WR',
 '6 OL, 1 RB, 2 TE, 1 WR',
 '6 OL, 2 RB, 0 TE, 2 WR',
 'Cover 1 Man',
 'Cover 2 Man',
 'Cover 2 Zone',
 'Cover 3 Zone',
 'Cover 4 Zone',
 'Cover 6 Zone',
 'Prevent Zone',
 'dome',
 'outdoors',
 'grass',
 'sportturf']


# In[19]:


# Reindex our columns
df_run = df_model.reindex(cols, axis=1, fill_value=0)


# In[20]:


# Default defenders in box + pass rushers
df_run.at[0,'defendersInTheBox']= 6
df_run.at[0,'numberOfPassRushers']= 4


# In[21]:


# Duplicate rows
df_coverages = pd.concat([df_run]*7, ignore_index=True)


# In[22]:


# Run model with each coverage type
df_coverages.at[0,'Cover 1 Man']= 1
df_coverages.at[1,'Cover 2 Man'] = 1
df_coverages.at[2,'Cover 2 Zone'] = 1
df_coverages.at[3,'Cover 3 Zone'] = 1
df_coverages.at[4,'Cover 4 Zone'] = 1
df_coverages.at[5,'Cover 6 Zone'] = 1
df_coverages.at[6,'Prevent Zone'] = 1


# In[23]:


# Predict
prediction = gbr.predict(df_coverages)


# In[24]:


# Append to original DF
df_coverages['prediction1'] = prediction


# In[25]:


# Define lowest value
low_val = df_coverages[df_coverages['prediction1']==df_coverages['prediction1'].min()]
best_coverage = low_val.loc[:,"Cover 1 Man":"Prevent Zone"].eq(1).idxmax(1)


# In[26]:


st.write(best_coverage)

