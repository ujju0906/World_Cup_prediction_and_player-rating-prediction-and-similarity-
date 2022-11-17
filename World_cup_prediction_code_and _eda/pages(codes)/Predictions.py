#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/davidcamilo0710/QATAR_2022_Prediction/blob/master/Predictions.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # INDEX
# - RESUME
# - GROUP STAGE PREDICTION
# - KNOCKOUT STAGE PREDICTION

# # RESUME

# ### Importing data and models

# In[25]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import warnings
import streamlit as st 
warnings.filterwarnings('ignore')


# In[26]:

st.title("Group Stages")
last_team_scores = pd.read_csv('data/last_team_scores.csv')
last_team_scores.tail()


# In[27]:


squad_stats = pd.read_csv('data/squad_stats.csv')
squad_stats.tail()


# In[28]:


group_matches = pd.read_csv('data/Qatar_group_stage.csv')
round_16 = group_matches.iloc[48:56, :]
quarter_finals = group_matches.iloc[56:60, :]
semi_finals = group_matches.iloc[60:62, :]
final = group_matches.iloc[62:63, :]
second_final = group_matches.iloc[63:64, :]
group_matches = group_matches.iloc[:48, :]
group_matches.tail()


# In[29]:


gs_model = joblib.load("models/groups_stage_prediction.pkl")


# In[30]:


ks_model = joblib.load("models/knockout_stage_prediction.pkl")


# In[31]:


team_group = group_matches.drop(['country2'], axis=1)
team_group = team_group.drop_duplicates().reset_index(drop=True)
team_group = team_group.rename(columns = {"country1":"team"})
team_group.head(5)


# ### Declaring important functions

# In[32]:


def matches(g_matches):
    g_matches.insert(2, 'potential1', g_matches['country1'].map(squad_stats.set_index('nationality_name')['potential']))
    g_matches.insert(3, 'potential2', g_matches['country2'].map(squad_stats.set_index('nationality_name')['potential']))
    g_matches.insert(4, 'rank1', g_matches['country1'].map(last_team_scores.set_index('team')['rank']))
    g_matches.insert(5, 'rank2', g_matches['country2'].map(last_team_scores.set_index('team')['rank']))
    pred_set = []

    for index, row in g_matches.iterrows():
        if row['potential1'] > row['potential2'] and abs(row['potential1'] - row['potential2']) > 2:          
            pred_set.append({'Team1': row['country1'], 'Team2': row['country2']})
        elif row['potential2'] > row['potential1'] and abs(row['potential2'] - row['potential1']) > 2:
            pred_set.append({'Team1': row['country2'], 'Team2': row['country1']})
        else:
            if row['rank1'] > row['rank2']:
                pred_set.append({'Team1': row['country1'], 'Team2': row['country2']})
            else:
                pred_set.append({'Team1': row['country2'], 'Team2': row['country1']})
   
    pred_set = pd.DataFrame(pred_set)
    pred_set.insert(2, 'Team1_FIFA_RANK', pred_set['Team1'].map(last_team_scores.set_index('team')['rank']))
    pred_set.insert(3, 'Team2_FIFA_RANK', pred_set['Team2'].map(last_team_scores.set_index('team')['rank']))
    pred_set.insert(4, 'Team1_Goalkeeper_Score', pred_set['Team1'].map(last_team_scores.set_index('team')['goalkeeper_score']))
    pred_set.insert(5, 'Team2_Goalkeeper_Score', pred_set['Team2'].map(last_team_scores.set_index('team')['goalkeeper_score']))
    pred_set.insert(6, 'Team1_Defense', pred_set['Team1'].map(last_team_scores.set_index('team')['defense_score']))
    pred_set.insert(7, 'Team1_Offense', pred_set['Team1'].map(last_team_scores.set_index('team')['offense_score']))
    pred_set.insert(8, 'Team1_Midfield', pred_set['Team1'].map(last_team_scores.set_index('team')['midfield_score']))
    pred_set.insert(9, 'Team2_Defense', pred_set['Team2'].map(last_team_scores.set_index('team')['defense_score']))
    pred_set.insert(10, 'Team2_Offense', pred_set['Team2'].map(last_team_scores.set_index('team')['offense_score']))
    pred_set.insert(11, 'Team2_Midfield', pred_set['Team2'].map(last_team_scores.set_index('team')['midfield_score']))
    return pred_set


# In[33]:


def print_results(dataset, y_pred, matches, proba):
  results = []
  for i in range(dataset.shape[0]):
      st.text('')
      if y_pred[i] == 2:
          st.header(matches.iloc[i, 0] + " vs. " + matches.iloc[i, 1] + " => Draw")
          results.append({'result': 'Draw'})
      elif y_pred[i] == 1:
          st.header(matches.iloc[i, 0] + " vs. " + matches.iloc[i, 1] + " => Winner: " + dataset.iloc[i, 0])
          results.append({'result': dataset.iloc[i, 0]})
      else:
          st.header(matches.iloc[i, 0] + " vs. " + matches.iloc[i, 1] + " => Winner: " + dataset.iloc[i, 1])
          results.append({'result': dataset.iloc[i, 1]})
      try:
          st.text('Probability of ' + dataset.iloc[i, 0] + ' winning: ')
          st.text((proba[i][1]))
          st.text('Probability of Draw: ')
          st.text(proba[i][2])
          st.text('Probability of ' + dataset.iloc[i, 1] + ' winning: ')
          st.text( proba[i][0])
      except:

          st.text('Probability of ' + dataset.iloc[i, 1] + ' winning: ')
          st.text(proba[i][0])   
          st.text("")
  results = pd.DataFrame(results)
  matches = pd.concat([matches.group, results], axis=1)
  return matches


# In[34]:


def winner_to_match(round, prev_match):
    round.insert(0, 'c1', round['country1'].map(prev_match.set_index('group')['result']))
    round.insert(1, 'c2', round['country2'].map(prev_match.set_index('group')['result']))
    round = round.drop(['country1','country2'], axis=1)
    round = round.rename(columns={'c1':'country1', 'c2':'country2'}).reset_index(drop=True)
    return round


# In[35]:


def prediction_knockout(round):
    dataset_round = matches(round)
    prediction_round = ks_model.predict(dataset_round)
    proba_round = ks_model.predict_proba(dataset_round)
    results_round = print_results(dataset_round, prediction_round, round, proba_round)
    return results_round


# In[36]:


def center_str(round):
    spaces = ['',' ','  ','   ','    ','     ',]
    for j in range(2):
        for i in range(round.shape[0]):
            if (13 - len(round.iloc[i, j])) % 2 == 0:
                round.iloc[i, j] = spaces[int((13 - len(round.iloc[i, j])) / 2)] + round.iloc[i, j] + spaces[int((13 - len(round.iloc[i, j])) / 2)]
            else:
                round.iloc[i, j] = spaces[int(((13 - len(round.iloc[i, j])) / 2) - 0.5)] + round.iloc[i, j] + spaces[int(((13 - len(round.iloc[i, j])) / 2) + 0.5)]
    return round


# In[37]:


def center2(a):
    spaces = ['',' ','  ','   ','    ','     ','      ','       ','        ','         ','          ','           ','            ','             ','              ','               ','                ','                 ','                  ','                   ','                    ']
    if (29 - len(a)) % 2 == 0:
        a = spaces[int((29 - len(a)) / 2)] + a + spaces[int((29 - len(a)) / 2)]
    else:
        a = spaces[int(((29 - len(a)) / 2) - 0.5)] + a + spaces[int(((29 - len(a)) / 2) + 0.5)]
    return a


# # GROUP STAGE PREDICTION

# In[38]:


dataset_groups = matches(group_matches)
dataset_groups.tail()


# In[39]:


prediction_groups = gs_model.predict(dataset_groups)
proba = gs_model.predict_proba(dataset_groups)
results = print_results(dataset_groups, prediction_groups, group_matches, proba)


# In[40]:


team_group['points'] = 0
team_group
for i in range(results.shape[0]):
    for j in range(team_group.shape[0]):
        if results.iloc[i, 1] == team_group.iloc[j, 0]:
            team_group.iloc[j, 2] += 3


# **Points Table:** only the best two from each group advance to the next phase

# In[41]:


st.dataframe(team_group.groupby(['group','team']).mean().astype(int))


# # KNOCKOUT STAGE PREDICTION
# **Round of 16**

# In[42]:


round_of_16 = team_group[team_group['points'] > 5].reset_index(drop=True)
round_of_16['group'] = (4 - 1/3 * round_of_16.points).astype(int).astype(str) + round_of_16.group 
round_of_16 = round_of_16.rename(columns = {"team":"result"})
    
round_16 = winner_to_match(round_16, round_of_16)
st.title("KNOCKOUT STAGES")
results_round_16 = prediction_knockout(round_16)


# **Quarterfinals**

# In[43]:


quarter_finals = winner_to_match(quarter_finals, results_round_16)
results_quarter_finals = prediction_knockout(quarter_finals)


# **Semi-final**

# In[44]:


semi_finals = winner_to_match(semi_finals, results_quarter_finals)
results_finals = prediction_knockout(semi_finals)


# **Final**

# In[45]:


final = winner_to_match(final, results_finals)
winner = prediction_knockout(final)


# **Third place**

# In[46]:


second = results_finals[~results_finals.result.isin(winner.result)]
results_finals_3 = results_quarter_finals[~results_quarter_finals.result.isin(results_finals.result)]
results_finals_3.iloc[0, 0]='z1'
results_finals_3.iloc[1, 0]='z2'
second_final = winner_to_match(second_final, results_finals_3)
third = prediction_knockout(second_final)


# **Printing Tournament Table**

# In[47]:


round_16 = center_str(round_16)
quarter_finals = center_str(quarter_finals)
semi_finals = center_str(semi_finals)
final = center_str(final)
group_matches = center_str(group_matches)

st.image("/Users/ujwal_nischal/Downloads/World_cup_prediction/pred.png")
# In[48]:


print(round_16.iloc[0, 0]+'━━━━┓                                                                                                                             ┏━━━━'+round_16.iloc[4, 0])
print('                 ┃                                                                                                                             ┃')
print('                 ┃━━━━'+quarter_finals.iloc[0, 0]+'━━━━┓                                                                                 ┏━━━━'+quarter_finals.iloc[2, 0]+'━━━━┃')
print('                 ┃                     ┃                                                                                 ┃                     ┃')
print(round_16.iloc[0, 1]+'━━━━┛                     ┃                                                                                 ┃                     ┗━━━━'+round_16.iloc[4, 1])
print('                                       ┃━━━━'+semi_finals.iloc[0, 0]+'━━━━┓                                     ┏━━━━'+semi_finals.iloc[1, 0]+'━━━━┃')
print(round_16.iloc[1, 0]+'━━━━┓                     ┃                     ┃                                     ┃                     ┃                     ┏━━━━'+round_16.iloc[5, 0])
print('                 ┃                     ┃                     ┃                                     ┃                     ┃                     ┃')
print('                 ┃━━━━'+quarter_finals.iloc[0, 1]+'━━━━┛                     ┃                                     ┃                     ┗━━━━'+quarter_finals.iloc[2, 1]+'━━━━┃')
print('                 ┃                                           ┃                                     ┃                                           ┃')
print(round_16.iloc[1, 1]+'━━━━┛                                           ┃                                     ┃                                           ┗━━━━'+round_16.iloc[5, 1])
print('                                                             ┃━━━━'+final.iloc[0, 0]+'vs.'+final.iloc[0, 1]+'━━━━┃')
print(round_16.iloc[2, 0]+'━━━━┓                                           ┃                                     ┃                                           ┏━━━━'+round_16.iloc[6, 0])
print('                 ┃                                           ┃                                     ┃                                           ┃')
print('                 ┃━━━━'+quarter_finals.iloc[1, 0]+'━━━━┓                     ┃                                     ┃                     ┏━━━━'+quarter_finals.iloc[3, 0]+'━━━━┃')
print('                 ┃                     ┃                     ┃                                     ┃                     ┃                     ┃')
print(round_16.iloc[2, 1]+'━━━━┛                     ┃                     ┃                                     ┃                     ┃                     ┗━━━━'+round_16.iloc[6, 1])
print('                                       ┃━━━━'+semi_finals.iloc[0, 1]+'━━━━┛                                     ┗━━━━'+semi_finals.iloc[1, 1]+'━━━━┃')
print(round_16.iloc[3, 0]+'━━━━┓                     ┃                                                                                 ┃                     ┏━━━━'+round_16.iloc[7, 0])
print('                 ┃                     ┃                                                                                 ┃                     ┃')
print('                 ┃━━━━'+quarter_finals.iloc[1, 1]+'━━━━┛                                                                                 ┗━━━━'+quarter_finals.iloc[3, 1]+'━━━━┃')
print('                 ┃                                                                                                                             ┃')
print(round_16.iloc[3, 1]+'━━━━┛                                                                                                                             ┗━━━━'+round_16.iloc[7, 1])
print("                                                                 "+center2("\U0001F947"+winner.iloc[0, 1]))
print("                                                                 "+center2("\U0001F948"+second.iloc[0, 1]))
print("                                                                 "+center2("\U0001F949"+third.iloc[0, 1]))

