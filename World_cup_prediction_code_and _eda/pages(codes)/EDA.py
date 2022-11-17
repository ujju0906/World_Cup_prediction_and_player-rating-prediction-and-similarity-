
# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import streamlit as st
warnings.filterwarnings('ignore')


st.title("Visualisation of various characteristics")
# In[2]:


df = pd.read_csv('data/international_matches.csv', parse_dates=['date'])
df.tail()


# In[3]:


# df.columns


# In[4]:


df.isnull().sum()


# # PRE-ANALYSIS
# The dataset has a lot of blank fields that need to be fixed.
# However, before modifying any field, I want to analyze the teams' qualifications on the last FIFA date (June 2022). This is important because, from these qualifications, I will create the inference dataset that enters the machine learning algorithm that predicts the World Cup matches.

# ### Top 10 FIFA Ranking
# Top 10 national teams to date FIFA June 2022.
# **ref:** https://www.fifa.com/fifa-world-ranking/men?dateId=id13603

# In[5]:


fifa_rank = df[['date','home_team', 'away_team', 'home_team_fifa_rank', 'away_team_fifa_rank']]
home = fifa_rank[['date', 'home_team', 'home_team_fifa_rank']].rename(columns={"home_team":"team", "home_team_fifa_rank":"rank"})
away = fifa_rank[['date', 'away_team', 'away_team_fifa_rank']].rename(columns={"away_team":"team", "away_team_fifa_rank":"rank"})
fifa_rank = pd.concat([home, away])
#Select each country latest match
fifa_rank = fifa_rank.sort_values(['team', 'date'], ascending=[True, False])
last_rank = fifa_rank
fifa_rank_top10 = fifa_rank.groupby('team').first().sort_values('rank', ascending=True)[0:10].reset_index()
# fifa_rank_top10


# ### Top 10 teams with the highest winning percentage at home and away

# In[6]:


def home_percentage(team):
    score = len(df[(df['home_team'] == team) & (df['home_team_result'] == "Win")]) / len(df[df['home_team'] == team]) * 100
    return round(score)
def away_percentage(team):
    score = len(df[(df['away_team'] == team) & (df['home_team_result'] == "Lose")]) / len(df[df['away_team'] == team]) * 100
    return round(score)


# In[7]:


fifa_rank_top10['Home_win_Per'] = np.vectorize(home_percentage)(fifa_rank_top10['team'])
fifa_rank_top10['Away_win_Per'] = np.vectorize(away_percentage)(fifa_rank_top10['team'])
fifa_rank_top10['Average_win_Per'] = round((fifa_rank_top10['Home_win_Per'] + fifa_rank_top10['Away_win_Per']) / 2)
fifa_rank_win = fifa_rank_top10.sort_values('Average_win_Per', ascending = False)
fifa_rank_win


# ### Top 10 attacking teams in the last FIFA date

# In[8]:


fifa_offense = df[['date', 'home_team', 'away_team', 'home_team_mean_offense_score', 'away_team_mean_offense_score']]
home = fifa_offense[['date', 'home_team', 'home_team_mean_offense_score']].rename(columns={"home_team":"team", "home_team_mean_offense_score":"offense_score"})
away = fifa_offense[['date', 'away_team', 'away_team_mean_offense_score']].rename(columns={"away_team":"team", "away_team_mean_offense_score":"offense_score"})
fifa_offense = pd.concat([home, away])
fifa_offense = fifa_offense.sort_values(['date', 'team'],ascending=[False, True])
last_offense = fifa_offense
fifa_offense_top10 = fifa_offense.groupby('team').first().sort_values('offense_score', ascending=False)[0:10].reset_index()
st.dataframe(fifa_offense_top10)


# In[9]:


sns.barplot(data=fifa_offense_top10, x='offense_score', y='team', color="#7F1431")
plt.xlabel('Offense Score', size = 20) 
plt.ylabel('Team', size = 20) 
plt.title("Top 10 Attacking teams");
st.pyplot()

# ### Top 10 Midfield teams in the last FIFA date

# In[10]:


fifa_midfield = df[['date', 'home_team', 'away_team', 'home_team_mean_midfield_score', 'away_team_mean_midfield_score']]
home = fifa_midfield[['date', 'home_team', 'home_team_mean_midfield_score']].rename(columns={"home_team":"team", "home_team_mean_midfield_score":"midfield_score"})
away = fifa_midfield[['date', 'away_team', 'away_team_mean_midfield_score']].rename(columns={"away_team":"team", "away_team_mean_midfield_score":"midfield_score"})
fifa_midfield = pd.concat([home,away])
fifa_midfield = fifa_midfield.sort_values(['date','team'],ascending=[False,True])
last_midfield = fifa_midfield
fifa_midfield_top10 = fifa_midfield.groupby('team').first().sort_values('midfield_score',ascending=False)[0:10].reset_index()
st.dataframe(fifa_midfield_top10)


# In[11]:


sns.barplot(data=fifa_midfield_top10, x='midfield_score', y='team', color="#7F1431")
plt.xlabel('Midfield Score', size = 20) 
plt.ylabel('Team', size = 20) 
plt.title("Top 10 Midfield teams");
st.pyplot()

# ### Top 10 defending teams in the last FIFA date

# In[12]:


fifa_defense = df[['date', 'home_team', 'away_team', 'home_team_mean_defense_score', 'away_team_mean_defense_score']]
home = fifa_defense[['date', 'home_team', 'home_team_mean_defense_score']].rename(columns={"home_team":"team", "home_team_mean_defense_score":"defense_score"})
away = fifa_defense[['date', 'away_team', 'away_team_mean_defense_score']].rename(columns={"away_team":"team", "away_team_mean_defense_score":"defense_score"})
fifa_defense = pd.concat([home, away])
fifa_defense = fifa_defense.sort_values(['date', 'team'],ascending=[False, True])
last_defense = fifa_defense 
fifa_defense_top10 = fifa_defense.groupby('team').first().sort_values('defense_score', ascending = False)[0:10].reset_index()
st.dataframe(fifa_defense_top10)


# In[13]:


sns.barplot(data = fifa_defense_top10, x='defense_score', y='team')
plt.xlabel('Defense Score', size = 20) 
plt.ylabel('Team', size = 20) 
plt.title("Top 10 Defense Teams")

st.pyplot()
# ### Do Home teams have any advantage?

# In[14]:


# Select all matches played at non-neutral locations
home_team_advantage = df[df['neutral_location'] == False]['home_team_result'].value_counts(normalize = True)

# Plot
fig, axes = plt.subplots(1, 1, figsize=(8,8))
ax =plt.pie(home_team_advantage  ,labels = ['Win', 'Lose', 'Draw'], autopct='%.0f%%')
plt.title('Home team match result', fontsize = 15)
plt.show()

st.pyplot(plt)

# As the graph shows, the home team has an advantage over the away team. This is due to factors such as the fans, the weather and the confidence of the players. For this reason, in the World Cup, those teams that sit at home will have an advantage.

# # DATA PREPARATION AND FEATURE ENGINEERING
# In this section, I will fill in the empty fields in the dataset and clean up the data for teams that did not qualify for the World Cup. Then, I will use the correlation matrix to choose the characteristics that will define the training dataset of the Machine Learning model. Finally, I will use the ratings of the teams in their last match to define the "Last Team Scores" dataset (i.e., the dataset that I will use to predict the World Cup matches).

# ### Analyze and fill na's

# In[15]:


df.isnull().sum()


# In[16]:


# We can fill mean for na's in goal_keeper_score
df[df['home_team'] == "Brazil"]['home_team_goalkeeper_score'].describe()


# In[17]:


df['home_team_goalkeeper_score'] = round(df.groupby("home_team")["home_team_goalkeeper_score"].transform(lambda x: x.fillna(x.mean())))
df['away_team_goalkeeper_score'] = round(df.groupby("away_team")["away_team_goalkeeper_score"].transform(lambda x: x.fillna(x.mean())))


# In[18]:


# We can fill mean for na's in defense score
df[df['away_team'] == "Uruguay"]['home_team_mean_defense_score'].describe()


# In[19]:


df['home_team_mean_defense_score'] = round(df.groupby('home_team')['home_team_mean_defense_score'].transform(lambda x : x.fillna(x.mean())))
df['away_team_mean_defense_score'] = round(df.groupby('away_team')['away_team_mean_defense_score'].transform(lambda x : x.fillna(x.mean())))


# In[20]:


# We can fill mean for na's in offense score
df[df['away_team'] == "Uruguay"]['home_team_mean_offense_score'].describe()


# In[21]:


df['home_team_mean_offense_score'] = round(df.groupby('home_team')['home_team_mean_offense_score'].transform(lambda x : x.fillna(x.mean())))
df['away_team_mean_offense_score'] = round(df.groupby('away_team')['away_team_mean_offense_score'].transform(lambda x : x.fillna(x.mean())))


# In[22]:


# We can fill mean for na's in midfield score
st.table(df[df['away_team'] == "Uruguay"]['home_team_mean_midfield_score'].describe())


# In[23]:


df['home_team_mean_midfield_score'] = round(df.groupby('home_team')['home_team_mean_midfield_score'].transform(lambda x : x.fillna(x.mean())))
df['away_team_mean_midfield_score'] = round(df.groupby('away_team')['away_team_mean_midfield_score'].transform(lambda x : x.fillna(x.mean())))


# In[24]:


df.isnull().sum()


# In[25]:


# Teams are not available in FIFA game itself, so they are not less than average performing teams, so giving a average score of 50 for all.
df.fillna(50,inplace=True)


# ### Filter the teams participating in QATAR - World cup 2022

# In[26]:


list_2022 = ['Qatar', 'Germany', 'Denmark', 'Brazil', 'France', 'Belgium', 'Croatia', 'Spain', 'Serbia', 'England', 'Switzerland', 'Netherlands', 'Argentina', 'IR Iran', 'Korea Republic', 'Japan', 'Saudi Arabia', 'Ecuador', 'Uruguay', 'Canada', 'Ghana', 'Senegal', 'Portugal', 'Poland', 'Tunisia', 'Morocco', 'Cameroon', 'USA', 'Mexico', 'Wales', 'Australia', 'Costa Rica']
final_df = df[(df["home_team"].apply(lambda x: x in list_2022)) | (df["away_team"].apply(lambda x: x in list_2022))]


# **Top 10 teams in QATAR 2022**

# In[27]:


rank = final_df[['date','home_team','away_team','home_team_fifa_rank', 'away_team_fifa_rank']]
home = rank[['date','home_team','home_team_fifa_rank']].rename(columns={"home_team":"team","home_team_fifa_rank":"rank"})
away = rank[['date','away_team','away_team_fifa_rank']].rename(columns={"away_team":"team","away_team_fifa_rank":"rank"})
rank = pd.concat([home,away])

# Select each country latest match
rank = rank.sort_values(['team','date'],ascending=[True,False])
rank_top10 = rank.groupby('team').first().sort_values('rank',ascending=True).reset_index()
rank_top10 = rank_top10[(rank_top10["team"].apply(lambda x: x in list_2022))][0:10]
st.dataframe(rank_top10)


# **Top 10 teams with the highest winning percentage in QATAR 2022**

# In[28]:


rank_top10['Home_win_Per'] = np.vectorize(home_percentage)(rank_top10['team'])
rank_top10['Away_win_Per'] = np.vectorize(away_percentage)(rank_top10['team'])
rank_top10['Average_win_Per'] = round((rank_top10['Home_win_Per'] + rank_top10['Away_win_Per'])/2)
rank_top10_Win = rank_top10.sort_values('Average_win_Per',ascending=False)
st.dataframe(rank_top10_Win)


# In[29]:


sns.barplot(data=rank_top10_Win,x='Average_win_Per',y='team',color="#7F1431")
plt.xticks()
plt.xlabel('Win Average', size = 20) 
plt.ylabel('Team', size = 20) 
plt.title('Top 10 QATAR 2022 teams with the highest winning percentage')
st.pyplot()


# ### Correlation Matrix

# In[30]:


# Mapping numeric values for home_team_result to find the correleations
final_df['home_team_result'] = final_df['home_team_result'].map({'Win':1, 'Draw':2, 'Lose':0})


# In[31]:


st.table(final_df.corr()['home_team_result'].sort_values(ascending=False))


# Dropping unnecessary colums.

# In[32]:


#Dropping unnecessary colums
final_df = final_df.drop(['date', 'home_team_continent', 'away_team_continent', 'home_team_total_fifa_points', 'away_team_total_fifa_points', 'home_team_score', 'away_team_score', 'tournament', 'city', 'country', 'neutral_location', 'shoot_out'],axis=1)


# In[33]:


final_df.columns


# In[34]:


# Change column names
final_df.rename(columns={"home_team":"Team1", "away_team":"Team2", "home_team_fifa_rank":"Team1_FIFA_RANK", 
                         "away_team_fifa_rank":"Team2_FIFA_RANK", "home_team_result":"Team1_Result", "home_team_goalkeeper_score":"Team1_Goalkeeper_Score",
                        "away_team_goalkeeper_score":"Team2_Goalkeeper_Score", "home_team_mean_defense_score":"Team1_Defense",
                        "home_team_mean_offense_score":"Team1_Offense", "home_team_mean_midfield_score":"Team1_Midfield",
                        "away_team_mean_defense_score":"Team2_Defense", "away_team_mean_offense_score":"Team2_Offense",
                        "away_team_mean_midfield_score":"Team2_Midfield"}, inplace=True)


# In[35]:


plt.figure(figsize=(10, 4), dpi=200)
sns.heatmap(final_df.corr(), annot=True)

st.pyplot()
# In[36]:


final_df.info()


# In[37]:


st.dataframe(final_df)


# Exporting the training dataset.

# In[38]:


final_df.to_csv("data/training.csv", index = False)


# ### Creating "Last Team Scores" dataset
# This dataset contains the qualifications of each team on the previous FIFA date and will be used to predict the World Cup matches.

# In[39]:


last_goalkeeper = df[['date', 'home_team', 'away_team', 'home_team_goalkeeper_score', 'away_team_goalkeeper_score']]
home = last_goalkeeper[['date', 'home_team', 'home_team_goalkeeper_score']].rename(columns={"home_team":"team", "home_team_goalkeeper_score":"goalkeeper_score"})
away = last_goalkeeper[['date', 'away_team', 'away_team_goalkeeper_score']].rename(columns={"away_team":"team", "away_team_goalkeeper_score":"goalkeeper_score"})
last_goalkeeper = pd.concat([home,away])

last_goalkeeper = last_goalkeeper.sort_values(['date', 'team'],ascending=[False, True])

list_2022 = ['Qatar', 'Germany', 'Denmark', 'Brazil', 'France', 'Belgium', 'Croatia', 'Spain', 'Serbia', 'England', 'Switzerland', 'Netherlands', 'Argentina', 'IR Iran', 'Korea Republic', 'Japan', 'Saudi Arabia', 'Ecuador', 'Uruguay', 'Canada', 'Ghana', 'Senegal', 'Portugal', 'Poland', 'Tunisia', 'Morocco', 'Cameroon', 'USA', 'Mexico', 'Wales', 'Australia', 'Costa Rica']

rank_qatar = last_rank[(last_rank["team"].apply(lambda x: x in list_2022))]
rank_qatar = rank_qatar.groupby('team').first().reset_index()
goal_qatar = last_goalkeeper[(last_goalkeeper["team"].apply(lambda x: x in list_2022))]
goal_qatar = goal_qatar.groupby('team').first().reset_index()
goal_qatar = goal_qatar.drop(['date'], axis = 1)
off_qatar = last_offense[(last_offense["team"].apply(lambda x: x in list_2022))]
off_qatar = off_qatar.groupby('team').first().reset_index()
off_qatar = off_qatar.drop(['date'], axis = 1)
mid_qatar = last_midfield[(last_midfield["team"].apply(lambda x: x in list_2022))]
mid_qatar = mid_qatar.groupby('team').first().reset_index()
mid_qatar = mid_qatar.drop(['date'], axis = 1)
def_qatar = last_defense[(last_defense["team"].apply(lambda x: x in list_2022))]
def_qatar = def_qatar.groupby('team').first().reset_index()
def_qatar = def_qatar.drop(['date'], axis = 1)

qatar = pd.merge(rank_qatar, goal_qatar, on = 'team')
qatar = pd.merge(qatar, def_qatar, on ='team')
qatar = pd.merge(qatar, off_qatar, on ='team')
qatar = pd.merge(qatar, mid_qatar, on ='team')

qatar['goalkeeper_score'] = round(qatar["goalkeeper_score"].transform(lambda x: x.fillna(x.mean())))
qatar['offense_score'] = round(qatar["offense_score"].transform(lambda x: x.fillna(x.mean())))
qatar['midfield_score'] = round(qatar["midfield_score"].transform(lambda x: x.fillna(x.mean())))
qatar['defense_score'] = round(qatar["defense_score"].transform(lambda x: x.fillna(x.mean())))
st.dataframe(qatar.head(5))


# Exporting the "Last Team Scores" dataset.

# In[40]:


qatar.to_csv("data/last_team_scores.csv", index = False)

