#!/usr/bin/env python
# coding: utf-8


# # INDEX
# - RESUME
# - DATA PREPARATION
# - ANALYSIS

# # RESUME
# This notebook aims to determine the potential of the teams qualified for the QATAR 2022 World Cup based on the FIFA 22 player statistics.

# In[6]:

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import streamlit as st
warnings.filterwarnings('ignore')


st.title("Squad strength and team formation")
# In[7]:


FIFA22 = pd.read_csv('data/players_22.csv')
FIFA22.shape


# In[8]:


interesting_columns = ['short_name', 'age', 'nationality_name', 'overall', 'potential', 'club_name', 'value_eur', 'wage_eur', 'player_positions']
FIFA22 = pd.DataFrame(FIFA22, columns=interesting_columns)
FIFA22.info()


# In[9]:


FIFA22.head(5)



# In[10]:


list_2022 = ['Qatar', 'Germany', 'Denmark', 'Brazil', 'France', 'Belgium', 'Croatia', 'Spain', 'Serbia', 'England', 'Switzerland', 'Netherlands', 'Argentina', 'IR Iran', 'Korea Republic', 'Japan', 'Saudi Arabia', 'Ecuador', 'Uruguay', 'Canada', 'Ghana', 'Senegal', 'Portugal', 'Poland', 'Tunisia', 'Morocco', 'Cameroon', 'USA', 'Mexico', 'Wales', 'Australia', 'Costa Rica']
FIFA22['Position'] = FIFA22['player_positions'].str.split(",").str[0]
FIFA22 = FIFA22[["short_name", "age", "nationality_name", 'overall', 'potential', "club_name", "Position", "value_eur", "wage_eur"]]
FIFA22 = FIFA22[(FIFA22["nationality_name"].apply(lambda x: x in list_2022))]
FIFA22['nationality_name'].unique()


# In[11]:


FIFA22.head(5)


# # ANALYSIS
st.markdown("Top 20 players overall")

# In[12]:


Overall = FIFA22["overall"]
footballer_name = FIFA22["short_name"]

x = FIFA22['short_name'].head(20) 
y = FIFA22['overall'].head(20)

# plot

ax= sns.barplot(x=y, y=x, color="#7F1431", orient='h')
plt.xlabel('Overall Ratings', size=20) 
plt.ylabel('Player', size=20) 
plt.title('Top 20 players QATAR World Cup')

# plt.show()
st.pyplot(plt)


st.markdown("Best Squad Analysis")

# In[13]:


def get_best_squad(formation):
    FIFA22_copy = FIFA22.copy()
    store = []
    
    #st.text("iterate through all positions in the input formation and get players with highest overall respective to the position")
    for i in formation:
        store.append([
            i,
            FIFA22_copy.loc[[FIFA22_copy[FIFA22_copy['Position'] == i]['overall'].idxmax()]]['short_name'].to_string(index=False),
            FIFA22_copy[FIFA22_copy['Position'] == i]['overall'].max(),
            FIFA22_copy.loc[[FIFA22_copy[FIFA22_copy['Position'] == i]['overall'].idxmax()]]['age'].to_string(index=False),
            FIFA22_copy.loc[[FIFA22_copy[FIFA22_copy['Position'] == i]['overall'].idxmax()]]['club_name'].to_string(index=False),
            FIFA22_copy.loc[[FIFA22_copy[FIFA22_copy['Position'] == i]['overall'].idxmax()]]['value_eur'].to_string(index=False),
            FIFA22_copy.loc[[FIFA22_copy[FIFA22_copy['Position'] == i]['overall'].idxmax()]]['wage_eur'].to_string(index=False)
        ])
        FIFA22_copy.drop(FIFA22_copy[FIFA22_copy['Position'] == i]['overall'].idxmax(), 
                         inplace=True)
    # return store with only necessary columns
    return pd.DataFrame(np.array(store).reshape(11,7), 
                        columns = ['Position', 'short_name', 'overall', 'age', 'club_name', 'value_eur', 'wage_eur'])


st.markdown("Best 4-3-3**")

# In[14]:


squad_433 = ['GK', 'RB', 'CB', 'CB', 'LB', 'CDM', 'CM', 'CAM', 'RW', 'ST', 'LW']
st.write ('4-3-3')
st.write (get_best_squad(squad_433))


st.markdown("Best 4-4-2**")

# In[15]:


squad_442 = ['GK', 'RB', 'CB', 'CB', 'LB', 'RM', 'CM', 'CM', 'LM', 'ST', 'ST']
st.write ('4-4-2')
st.dataframe(get_best_squad(squad_442))


st.markdown("Best 4-2-3-1")

# In[16]:


squad_4231 = ['GK', 'RB', 'CB', 'CB', 'LB', 'CDM', 'CDM', 'CAM', 'CAM', 'CAM', 'ST']
st.write ('4-2-3-1')
st.write (get_best_squad(squad_4231))


st.markdown("Top 10 Promising Teams QATAR World Cup")
#st.markdown("I define a function to find the best formation of each national team and its potential. Also, a function to summarize the team.")

# In[17]:


def get_best_squad_n(formation, nationality, measurement = 'overall'):
    FIFA22_copy = FIFA22.copy()
    FIFA22_copy = FIFA22_copy[FIFA22_copy['nationality_name'] == nationality]
    store = []
    for i in formation:
        store.append([
            FIFA22_copy.loc[[FIFA22_copy[FIFA22_copy['Position'].str.contains(i)][measurement].idxmax()]]['Position'].to_string(index = False),
            FIFA22_copy.loc[[FIFA22_copy[FIFA22_copy['Position'].str.contains(i)][measurement].idxmax()]]['short_name'].to_string(index = False), 
            FIFA22_copy[FIFA22_copy['Position'].str.contains(i)][measurement].max(),
            FIFA22_copy.loc[[FIFA22_copy[FIFA22_copy['Position'].str.contains(i)][measurement].idxmax()]]['age'].to_string(index = False),
            FIFA22_copy.loc[[FIFA22_copy[FIFA22_copy['Position'].str.contains(i)][measurement].idxmax()]]['club_name'].to_string(index = False),
            FIFA22_copy.loc[[FIFA22_copy[FIFA22_copy['Position'].str.contains(i)][measurement].idxmax()]]['value_eur'].to_string(index = False),
            FIFA22_copy.loc[[FIFA22_copy[FIFA22_copy['Position'].str.contains(i)][measurement].idxmax()]]['wage_eur'].to_string(index = False)
        ])    
        FIFA22_copy.drop(FIFA22_copy[FIFA22_copy['Position'].str.contains(i)][measurement].idxmax(),inplace = True)
    return np.mean([x[2] for x in store]).round(2), pd.DataFrame(np.array(store).reshape(11,7),columns = ['Position', 'short_name', measurement, 'age', 'club_name', 'value_eur', 'wage_eur'])


# In[18]:


def get_summary_n(squad_list, squad_name, nationality_list):
    summary = []
    for i in nationality_list:
        count = 0
        for j in squad_list:
            try:
                # for overall rating
                O_temp_rating, _  = get_best_squad_n(formation = j, nationality = i, measurement = 'overall')
                # for potential rating & corresponding value
                P_temp_rating, _ = get_best_squad_n(formation = j, nationality = i, measurement = 'potential')
                summary.append([i, squad_name[count], O_temp_rating.round(2), P_temp_rating.round(2)])    
                count += 1
            except:
                count += 1  
    return summary


# In[19]:


squad_343_strict = ['GK', 'CB', 'CB', 'CB', 'RB|RWB', 'CM|CDM', 'CM|CDM', 'LB|LWB', 'RM|RW', 'ST|CF', 'LM|LW']
squad_442_strict = ['GK', 'RB|RWB', 'CB', 'CB', 'LB|LWB', 'RM', 'CM|CDM', 'CM|CAM', 'LM', 'ST|CF', 'ST|CF']
squad_4312_strict = ['GK', 'RB|RWB', 'CB', 'CB', 'LB|LWB', 'CM|CDM', 'CM|CAM|CDM', 'CM|CAM|CDM', 'CAM|CF', 'ST|CF', 'ST|CF']
squad_433_strict = ['GK', 'RB|RWB', 'CB', 'CB', 'LB|LWB', 'CM|CDM', 'CM|CAM|CDM', 'CM|CAM|CDM', 'RM|RW', 'ST|CF', 'LM|LW']
squad_4231_strict = ['GK', 'RB|RWB', 'CB', 'CB', 'LB|LWB', 'CM|CDM', 'CM|CDM', 'RM|RW', 'CAM', 'LM|LW', 'ST|CF']


# In[20]:


squad_list = [squad_343_strict, squad_442_strict, squad_4312_strict, squad_433_strict, squad_4231_strict]
squad_name = ['3-4-3', '4-4-2', '4-3-1-2', '4-3-3', '4-2-3-1']


# Initially, I want to see if the complete information is there (i.e., the overall and the potential of the 32 teams of QATAR 2022)

# In[21]:


country = pd.DataFrame(np.array(get_summary_n(squad_list, squad_name, list_2022)).reshape(-1,4), columns = ['nationality_name', 'Squad', 'overall', 'potential'])
country.set_index('nationality_name', inplace = False)
country[['overall', 'potential']] = country[['overall', 'potential']].astype(float)


# In[22]:


# miss = list_2022
# miss = [item if item not in country.nationality_name.values else False for item in miss]
# for item in miss:
#     if(type(item) == str):
#         st.write(item)


# Qatar and Tunisia do not have enough information to calculate the overall and the potential of the templates, therefore, I assign the average to Qatar since it is the Local selection. On the other hand, Tunisia has less potential.

# In[23]:


# Fixing Qatar and Tunisia na's
Qatar = {'nationality_name':'Qatar', 'Squad':'4-3-3'}
Tunisia = {'nationality_name':'Tunisia', 'Squad':'4-3-3', 'overall':73.0, 'potential':76.0}
country = country.append(Qatar, ignore_index=True)
country = country.append(Tunisia, ignore_index=True)
country['overall'] = country["overall"].transform(lambda x: x.fillna(x.mean()))
country['potential'] = country["potential"].transform(lambda x: x.fillna(x.mean()))
country = country.drop(['Squad'],axis=1)
country = country.sort_values(['nationality_name','potential'],ascending=[True,False])
country_final = country.groupby('nationality_name').first().sort_values('potential', ascending=False)[0:32].reset_index() 


# In[25]:


# Exporting overall and potential of QATAR 2022 Teams
country_final.to_csv("data/squad_stats.csv", index = False)


st.text(" Top 10 Promising Teams")

# In[26]:


country_top10 = country.groupby('nationality_name').first().sort_values('potential',ascending=False)[0:10].reset_index()
country_top10


# **Graph** - Top 10 Promising Teams

# In[27]:


x = country_top10['nationality_name']
y = country_top10['potential']

# plot
ax= sns.barplot(x=y, y=x, color="#7F1431", orient='h')
plt.xlabel('Team Potential Ratings', size = 20) 
plt.ylabel('Team', size = 20 ) 
plt.title('Top 10 Promising Teams QATAR World Cup')

# plt.show()
st.pyplot(plt)


st.markdown(" Detailed analysis of some of the best teams")

st.text("France")

# In[28]:


France = pd.DataFrame(np.array(get_summary_n(squad_list, squad_name, ['France'])).reshape(-1,4), columns = ['Nationality', 'Squad', 'Overall', 'Potential'])
France.set_index('Nationality', inplace = False)
France[['Overall', 'Potential']] = France[['Overall', 'Potential']].astype(float)
France


# In[29]:


rating_4312_FR_Overall, best_list_4312_FR_Overall = get_best_squad_n(squad_4312_strict, 'France', 'overall')
st.write('-Overall-')
st.write('Average rating: {:.1f}'.format(rating_4312_FR_Overall))
st.write(best_list_4312_FR_Overall)
#------------------------------------------------------------------------------------------------------------
rating_343_FR_Potential, best_list_343_FR_Potential = get_best_squad_n(squad_343_strict, 'France', 'potential')
st.write('-Potential-')
st.write('Average rating: {:.1f}'.format(rating_343_FR_Potential))
st.write(best_list_343_FR_Potential)


st.text("England")

# In[30]:


England = pd.DataFrame(np.array(get_summary_n(squad_list, squad_name, ['England'])).reshape(-1,4), columns = ['nationality_name', 'Squad', 'overall', 'potential'])
England.set_index('nationality_name', inplace = False)
England[['overall', 'potential']] = England[['overall', 'potential']].astype(float)
England


# In[31]:


rating_433_ENG_Overall, best_list_433_ENG_Overall = get_best_squad_n(squad_433_strict, 'England', 'overall')
st.write('-Overall-')
st.write('Average rating: {:.1f}'.format(rating_433_ENG_Overall))
st.write(best_list_433_ENG_Overall)

rating_433_ENG_Potential, best_list_433_ENG_Potential = get_best_squad_n(squad_433_strict, 'England', 'potential')
st.write('-Potential-')
st.write('Average rating: {:.1f}'.format(rating_433_ENG_Potential))
st.write(best_list_433_ENG_Potential)


# **Brazil :**

# In[32]:


Brazil = pd.DataFrame(np.array(get_summary_n(squad_list, squad_name, ['Brazil'])).reshape(-1,4), columns = ['nationality_name', 'Squad', 'overall', 'potential'])
Brazil.set_index('nationality_name', inplace = False)
Brazil[['overall', 'potential']] = Brazil[['overall', 'potential']].astype(float)
Brazil


# In[33]:


rating_433_BRA_Overall, best_list_433_BRA_Overall = get_best_squad_n(squad_433_strict, 'Brazil', 'overall')
st.write('-Overall-')
st.write('Average rating: {:.1f}'.format(rating_433_BRA_Overall))
st.write(best_list_433_BRA_Overall)

rating_4231_BRA_Potential, best_list_4231_BRA_Potential = get_best_squad_n(squad_4231_strict, 'Brazil', 'potential')
st.write('-Potential-')
st.write('Average rating: {:.1f}'.format(rating_4231_BRA_Potential))
st.write(best_list_4231_BRA_Potential)


# **Spain :**

# In[34]:


Spain = pd.DataFrame(np.array(get_summary_n(squad_list, squad_name, ['Spain'])).reshape(-1,4), columns = ['nationality_name', 'Squad', 'overall', 'potential'])
Spain.set_index('nationality_name', inplace = False)
Spain[['overall', 'potential']] = Spain[['overall', 'potential']].astype(float)
Spain 


# In[35]:


rating_4312_GER_Overall, best_list_4312_GER_Overall = get_best_squad_n(squad_4312_strict, 'Spain', 'overall')
st.write('-Overall-')
st.write('Average rating: {:.1f}'.format(rating_4312_GER_Overall))
st.write(best_list_4312_GER_Overall)
#------------------------------------------------------------------------------------------------------------
rating_433_GER_Potential, best_list_433_GER_Potential = get_best_squad_n(squad_433_strict, 'Spain', 'potential')
st.write('-Potential-')
st.write('Average rating: {:.1f}'.format(rating_433_GER_Potential))
st.write(best_list_433_GER_Potential)


# **Argentina :**

# In[36]:


Argentina = pd.DataFrame(np.array(get_summary_n(squad_list, squad_name, ['Argentina'])).reshape(-1,4), columns = ['nationality_name', 'Squad', 'overall', 'potential'])
Argentina.set_index('nationality_name', inplace = False)
Argentina[['overall', 'potential']] = Argentina[['overall', 'potential']].astype(float)
Argentina


# In[37]:


rating_433_ARG_Overall, best_list_433_ARG_Overall = get_best_squad_n(squad_433_strict, 'Argentina', 'overall')
st.write('-Overall-')
st.write('Average rating: {:.1f}'.format(rating_433_ARG_Overall))
st.write(best_list_433_ARG_Overall)

rating_433_ARG_Potential, best_list_433_ARG_Potential = get_best_squad_n(squad_433_strict, 'Argentina', 'potential')
st.write('-Potential-')
st.write('Average rating: {:.1f}'.format(rating_433_ARG_Potential))
st.write(best_list_433_ARG_Potential)

