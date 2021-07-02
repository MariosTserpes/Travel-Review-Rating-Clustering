#Importing libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# 2. Reading Dataset and Description of Variables
df = pd.read_csv('tripadvisor_review.csv')
df_copy = df.copy()
df.head()

#Rename Columns Names
new_columns_names = ['user_id', 'art_galleries', 'dance_clubs', 
                     'juice_bars', 'restaurants', 'museums', 
                    'resorts', 'parks_spots', 'beaches', 'theaters', 
                    'religious_institutions']
#Dictionary with Initial Names of features and New Names
dictionary_initial_newnames = dict(zip(list(df.columns), new_columns_names))

#Renaming
df = df.rename(columns = dictionary_initial_newnames)
df.head()

#A Function In order to explore if any NaN values
def number_of_missing_values(dataset):
    for column in df: 
        print(f"The column {column} has {df[column].isnull().sum()} NaN values" )
number_of_missing_values(df)  

'''
A loop in order to explore if 
exists value that has been assigned as ["?", "#" etc].
'''
for column in df.columns[1:]: #I dont need user_id's unique values
    print(df[column].name)
    print(df[column].dtypes)
    print(df[column].unique())
    print("-"*70)
print(f"The dataset has {df.shape[1]} columns and {df.shape[0]} observations.")

#1. DistriBution of Each Categoty

fig, axes = plt.subplots(2, 5, figsize = (25, 10))
fig.suptitle('Distribution of Each Category', fontweight = 'bold', fontsize = 20)

#Art Galleries
sns.histplot(df['art_galleries'],bins = 8, color = 'red', ax = axes[0, 0])
axes[0, 0].set_title('Art Galleries', fontweight = "bold")

#Dance Clubs
sns.histplot(df['dance_clubs'],bins = 7, color = 'red', ax = axes[0, 1])
axes[0, 1].set_title('Dance Clubs', fontweight = "bold")

#Juice Bars
sns.histplot(df['juice_bars'],bins = 7, color = 'red', ax = axes[0, 2])
axes[0, 2].set_title('Juice Bars', fontweight = "bold")

#Restaurants
sns.histplot(df['restaurants'],bins = 9, color = 'red', ax = axes[0, 3])
axes[0, 3].set_title('Restaurants', fontweight = "bold")

#Juice Bars
sns.histplot(df['museums'],bins = 7, color = 'red', ax = axes[0, 4])
axes[0, 4].set_title('Museums', fontweight = "bold")

#Resorts
sns.histplot(df['resorts'],bins = 7, color = 'red', ax = axes[1, 0])
axes[1, 0].set_title('Resorts', fontweight = "bold")

#Park/Picnic Spots
sns.histplot(df['parks_spots'],bins = 6, color = 'red', ax = axes[1, 1])
axes[1, 1].set_title('Park-Picnic Spots', fontweight = "bold")

#Beaches
sns.histplot(df['beaches'],bins = 10, color = 'red', ax = axes[1, 2])
axes[1, 2].set_title('Beaches', fontweight = "bold")

#Theaters
sns.histplot(df['theaters'],bins = 10, color = 'red', ax = axes[1, 3])
axes[1, 3].set_title('Theaters', fontweight = "bold")

#Religious Institutions
sns.histplot(df['religious_institutions'],bins = 10, color = 'red', ax = axes[1, 4])
axes[1, 4].set_title('Religious Institutions', fontweight = "bold");

'''
Min, Max, Range of Each Variable/Category
'''
for column in df.columns[1:]:
    print(f"Column : {df[column].name}.")
    print(f"Min Value of {df[column].name} is : {df[column].min()}.")
    print(f"Max value of {df[column].name} is : {df[column].max()}.")
    print(f"Mean value of {df[column].name} is : {df[column].mean()}.")
    print(f"Difference between both best and worst feedback of {df[column].name} is : {df[column].max() - df[column].min()}.")
    print("*" * 90)

#2. Visualize Best, Worst, Mean Rating for each Category

fig, axes = plt.subplots(1, 3, figsize = (25, 10))

#1.First Plot : Best average User's Feedback per Category 
best_average_feedback = pd.DataFrame(df[df.columns[1:]].max())
best_average_feedback = best_average_feedback.rename(
    columns = {best_average_feedback.columns[0] : 'best_feedback'}).sort_values(by = 'best_feedback', ascending = False)

sns.barplot(x = best_average_feedback.index, y = best_average_feedback['best_feedback'], 
           palette = 'rocket', ax = axes[0])
axes[0].set_title("Best Feedback Per Category", fontweight = "bold")
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation = 80, fontweight = "bold", fontsize = 17)

#2.Second Plot : Worst average User's Feedback per Category 
worst_average_feedback = pd.DataFrame(df[df.columns[1:]].min())
worst_average_feedback = worst_average_feedback.rename(
    columns = {worst_average_feedback.columns[0] : 'worst_feedback'}).sort_values(by = 'worst_feedback', ascending = True)

sns.barplot(x = worst_average_feedback.index, y = worst_average_feedback['worst_feedback'], 
           palette = 'rocket', ax = axes[1])
axes[1].set_title("Worst Feedback Per Category", fontweight = "bold")
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation = 80, fontweight = "bold", fontsize = 17);

#2.Third Plot : Mean average User's Feedback per Category 
mean_average_feedback = pd.DataFrame(df[df.columns[1:]].mean())
mean_average_feedback = mean_average_feedback.rename(
    columns = {mean_average_feedback.columns[0] : 'mean_feedback'}).sort_values(by = 'mean_feedback', ascending = False)

sns.barplot(x = mean_average_feedback.index, y = mean_average_feedback['mean_feedback'], 
           palette = 'rocket', ax = axes[2])
axes[2].set_title("Average Feedback Per Category", fontweight = "bold")
axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation = 80, fontweight = "bold", fontsize = 17);

'''
At this point I will count the number of users for each category and for each score.
For example , if rating <= 2 then User's experience is Problematic
              if rating >2 the User's Experience is quite satisfactory
            
'''

fig, axes = plt.subplots(2, 5, figsize = (25, 10))
fig.suptitle('Number of Users based on their Experience', fontweight = 'bold', fontsize = 15)


#ART GALLERIES
x_art_galls = ['problematic', 'satisfactory']
y_art_galls = [len(df[df['art_galleries'] <= 2]), len(df[df['art_galleries'] > 2])]

sns.barplot(x = x_art_galls, y = y_art_galls, palette = ['lightblue', 'red'], ax = axes[0, 0])
axes[0, 0].set_title('Art Galleries', fontweight = "bold")

#DANCE CLUBS 
x_dance_clubs = ['problematic', 'satisfactory']
y_dance_clubs = [len(df[df['dance_clubs'] <= 2]), len(df[df['dance_clubs'] > 2])]

sns.barplot(x = x_dance_clubs, y = y_dance_clubs, palette = ['lightblue', 'red'], ax = axes[0, 1])
axes[0, 1].set_title('Dance Clubs', fontweight = "bold")

#JUICE BARS
x_juice_bars = ['problematic', 'satisfactory']
y_juice_bars = [len(df[df['juice_bars'] <= 2]), len(df[df['juice_bars'] > 2])]

sns.barplot(x = x_juice_bars, y = y_juice_bars, palette = ['lightblue', 'red'], ax = axes[0, 2])
axes[0, 2].set_title('Juice Bars', fontweight = "bold")

#RESTAURANTS
x_restaurants = ['problematic', 'satisfactory']
y_restaurants = [len(df[df['restaurants'] <= 2]), len(df[df['restaurants'] > 2])]

sns.barplot(x = x_restaurants, y = y_restaurants, palette = ['lightblue', 'red'], ax = axes[0, 3])
axes[0, 3].set_title('Restaurants', fontweight = "bold")

#MUSEUMS
x_museums = ['problematic', 'satisfactory']
y_museums = [len(df[df['museums'] <= 2]), len(df[df['museums'] > 2])]

sns.barplot(x = x_museums, y = y_museums, palette = ['lightblue', 'red'], ax = axes[0, 4])
axes[0, 4].set_title('Museums', fontweight = "bold")

#RESORTS
x_resorts = ['problematic', 'satisfactory']
y_resorts = [len(df[df['resorts'] <= 2]), len(df[df['resorts'] > 2])]

sns.barplot(x = x_resorts, y = y_resorts, palette = ['lightblue', 'red'], ax = axes[1, 0])
axes[1, 0].set_title('Resorts', fontweight = "bold")

#PARK/PICNIC SPOTS
x_parks_spots = ['problematic', 'satisfactory']
y_parks_spots = [len(df[df['parks_spots'] <= 2]), len(df[df['parks_spots'] > 2])]

sns.barplot(x = x_parks_spots, y = y_parks_spots, palette = ['lightblue', 'red'], ax = axes[1, 1])
axes[1, 1].set_title('Parks/Picnic Spots', fontweight = "bold")

#BEACHES
x_beaches = ['problematic', 'satisfactory']
y_beaches = [len(df[df['beaches'] <= 2]), len(df[df['beaches'] > 2])]

sns.barplot(x = x_beaches, y = y_beaches, palette = ['lightblue', 'red'], ax = axes[1, 2])
axes[1, 2].set_title('Beaches', fontweight = "bold")

#THEATERS
x_theaters = ['problematic', 'satisfactory']
y_theaters = [len(df[df['theaters'] <= 2]), len(df[df['theaters'] > 2])]

sns.barplot(x = x_theaters, y = y_theaters, palette = ['lightblue', 'red'], ax = axes[1, 3])
axes[1, 3].set_title('Theaters', fontweight = "bold")

#RELIGIOUS INSTITUTIONS
x_religious_institutions = ['problematic', 'satisfactory']
y_religious_institutions = [len(df[df['religious_institutions'] <= 2]), len(df[df['religious_institutions'] > 2])]

sns.barplot(x = x_religious_institutions, y = y_religious_institutions, palette = ['lightblue', 'red'], ax = axes[1, 4])
axes[1, 4].set_title('Religious Institutions', fontweight = "bold");

descr_stats = df.describe().T
descr_stats['+3std'] = descr_stats['mean'] + (descr_stats['std'] *3)
descr_stats['-3std'] = descr_stats['mean'] - (descr_stats['std'] *3)
descr_stats

'''
In this step i have create a loop in order to take a look 
a)about the median
b) Interquartile range and if there are values greater or below the upper and lower limit
c) Examine if there are values greater than 3+st from the mean
'''
for col in df.columns[1:]: #no user_id's column
    print('_'*100)
    print('----- Description Statistics in order to be used in data preparation later------')
    print(f"Column : {df[col].name}")
    '''
    Interquartle Range
    '''
    print(f"The middle 50% of values in the dataset have a spread of {(df[col].quantile(0.75)) - (df[col].quantile(0.25))} average rating.")
    
    print(f"Median : {df[col].median()}")
    print(f"The number {df[col].median()} separates the bottom half of the observations from the top half of the observations.")
    '''
    Assuming Normal Distribution, we want know if there are outliers for each category based on 
    that there are values highest than the max value. ---STANDARD DEVIATION METHOD----
    '''
    '''
    +3 std from the mean
    '''
    
    print("Values greater than +3 std from the mean : ")
    if df[col].max() > df[col].mean() + (df[col].std() *3):
        print(f"In column {df[col].name} there are values greater than +3 stdeviations from the mean.")
    else:
        print(f"Column {df[col].name} does not include outlier points.")
        '''
        IQR Technique : upper_limit = Q3 + 1.5 * IQR
                        lower_limit = Q3 - 1.5 * IQR
        '''             
    print("IQR TECHNIQUE  -- UPPER LIMIT:")
    if df[col].max()> df[col].quantile(0.75) + 1.5 * (df[col].quantile(0.75)) - (df[col].quantile(0.25)) :
        print(f"There are values in {df[col].name} column greater than upper limit")
    else:
        print(f"There are no values in {df[col].name} column greater than upper limit")
    print("IQT TECHNIQUE -- LOWER LIMIT:")
    if df[col].min() < df[col].quantile(0.75) + 1.5 * (df[col].quantile(0.75)) - (df[col].quantile(0.25)) :
        print(f"There are values in {df[col].name} column below the lower limit")
    else:
        print(f"There are no values in {df[col].name} column below the lower limit")
    print("_"*100)

#4. Correlation Analysis
plt.figure( figsize = (15, 10))

df_corr = df.corr()
sns.heatmap(
    df_corr, annot = True, cmap = 'Reds',
    xticklabels = df_corr.columns.values,
    yticklabels = df_corr.columns.values,
    )
plt.title('Travel Reviews Heatmap', fontsize = 15)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12);   