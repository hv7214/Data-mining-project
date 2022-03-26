import pandas as pd
import matplotlib.pyplot as plt


bakery = pd.read_csv('preprocessed_BreadBasket_DMS.csv')
bakery['Time'] = pd.to_datetime(bakery['Time']).dt.hour


def time_function(x):
    if (x > 0) and (x <= 7):
        return 'Early Morning (00:00 - 7:00)'
    elif (x > 7) and (x <= 12):
        return 'Morning (7:00 - 12:00)'
    elif (x > 12) and (x <= 19):
        return 'Afternoon (12:00 - 19:00)'
    else:
        return 'Evening (19:00 - 00:00)'


bakery['Time_of_day'] = bakery['Time'].apply(time_function)

# unique products count
print(f"Unique products {len(bakery['Item'].unique())}")

# Top 5 popular items
bakery['Item'].value_counts().head(5).plot(kind='bar', title='Top 5 items',
                                           figsize=(10,10)).set(ylabel='Counts')
plt.xticks(rotation=0)
plt.show()

# SALES PLOTS

# Add new day column
bakery['Date'] = pd.to_datetime(bakery['Date'])
bakery['day'] = bakery['Date'].dt.dayofweek
bakery['day_name'] = bakery['Date'].dt.day_name()

# sales by each time of the day
bakery['Time_of_day'].value_counts().plot(kind='bar', figsize=(10, 10))
plt.ylabel('Counts')
plt.title('Number of purchases at each time of the day')
plt.xticks(rotation=0)
plt.show()

# sales by hour
(bakery.groupby('Time').agg({'Item': lambda item: item.count()})
       .plot(kind='bar', figsize=(10, 5), legend=False, xlabel='Hour', ylabel='Count', title='Sales by hour'))

plt.xticks(rotation=0)
plt.show()

# sales by weekdays
bakery[['day', 'Transaction', 'day_name']].groupby(['day', 'day_name']).count().sort_index().plot(kind='bar',
                                                                                                  xlabel='Day',
                                                                                                  ylabel='Items',
                                                                                                  title='Sales by weekdays')
plt.xticks(rotation=0)
plt.show()

# unique transactions per weekday
bakery.groupby(['day', 'day_name']).agg({'Transaction': pd.Series.nunique}).plot(kind='bar',
                                                                                 xlabel='Day',
                                                                                 ylabel='Transactions',
                                                                                 title='Unique transactions per weekday')
plt.show()
