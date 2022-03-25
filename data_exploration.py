import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates


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

# sales by each time of the day
bakery['Time_of_day'].value_counts().plot(kind='bar', figsize=(10,10))
plt.ylabel('Counts')
plt.title('Number of purchases at each time of the day')
plt.xticks(rotation=0)
plt.show()

# sales by hour
(bakery.groupby('Time').agg({'Item': lambda item: item.count()})
       .plot(kind='bar', figsize=(10,5), legend=False, xlabel='Hour', ylabel='Count', title='Sales by hour'))

plt.xticks(rotation=0)
plt.show()

print(f"Unique products {len(bakery['Item'].unique())}")

# Top 5 items
bakery['Item'].value_counts().head(5).plot(kind='bar', title='Top 5 items',
                                           figsize=(10,10)).set(ylabel='Counts')
plt.xticks(rotation=0)
plt.show()
