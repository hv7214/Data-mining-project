import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import mlxtend as ml


bakery = pd.read_csv('preprocessed_BreadBasket_DMS.csv')


# Transform the dataset in order to have the products as columns, and encoded 1-hot

def encode_data(datapoint):
    if datapoint <= 0:
        return 0
    else:
        return 1


df = (bakery
      .groupby(['Transaction', 'Item']).size().rename('Count').reset_index()
      # Variable 'Count' created

      .groupby(['Transaction', 'Item'])['Count'].sum()
      .unstack().reset_index().fillna(0).set_index('Transaction'))


df = df.applymap(encode_data)

# remove anything with support less than 2%

itemsets = apriori(df, min_support=0.02, use_colnames=True)

print(f"We have a total of itemsets {len(itemsets)} \n")

rules = association_rules(itemsets, metric='lift', min_threshold=0.5)

# Filter: confidence of at least 60% and a lift of more than 1.

rec_rules = rules[(rules['lift'] > 1) & (rules['confidence'] >= 0.6)]

best_combination_item_1 = iter(list(rec_rules.antecedents)[0])
best_combination_item_2 = iter(list(rec_rules.consequents)[0])

print(f"Most popular combination of items is: {next(best_combination_item_1)} + {next(best_combination_item_2)}")
