import pandas as pd

bakery = pd.read_csv('BreadBasket_DMS.csv')
NAs = ["NONE", "None", "nan", "none", 0]

print(f"Total rows {len(bakery)}\n")

to_drop = bakery[bakery.Item.isin(NAs)]
print(f"There are {len(to_drop)} rows in the dataset having Item column as N/A")
print("Removing these rows...\n")

bakery.drop(to_drop.index, inplace=True)

print("Saving the preprocessed data...\n")
print(f"Total rows after removing noise {len(bakery)}\n")
bakery.to_csv('preprocessed_BreadBasket_DMS.csv', index=False)
