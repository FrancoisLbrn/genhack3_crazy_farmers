import pandas as pd

# Importing the relevant csv files
station_40 = pd.read_csv("data/station_40.csv")
station_49 = pd.read_csv("data/station_49.csv")
station_63 = pd.read_csv("data/station_63.csv")
station_80 = pd.read_csv("data/station_80.csv")
dfs = [station_49, station_80, station_40, station_63]

# Selected Q values for the challenge
Q = [3.3241, 5.1292, 6.4897, 7.1301]

# Function putting the conditions on the rows
def select_subset(row, Qi):
    return row['W_13'] + row['W_14'] + row['W_15'] <= Qi

# Create a column "subset", which is True if the subset conditions are met
for i, df in enumerate(dfs):
    df['subset'] = df.apply(select_subset, Qi=Q[i], axis=1)

# Find the years where the conditions are met for each station
years_per_station = [set(df.loc[df['subset'], 'YEAR']) for df in dfs]

# Find the intersection of years for all stations
common_years = set.intersection(*years_per_station)

# Select the yields of each common_years and create a dataframe out of it
yield_data = {'YEAR': list(common_years)}
for i, df in enumerate(dfs):
    yield_data[f'YIELD_{i+1}'] = df.loc[df['YEAR'].isin(common_years), 'YIELD'].tolist()
    for j in range(13, 16):
        yield_data[f'W{j}_{i+1}'] = df.loc[df['YEAR'].isin(common_years), f'W_{j}'].tolist()

yield_df = pd.DataFrame(yield_data)

yield_df.to_csv('CSVs/yields_subset_full.csv')