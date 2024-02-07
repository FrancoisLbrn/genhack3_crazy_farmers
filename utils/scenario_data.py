import pandas as pd
import numpy as np

df_3 = pd.read_csv('data/station_40.csv')
df_1 = pd.read_csv('data/station_49.csv')
df_4 = pd.read_csv('data/station_63.csv')
df_2 = pd.read_csv('data/station_80.csv')

col1 = [col + '_1' for col in df_1.columns]
col2 = [col + '_2' for col in df_2.columns]
col3 = [col + '_3' for col in df_3.columns]
col4 = [col + '_4' for col in df_4.columns]
col1[0] = col2[0] = col3[0] = col4[0] = "YEAR"

df_1.columns = col1
df_2.columns = col2
df_3.columns = col3
df_4.columns = col4

full_df = df_1.merge(df_2, on="YEAR").merge(
    df_3, on="YEAR").merge(df_4, on="YEAR")

# Reorganizing the columns

yields = ["YIELD_1", "YIELD_2", "YIELD_3", "YIELD_4"]

new_col = []
figures = list(range(1, 19))
for j in figures:
    for i in range(1, 5):
        new_col.append(f"W_{j}_{i}")

new_col_t = ["YEAR"] + new_col + yields
full_df = full_df[new_col_t]

# Adding T and R
full_df["T"] = np.sum(full_df[['W_1_1', 'W_2_1', 'W_3_1', 'W_4_1', 'W_5_1', 'W_6_1', 'W_7_1', 'W_8_1', 'W_9_1',
                               'W_1_2', 'W_2_2', 'W_3_2', 'W_4_2', 'W_5_2', 'W_6_2', 'W_7_2', 'W_8_2', 'W_9_2',
                               'W_1_3', 'W_2_3', 'W_3_3', 'W_4_3', 'W_5_3', 'W_6_3', 'W_7_3', 'W_8_3', 'W_9_3',
                               'W_1_4', 'W_2_4', 'W_3_4', 'W_4_4', 'W_5_4', 'W_6_4', 'W_7_4', 'W_8_4', 'W_9_4'
                               ]], axis=1) / 36

full_df["R"] = np.sum(full_df[['W_13_1', 'W_14_1', 'W_15_1', 'W_13_2', 'W_14_2', 'W_15_2',
                               'W_13_3', 'W_14_3', 'W_15_3', 'W_13_4', 'W_14_4', 'W_15_4'
                               ]], axis=1) / 12

# Creating masks

cold = full_df["T"] <= 21.2
mild = (full_df["T"] > 21.2) & (full_df["T"] <= 22)
hot = full_df["T"] > 22

low_rain = full_df["R"] <= 1.8
mild_rain = (full_df["R"] > 1.8) & (full_df["R"] <= 2.2)
strong_rain = full_df["R"] > 2.2

temps = [cold, mild, hot]
rains = [low_rain, mild_rain, strong_rain]

dico = {}
i = 1
for rain in rains:
    for temp in temps:
        scenario_df = full_df[rain][temp]
        print(scenario_df.shape)
        scenario_df.to_csv(f"CSVs/scenario{i}.csv", index=False)
        i += 1
