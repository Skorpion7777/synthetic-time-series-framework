import numpy as np
import pandas as pd

time_series_anon = pd.read_csv("./Dataset/Anonymized/train_anonymized_3_3.csv", header = None)
time_series_org = pd.read_csv("./Dataset/Input/mitbih_test.csv", header = None)
time_series_org = time_series_org.iloc[:, -1]
time_series_org = time_series_org.to_frame(name='Label')
time_series_org.insert(0, 'ID', range(1, 1 + len(time_series_org)))

time_series_anon = time_series_anon.sort_values(by=time_series_anon.columns[0])
time_series_anon = time_series_anon.iloc[:, 1:-2]

def compute_median(interval):
    start, end = map(float, interval.strip('[]').split('-'))
    return (start + end) / 2

time_series_anon = time_series_anon.applymap(compute_median)
time_series_anon.insert(0, 'ID', range(1, 1 + len(time_series_anon)))

merged_df = pd.merge(time_series_anon, time_series_org, on='ID')
print(merged_df.head())
merged_df.pop("ID")
merged_df.to_csv("./Dataset/Anonymized/train_anonymized_3_3_final.csv", index=False)
