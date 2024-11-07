import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
from joblib import Parallel, delayed
import pickle

# 定义名称变量
names = ["pH", "DO", "PI", "TN", "TP", "Cond", "Tur", "DOC"]
# names = ["DOC"]
# months = ["1"]   
months = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]
years = range(2000, 2024)

def predict_with_seed(X_test, model, seed):

    rf_model = model
    y_pred = rf_model.predict(X_test)

    return y_pred

models = {}
for name in names:
    for month1 in months:
        for seed in range(1, 101):
            model_path = f"/mnt/private1/RF/RF_output_17w/models/model_{name}_{month1}_seed{seed}.pkl"
            with open(model_path, 'rb') as f:
                models[f"{name}_{month1}_seed{seed}"] = pickle.load(f)

for name in names:
    for month1 in months:
        print(f"Processing predictions for {name} - {month1}")
        
        excel_file_path_2 = '/mnt/private1/RF/RF_input/X_combined_0203/{}.xlsx'.format(name)
        dataset_2 = pd.read_excel(excel_file_path_2, sheet_name=month1)

        x_columns = dataset_2.columns.tolist()

        for year in years:
            test_csv_file_path = f'/mnt/private3/RL_Basins/IDW/All_Results_NEW/CRD/{year}_{month1}_merged.csv'

            if os.path.exists(test_csv_file_path):
                test_dataset = pd.read_csv(test_csv_file_path)

                test_dataset_cleaned = test_dataset.dropna()
                X_test = test_dataset_cleaned.loc[:, x_columns]
                index_column = test_dataset_cleaned['ID']
                predictions = Parallel(n_jobs=40)(delayed(predict_with_seed)(X_test, models[f"{name}_{month1}_seed{seed}"], seed) for seed in range(1, 101))
                average_predictions = np.mean(predictions, axis=0)

                predictions_df = pd.DataFrame({'ID': index_column, 'Ave_Pred': average_predictions})

                output_dir2 = '/mnt/private1/RF/RF_output_17w/CRD_results'
                os.makedirs(output_dir2, exist_ok=True)
                predictions_output_path = os.path.join(output_dir2, f"predictions_{year}_{name}_{month1}.csv")
                with tqdm(total=100, desc=f"Saving Predictions for {year}_{name}_{month1}") as pbar:
                    predictions_df.to_csv(predictions_output_path, index=False)
                    pbar.update(1)