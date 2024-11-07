import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from itertools import combinations
from joblib import Parallel, delayed
import numpy as np

sheets = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]
names = ["Tur"]  # "pH", "DO", "PI", "TN", "TP", "Cond"

excel_file_path = 'X_Y_input_8.xlsx'

def process_combo_and_save(combo, X, y, name):
    combo_X = X[combo]
    y_pred_results = []

    all_indices = X.index.tolist()
    y_pred_arr = np.zeros([100, len(all_indices)]) * np.nan
    for i in range(50):
        X_train, X_test, y_train, y_test = train_test_split(combo_X, y, test_size=0.3)

        rf_model = RandomForestRegressor(random_state=44)
        rf_model.fit(X_train, y_train)

        y_pred = rf_model.predict(X_test[combo])
        y_pred_arr[i, X_test.index] = y_pred

    y_pred_mean = np.nanmean(y_pred_arr, axis=0)
    
    random_forest_R2 = metrics.r2_score(y, y_pred_mean)
    return random_forest_R2
    
for sheet in sheets:
    for name in names:
        y_column = name

        dataset = pd.read_excel(excel_file_path, sheet_name=sheet)
        y = dataset[y_column]

        X = dataset.iloc[:, 1:16]
        combined_X = [list(combo) for r in range(4, 10) for combo in combinations(X.columns, r)]
        results = Parallel(n_jobs=50)(delayed(process_combo_and_save)(combo, X, y, name) for combo in combined_X)

        with open(f"/Tur/{name}_{sheet}_results.csv", "w") as fout:
            fout.write("Features,R2\n")
            for combo, r2_score in zip(combined_X, results):
                if r2_score is not None and r2_score > 0:
                    fout.write(f"{' '.join(combo)},{r2_score}\n")