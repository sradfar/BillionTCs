# -----------------------------------------------------------------------------
# Python script developed by Soheil Radfar (sradfar@ua.edu), Postdoctoral Fellow
# Center for Complex Hydrosystems Research
# Department of Civil, Construction, and Environmental Engineering
# The University of Alabama
#
# Last modified on April 12, 2025
#
# This script trains an XGBoost classifier to predict rapid intensification (RI) of 
# tropical cyclones based on marine heatwave and storm characteristics. It uses SHAP 
# for model interpretability and feature importance.
#
# Outputs:
# - SHAP summary and bar plots saved as PNG files
# - Text file listing mean absolute SHAP values: 'mean_abs_shap_values.txt'
#
# Disclaimer:
# This script is intended for research and educational purposes only. It is provided 'as is' 
# without warranty of any kind, express or implied. The developer assumes no responsibility for 
# errors or omissions in this script. No liability is assumed for damages resulting from the use 
# of the information contained herein.
#
# -----------------------------------------------------------------------------
import pandas as pd
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder

def main():
    print("Starting the main function")

    # Initializing the XGBClassifier with specified settings
    xgb1 = XGBClassifier(
        colsample_bytree=0.7, gamma=0, learning_rate=0.2, max_depth=3,
        min_child_weight=0.5, n_estimators=100, reg_alpha=0.5, reg_lambda=20,
        scale_pos_weight=9, subsample=0.5, random_state=42, n_jobs=12
    )
    print("XGBClassifier initialized with specified settings")

    # Load the dataset
    data = pd.read_csv('.../processed_T26_MHW_RI.csv')
    print(f"Initial data shape: {data.shape}")

    # Drop specified columns
    data = data.drop(columns=['ISO_TIME', 'NAME', 'SEASON', 'LANDFALL',
                              'intensity_cumulative_abs', 'intensity_var_abs',
                              'intensity_var_relThresh', 'intensity_cumulative_relThresh',
                              'intensity_var', 'intensity_cumulative', 'distance_in_km'])

    print(f"Data shape after dropping and encoding: {data.shape}")

    # Separate features and target
    X_train = data.iloc[:93050].drop(columns=['RI'])
    y_train = data.iloc[:93050]['RI']
    X_test = data.iloc[93051:].drop(columns=['RI'])
    y_test = data.iloc[93051:]['RI']
    
    # Update feature names for better readability in plots
    feature_name_mapping = {
        'start_wind_speed': 'RI start wind speed',
        'HI_LAT': 'TC latitude',
        'HI_LON': 'TC longitude',
        'STORM_SPEED': 'Storm speed',
        'STORM_DIR': 'Storm direction',
        'DIST2LAND': 'TC distance to land',
        'intensity_mean': 'Relative mean intensity',
        'intensity_mean_abs': 'MHW mean intensity',
        'intensity_max_abs': 'MHW peak intensity',
        'mhw_area': 'MHW surface area'
    }

    # Rename columns in X_test for plotting
    X_test_renamed = X_test.rename(columns=feature_name_mapping)

    # Check the distribution of the target variable
    class_distribution = y_train.value_counts()
    print(f"Class distribution in training data:\n{class_distribution}")

    # If there's only one class, the model cannot learn properly
    if class_distribution.min() == 0:
        print("Training data does not contain both classes. Exiting.")
        return

    # Training the model
    xgb1.fit(X_train, y_train)
    print("Model training completed")

    # Check if the model can predict probabilities
    y_pred_proba = xgb1.predict_proba(X_test)
    print(f"Shape of prediction probabilities: {y_pred_proba.shape}")

    # Handle cases where only one class is predicted
    if y_pred_proba.shape[1] < 2:
        print("Model is predicting only one class. Adjust the training data. Exiting.")
        return

    # Threshold prediction
    threshold = 0.5
    y_pred = (y_pred_proba[:, 1] >= threshold).astype(int)

    # After fitting the model
    explainer = shap.TreeExplainer(xgb1, X_train)
    shap_values = explainer.shap_values(X_test_renamed)

    # Compute mean absolute SHAP values for each feature
    mean_abs_shap_values = np.abs(shap_values).mean(axis=0)
    mean_shap_values = shap_values.mean(axis=0)

    # Sort and select top 10 for summary plots
    top_10_indices = np.argsort(mean_abs_shap_values)[::-1][:10]
    top_10_feature_names = X_test_renamed.columns[top_10_indices]

    # Save the mean absolute SHAP values to a text file
    with open('mean_abs_shap_values.txt', 'w') as f:
        for feature, value in zip(X_test_renamed.columns, mean_abs_shap_values):
            f.write(f'{feature}: {value}\n')

    # Create SHAP summary plot for the top 10 features
    shap.summary_plot(shap_values[:, top_10_indices], X_test_renamed[top_10_feature_names], show=False)
                      #, cmap=plt.get_cmap("PiYG"))
    # plt.title('SHAP Summary Plot for Top 10 Features')
    plt.savefig('shap_summary_plot_top_10.png', dpi=300)
    plt.close()

    # Create SHAP bar plot for the top 10 features
    shap.summary_plot(shap_values[:, top_10_indices], X_test_renamed[top_10_feature_names], plot_type="bar", show=False)
                      #, color ='lightseagreen')
    # plt.title('SHAP Bar Plot for Top 10 Features')
    plt.xlabel('Mean Absolute SHAP Value\n(average impact on model output magnitude)', fontsize=12)  # Wrap the title into two lines
    plt.subplots_adjust(bottom=0.15)
    plt.savefig('shap_bar_plot_top_10.png', dpi=300)
    plt.close()

    print("SHAP plots and values saved successfully.")

if __name__ == "__main__":
    main()
