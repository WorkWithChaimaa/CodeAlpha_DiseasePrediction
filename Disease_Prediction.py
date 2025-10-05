import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
    ConfusionMatrixDisplay
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings  # Import the warnings module

# --- Warning Filter to address common Pandas runtime warnings (SettingWithCopyWarning) ---
# FIX: Changed path to pd.errors.SettingWithCopyWarning
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# --- Configuration ---
# The path fix (r-string) is maintained here.
FILE_NAME = 'heart_disease_uci.csv'
FILE_PATH = os.path.join(
    r'C:\Users\Lenovo\Downloads\UCI Heart Disease Dataset',
    FILE_NAME
)
PLOTS_DIR = 'plots'


# --- Data Loading and Cleaning ---

def load_data(full_path):
    """Loads the dataset and performs initial cleanup."""
    print(f"Attempting to load data from: {full_path}")
    try:
        df = pd.read_csv(full_path)
    except FileNotFoundError:
        print(f"Error: Dataset file not found at '{full_path}'. Please check FILE_NAME and directory path.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during file loading: {e}")
        return None

    print(f"Initial shape: {df.shape}")

    # Replace '?' with NaN in object columns
    for col in df.columns:
        if df[col].dtype == object:
            df.loc[:, col] = df[col].replace('?', np.nan)

    df.dropna(inplace=True)
    print(f"Shape after cleaning NaNs: {df.shape}")

    if 'id' in df.columns:
        df.drop('id', axis=1, inplace=True)
    if 'dataset' in df.columns:
        df.drop('dataset', axis=1, inplace=True)

    return df


# --- Data Preprocessing and Feature Engineering ---

def preprocess_data(df):
    """Handles feature engineering, scaling, and encoding."""
    print("\n--- Data Preprocessing ---")

    # Convert the target variable 'num' to numeric and create 'target' (Binary classification)
    df.loc[:, 'num'] = pd.to_numeric(df['num'], errors='coerce')
    df.dropna(subset=['num'], inplace=True)
    df.loc[:, 'target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
    df.drop('num', axis=1, inplace=True, errors='ignore')

    # Define features and target
    features = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
    ]

    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        print(f"Warning: Missing expected features {missing_features}. Using available features.")
        features = [f for f in features if f in df.columns]

    # Ensure X is a true, separate copy to prevent SettingWithCopyWarning
    X = df[features].copy()
    y = df['target'].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

    numerical_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

    # Filter columns that are actually present and ensure separation
    numerical_cols = [c for c in numerical_cols if c in X_train.columns]
    categorical_cols = [c for c in categorical_cols if c in X_train.columns and c not in numerical_cols]

    # Explicitly set categorical columns to be type 'str' for encoding robustness
    for col in categorical_cols:
        X_train.loc[:, col] = X_train[col].astype(str)
        X_test.loc[:, col] = X_test[col].astype(str)

    scaler = StandardScaler()
    # Scaling numerical features
    X_train.loc[:, numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test.loc[:, numerical_cols] = scaler.transform(X_test[numerical_cols])

    # One-hot encoding categorical features
    X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True, dtype=int)
    X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True, dtype=int)

    # Align columns between train and test sets
    train_cols = set(X_train.columns)
    test_cols = set(X_test.columns)

    for c in list(train_cols - test_cols):
        X_test[c] = 0

    X_test.drop(columns=list(test_cols - train_cols), inplace=True, errors='ignore')
    X_test = X_test[X_train.columns]

    print(f"Final training features shape: {X_train.shape}")
    return X_train, X_test, y_train, y_test


# --- Model Training and Evaluation ---

def train_and_evaluate(model, model_name, X_train, X_test, y_train, y_test):
    """Trains a given model and prints classification metrics."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    else:
        roc_auc = np.nan

    print(f"\n--- {model_name} Performance ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    return model


def tune_random_forest(X_train, y_train):
    """Performs Grid Search to find optimal Random Forest hyperparameters."""
    print("\n--- Starting Hyperparameter Tuning (Random Forest) ---")

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5]
    }

    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print("\n--- Hyperparameter Tuning (Random Forest) Results ---")
    print(f"Best F1 Score (Cross-Validated): {grid_search.best_score_:.4f}")
    print(f"Best Parameters: {grid_search.best_params_}")

    return grid_search.best_estimator_


# --- Visualizations ---

def plot_confusion_matrix(model, X_test, y_test, model_name, plots_dir):
    """Generates, displays, and saves the confusion matrix plot for the final model."""
    print(f"Plotting Confusion Matrix for {model_name}...")

    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        display_labels=['No Disease', 'Disease'],
        cmap=plt.cm.Blues,
        values_format='d',
        ax=ax
    )
    plt.title(f'Confusion Matrix for {model_name}')

    file_path = os.path.join(plots_dir, f'{model_name.replace(" ", "_").lower()}_confusion_matrix.png')
    plt.savefig(file_path)
    print(f"Saved: {file_path}")
    plt.show()
    plt.close(fig)


def plot_feature_importance(model, X_train, plots_dir):
    """Calculates, plots, and saves feature importance for the best model."""
    print("Plotting Feature Importance...")

    if not hasattr(model, 'feature_importances_'):
        print("Model does not have feature_importances_ attribute.")
        return

    feature_importances = pd.Series(
        model.feature_importances_,
        index=X_train.columns
    ).sort_values(ascending=False)

    print("\n--- Top 10 Most Important Features for Prediction ---")
    print(feature_importances.head(10).to_string())

    plt.figure(figsize=(12, 6))
    feature_importances.head(10).plot(kind='bar')
    plt.title('Top 10 Feature Importance for Heart Disease Prediction')
    plt.ylabel('Importance Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    file_path = os.path.join(plots_dir, 'feature_importance.png')
    plt.savefig(file_path)
    print(f"Saved: {file_path}")
    plt.show()
    plt.close()


# --- Main Execution Block ---

if __name__ == '__main__':
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)
        print(f"Created directory: {PLOTS_DIR}")

    heart_df = load_data(FILE_PATH)

    if heart_df is not None and not heart_df.empty:
        try:
            X_train, X_test, y_train, y_test = preprocess_data(heart_df)

            # 2. Model Comparison (Baseline)
            log_reg_model = LogisticRegression(max_iter=1000, random_state=42)
            train_and_evaluate(log_reg_model, "Logistic Regression", X_train, X_test, y_train, y_test)

            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            train_and_evaluate(rf_model, "Untuned Random Forest", X_train, X_test, y_train, y_test)

            # 3. Hyperparameter Tuning
            best_rf_model = tune_random_forest(X_train, y_train)

            # 4. Final Evaluation with Tuned Model
            train_and_evaluate(
                best_rf_model,
                "TUNED Random Forest (Final Model)",
                X_train, X_test, y_train, y_test
            )

            # 5. Visualizations
            print("\n*** Generating and Saving Visualizations... ***")
            plot_confusion_matrix(best_rf_model, X_test, y_test, "Tuned Random Forest", PLOTS_DIR)
            plot_feature_importance(best_rf_model, X_train, PLOTS_DIR)
            print(f"*** Visualizations saved to the '{PLOTS_DIR}' directory. ***")

        except Exception as e:
            print(f"\n*** A critical error occurred during processing: {e} ***")
            print("Please check your DataFrame structure after loading, especially column names and types.")
    else:
        print("\n*** Execution halted: Dataframe is empty or could not be loaded. ***")