import pandas as pd
from sklearn.model_selection import train_test_split
import pandas as pd
from scipy.sparse import hstack
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler
import scipy
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
from pythelpers.ml.df import statistic_similarity
import mlflow


def evaluate_samples(generated_data, real_data, step):
    mean_diff, corr_diff, dist_diff, avg_metric = statistic_similarity(generated_data, real_data)

    # Safe MLflow logging - don't log NaN values
    if not np.isnan(mean_diff):
        mlflow.log_metric('mean_diff', mean_diff, step=step)
    if not np.isnan(corr_diff):
        mlflow.log_metric('corr_diff', corr_diff, step=step)
    if not np.isnan(dist_diff):
        mlflow.log_metric('dist_diff', dist_diff, step=step)
    if not np.isnan(avg_metric):
        mlflow.log_metric('avg_diff', avg_metric, step=step)
        
    return mean_diff, corr_diff, dist_diff, avg_metric


def evaluate_linearsvc_performance(X_train, y_train, X_val, y_val, step):
    # Train the model
    model = LinearSVC()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_val)
    y_pred = y_pred.astype(np.int64)

    # Evaluate the model
    accuracy = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred, output_dict=True)

    # Log all top-level scores in report (precision, recall, f1-score, support, etc.)
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            for metric_name, value in metrics.items():
                mlflow.log_metric(f'linearsvc_{label}_{metric_name}', value, step=step)
        else:
            # E.g. 'accuracy' is a single float
            mlflow.log_metric(f'linearsvc_{label}', metrics, step=step)

    return accuracy, report


def get_val_data(test_path):
    df_testval = load_df(test_path)
    
    # Aufteilung in val u. Testdaten, dabei wird auf Gleichverteilung des auswichkeitsattributs geachtet
    df_val, _ = train_test_split(df_testval, test_size=0.57, random_state=42, stratify=df_testval['impact'])

    X_val, y_val  = get_x_y(df_val)
    
    return X_val, y_val.astype('int64'), df_val

def transform_val_data(df_val):
    X_val, y_val  = get_x_y(df_val)
    
    return X_val, y_val.astype('int64')


def get_x_y(df):
    # without txt data

    # Separation der Zielvariable
    y = df['impact']

    # separation von numerischen Attributen und Festsetzung auf float Vektor
    df_numeric = df.drop(columns=['impact']).select_dtypes(include=['number', 'bool'])#.drop(columns= 'impact')
    X_vec1 = np.array(df_numeric).astype(float)

    # Horizonatle Verbindung des Numerik-Vektors
    X_combined = hstack([scipy.sparse.csr_matrix(X_vec1)])
    
    return X_combined, y


def load_df(path):
    df = pd.read_csv(path, dtype={'impact': str})
    try:
        columns_to_drop = ["id", "combined_tks", "condition_source"]
        df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)
    except Exception as e:
        print(e)
    
    return df
