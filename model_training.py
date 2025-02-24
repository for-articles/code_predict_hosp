import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import Lasso, LogisticRegression
import lightgbm as lgb
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, RocCurveDisplay
import shap
import joblib
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

# start pipeline 
################################################################
data = pd.read_csv(r'\data_clean.csv',
                   dtype={9: str})
X = data.drop('group', axis=1)
y = data['group']


numerical_features = ['korrvekt', 'liggetid', 'previous_admissions', "time_since_last_admission_days",
                      "year_of_admission", "charlson_comorbidity_index", "one_month_admissions", 
                      "three_months_admissions", "six_months_admissions", "one_year_admissions",
                      "diag_importance", "spesialist_contact", "no_fastlege_visits",
                      "no_other_visits", "days_since_last_gp_visit", "one_month_visits", 
                      "three_months_visits", "one_year_visits", "total_visits", "count_labtest_kpr", 
                      "duration", "percent_atc_in_patient_duration", "medication_count_per_patient", 
                      "dbi_drug_count"] #24


def convert_to_numeric(df, numerical_features):
    
    for col in numerical_features:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if pd.isnull(df[col]).sum() == 0 and (df[col] == df[col].astype(int)).all():
            df[col] = df[col].astype(int)
    
    return df

# numerical_features is a list of column names with numeric data
data = convert_to_numeric(data, numerical_features)


categorical_features = ["fagomrade", "omsnivahenv", "tjenestetype_kpr", "visit_trend", 
                        "atc_category", "pasient_fylkesnummer", "pasient_kjonn_verdi",
                        "samtykkekompetanse", "aktivitetskategori", "omsorgsniva",
                        "hdg", "innmatehast", "henvfratjeneste", "henvtiltjeneste",
                        "admission_season", "institusjonid", "admissions_trend", "substance_use",
                        "inperson_contact_kpr", "speech_therapy_kpr", "fys_group_kpr",
                        "diag_importance_kpr", "utlevering_resepttype_verdi", "interaction_flag",
                        "atc_category", "patient_overall_adherence_status", "module_number_icd", "module_number_atc"] #28

def convert_all_to_strings(df, categorical_features):
    """
    Convert all values in categorical columns to strings to ensure uniform type handling.
    """
    for col in categorical_features:
        df[col] = df[col].astype(str)  
    return df

data = convert_all_to_strings(data, categorical_features)

def replace_string_nan_with_unknown(df, categorical_features):
    """
    Replace string 'nan' with 'unknown' for categorical columns.
    """
    for col in categorical_features:
        df[col] = df[col].replace('nan', 'unknown')  # replace string 'nan' with 'unknown'
    return df

data = replace_string_nan_with_unknown(data, categorical_features)

def replace_nan_with_unknown(df, categorical_features):
    """
    Replace actual NaN values (not 'nan' strings) with 'unknown' for categorical columns.
    """
    for col in categorical_features:
        df[col] = df[col].fillna('unknown')
    return df

data = replace_nan_with_unknown(data, categorical_features)

# Optionally convert the categorical columns to 'category' type
def convert_to_categorical(df, categorical_features):
    """
    Convert columns to 'category' type after converting all values to strings.
    """
    for col in categorical_features:
        df[col] = df[col].astype('category')
    return df

data = convert_to_categorical(data, categorical_features)

def check_categorical_columns_for_numeric(df, categorical_features):
    for col in categorical_features:

        try:
            pd.to_numeric(df[col], errors='raise')
            print(f"Warning: Column {col} contains numeric values but is classified as categorical.")
        except ValueError:
            pass

check_categorical_columns_for_numeric(data, categorical_features)


# Convert numeric-like categorical columns to strings explicitly
columns_to_convert = ['omsorgsniva', 'institusjonid', 'module_number_icd']

def convert_numeric_to_strings(df, columns):
    for col in columns:
        df[col] = df[col].astype(str)  
    return df
data = convert_numeric_to_strings(data, columns_to_convert)


# Check if categorical columns have numeric-like string values
def check_for_numeric_like_strings(df, categorical_features):
    numeric_like_columns = []
    for col in categorical_features:
        if df[col].str.match(r'^\d+(\.\d+)?$').any():  
            numeric_like_columns.append(col)
    return numeric_like_columns

# Run the check
numeric_like_columns = check_for_numeric_like_strings(data, categorical_features)

if numeric_like_columns:
    print(f"These columns have numeric-like string values: {numeric_like_columns}")
else:
    print("No numeric-like string values found in categorical columns.")


numeric_like_columns = ['fagomrade', 'omsnivahenv', 'pasient_fylkesnummer', 'pasient_kjonn_verdi', 
                        'samtykkekompetanse', 'aktivitetskategori', 'omsorgsniva', 'hdg', 
                        'innmatehast', 'henvfratjeneste', 'henvtiltjeneste', 'institusjonid', 
                        'substance_use', 'inperson_contact_kpr', 'speech_therapy_kpr', 
                        'fys_group_kpr', 'diag_importance_kpr', 'utlevering_resepttype_verdi', 
                        'interaction_flag', 'patient_overall_adherence_status', 
                        'module_number_icd', 'module_number_atc']

def force_convert_to_string(df, columns):
    for col in columns:
        df[col] = df[col].astype(str)
        print(f"Column {col} has been converted to string. Data type is now {df[col].dtype}")
    return df

data = force_convert_to_string(data, numeric_like_columns)

def format_numeric_like_strings(df, columns):
    for col in columns:
        df[col] = df[col].apply(lambda x: '{:.1f}'.format(float(x)) if x.replace('.', '', 1).isdigit() else x)
    return df

data = format_numeric_like_strings(data, numeric_like_columns)



#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
# Pipeline start
##########################################################################
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, average_precision_score, classification_report
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import lightgbm as lgb
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
import joblib
from sklearn.neural_network import MLPClassifier
import gc  
import pandas as pd  
from collections import defaultdict

# Define your functions
def create_cv_splits(X, y, n_splits=5, random_state=42):
    """Create stratified k-fold CV splits"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(skf.split(X, y))

def preprocess_data(X_train, X_val, numerical_features, categorical_features):
    """Preprocess both training and validation data"""
    # Numerical preprocessing
    num_imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()

    # Categorical preprocessing
    cat_imputer = SimpleImputer(strategy='constant', fill_value='unknown')
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    # Process numerical features
    X_train_num = num_imputer.fit_transform(X_train[numerical_features])
    X_train_num = scaler.fit_transform(X_train_num)
    X_val_num = num_imputer.transform(X_val[numerical_features])
    X_val_num = scaler.transform(X_val_num)

    # Process categorical features
    X_train_cat = X_train[categorical_features].astype(str)
    X_val_cat = X_val[categorical_features].astype(str)
    X_train_cat = cat_imputer.fit_transform(X_train_cat)
    X_train_cat = encoder.fit_transform(X_train_cat)
    X_val_cat = cat_imputer.transform(X_val_cat)
    X_val_cat = encoder.transform(X_val_cat)

    # Combine features
    X_train_processed = np.hstack([X_train_num, X_train_cat])
    X_val_processed = np.hstack([X_val_num, X_val_cat])

    # Generate feature names
    feature_names = np.concatenate([
        numerical_features,
        encoder.get_feature_names_out(categorical_features)
    ])

    return X_train_processed, X_val_processed, feature_names

def select_features_filter(X, y, feature_names, k=20):
    """Filter-based feature selection with output reports."""
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y)

    scores = selector.scores_
    pvalues = selector.pvalues_
    selected_features = feature_names[selector.get_support()]

    # Create DataFrame for better output structure
    results_df = pd.DataFrame({
        'Feature': feature_names,
        'Score': scores,
        'P-Value': pvalues,
        'Selected': np.isin(feature_names, selected_features)
    }).sort_values(by='Score', ascending=False)

    return X_selected, selected_features, results_df

def select_features_embedded(X, y, feature_names, alpha=0.01):
    """Embedded feature selection using Lasso with output reports."""
    selector = Lasso(alpha=alpha, random_state=42)
    selector.fit(X, y)

    mask = selector.coef_ != 0
    selected_features = feature_names[mask]

    results_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': selector.coef_,
        'Selected': mask
    }).sort_values(by='Coefficient', ascending=False)

    return X[:, mask], selected_features, results_df

def combine_selected_features(X, feature_names, feature_sets, method='union',
                              filter_scores=None, filter_pvalues=None, embedded_coefs=None):
    """Combine features using union with detailed tabular output."""

    if method == 'union':
        final_features = np.unique(np.concatenate(feature_sets))
    else:
        raise ValueError("Currently only 'union' method is supported.")

    results_df = pd.DataFrame({'Feature': feature_names})

    results_df['Selected by Filter'] = np.isin(feature_names, feature_sets[0])
    results_df['Selected by Embedded'] = np.isin(feature_names, feature_sets[1])

    conditions = [
        (results_df['Selected by Filter'] & results_df['Selected by Embedded']),
        results_df['Selected by Filter'],
        results_df['Selected by Embedded']
    ]
    choices = ['Both', 'Filter', 'Embedded']
    results_df['Selection Method'] = np.select(conditions, choices, default='-')

    # Add scores safely
    if filter_scores is not None:
        results_df['Score (Filter)'] = np.nan
        mask = results_df['Selected by Filter']
        results_df.loc[mask, 'Score (Filter)'] = np.array(filter_scores)[np.where(np.isin(feature_names, feature_sets[0]))[0]]

    if filter_pvalues is not None:
        results_df['P-Value (Filter)'] = np.nan
        mask = results_df['Selected by Filter']
        results_df.loc[mask, 'P-Value (Filter)'] = np.array(filter_pvalues)[np.where(np.isin(feature_names, feature_sets[0]))[0]]

    if embedded_coefs is not None:
        results_df['Coefficient (Embedded)'] = np.nan
        mask = results_df['Selected by Embedded']
        results_df.loc[mask, 'Coefficient (Embedded)'] = np.array(embedded_coefs)[np.where(np.isin(feature_names, feature_sets[1]))[0]]

    # Get final feature matrix
    feature_indices = np.isin(feature_names, final_features)
    X_selected = X[:, feature_indices]

    return X_selected, final_features, results_df

def evaluate_and_report(model_name, y_true, y_pred, y_pred_proba):
    """Evaluate model performance and print metrics without plotting the ROC curve."""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'ROC AUC': roc_auc_score(y_true, y_pred_proba),
        'F1-Score': f1_score(y_true, y_pred),
        'AUC-PR': average_precision_score(y_true, y_pred_proba)
    }

    print(f"\n{model_name} Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    return metrics

def train_base_models(X_train, y_train, X_val, y_val):
    """Train models and display performance metrics."""
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(32, 32), max_iter=100, early_stopping=True, random_state=42),
        'LightGBM': lgb.LGBMClassifier(random_state=42)
    }

    probas, explainers, shap_values = {}, {}, {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]

        # Evaluate and print metrics
        evaluate_and_report(name, y_val, y_pred, y_pred_proba)

        # Compute SHAP values
        if name == 'LightGBM':
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_val)
            shap_values[name] = shap_vals
        else:
            explainer = None

        probas[name] = y_pred_proba
        explainers[name] = explainer

    return models, probas, explainers, shap_values

# # stacking with RF as a final estimator
# #----------------------------------------
# def train_stacked_model(X_train, y_train, X_val):
#     estimators = [
#         ('lr', LogisticRegression(max_iter=1000, random_state=42)),
#         ('nn', MLPClassifier(hidden_layer_sizes=(32, 32), max_iter=100, early_stopping=True, random_state=42)),
#         ('lgb', lgb.LGBMClassifier(random_state=42))
#     ]
    
#     # Using RandomForest as meta-learner with carefully chosen parameters
#     final_estimator = RandomForestClassifier(
#         n_estimators=200,          # More trees for better performance
#         max_depth=5,               # Control depth to prevent overfitting
#         min_samples_split=5,       # Minimum samples required to split
#         min_samples_leaf=2,        # Minimum samples in leaf nodes
#         class_weight='balanced',   # Handle any class imbalance
#         n_jobs=-1,                # Use all available cores
#         random_state=42
#     )
    
#     stack_clf = StackingClassifier(
#         estimators=estimators,
#         final_estimator=final_estimator,
#         cv=5,                     # 5-fold cross-validation for meta-features
#         n_jobs=-1,               # Parallel processing
#         passthrough=False        # Can be set to True to include original features
#     )
    
#     stack_clf.fit(X_train, y_train)
#     probas = stack_clf.predict_proba(X_val)[:, 1]
#     return stack_clf, probas

# stacking with LR as a final estimator
#----------------------------------------
# def train_stacked_model(X_train, y_train, X_val):
#     estimators = [
#         ('lr', LogisticRegression(max_iter=1000, random_state=42)),
#         ('nn', MLPClassifier(hidden_layer_sizes=(32, 32), max_iter=100, early_stopping=True, random_state=42)), # change to hidden_layer_sizes=(64, 32) and increase max_iter=200 for better performcance
#         ('lgb', lgb.LGBMClassifier(random_state=42))
#     ]
    
#     stack_clf = StackingClassifier(
#         estimators=estimators,
#         final_estimator=LogisticRegression(max_iter=1000),
#         cv=None
#     )
    
#     stack_clf.fit(X_train, y_train)
#     probas = stack_clf.predict_proba(X_val)[:, 1]
#     return stack_clf, probas


def train_stacked_model(X_train, y_train, X_val):
    estimators = [
        ('lr', LogisticRegression(max_iter=1000, random_state=42)),
        ('nn', MLPClassifier(hidden_layer_sizes=(32, 32), max_iter=100, early_stopping=True, random_state=42)),
        ('lgb', lgb.LGBMClassifier(random_state=42))
    ]

    # Using MLPClassifier as meta-learner with a different architecture
    final_estimator = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=300,
        early_stopping=True,
        learning_rate='adaptive',
        random_state=42
    )

    stack_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=None,
        passthrough=False
    )

    stack_clf.fit(X_train, y_train)
    probas = stack_clf.predict_proba(X_val)[:, 1]
    return stack_clf, probas

def plot_roc_curve(y_true, probas_dict, fold=None):
    plt.figure(figsize=(10, 8))

    for model_name, y_pred_proba in probas_dict.items():
        RocCurveDisplay.from_predictions(y_true, y_pred_proba, name=model_name, ax=plt.gca())

    title = f"ROC Curves - Fold {fold}" if fold is not None else "ROC Curves - All Models"
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

def plot_shap_summary(shap_values, X_val, feature_names, plot_type='summary'):
    if shap_values is not None and X_val is not None and feature_names is not None:
        if plot_type == 'summary':
            shap.summary_plot(shap_values, X_val, feature_names=feature_names)
        elif plot_type == 'violin':
            shap.summary_plot(shap_values, X_val, feature_names=feature_names, plot_type='violin')
        elif plot_type == 'bar':
            shap.summary_plot(shap_values, X_val, feature_names=feature_names, plot_type='bar')
        else:
            print(f"Unknown plot type '{plot_type}'. Please choose 'summary', 'violin', or 'bar'.")
    else:
        print("Missing data for SHAP plotting: ensure shap_values, X_val, and feature_names are not None.")

def save_model(model, filename='final_model.joblib', feature_names=None):
    joblib.dump(model, filename)
    print(f"Model saved as '{filename}'")

def bootstrap_metrics(y_true, y_pred_proba, n_folds=5, n_bootstraps=1000, metrics_list=['Accuracy', 'ROC AUC', 'F1-Score', 'AUC-PR'], random_state=42):

    np.random.seed(random_state)
    n_samples = len(y_true)
    fold_size = n_samples // n_folds
    bootstrapped_metrics = {metric: [] for metric in metrics_list}
    
    for _ in range(n_bootstraps):
        fold_metrics = {metric: [] for metric in metrics_list}
        
        # Bootstrap within each fold
        for fold in range(n_folds):
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < n_folds - 1 else n_samples
            
            # Get fold data
            fold_y_true = y_true[start_idx:end_idx]
            fold_y_pred_proba = y_pred_proba[start_idx:end_idx]
            
            # Stratified sampling within fold
            pos_idx = np.where(fold_y_true == 1)[0]
            neg_idx = np.where(fold_y_true == 0)[0]
            if len(pos_idx) == 0 or len(neg_idx) == 0:
                continue  # Skip folds with only one class
            
            n_pos = len(pos_idx)
            n_neg = len(neg_idx)
            
            # Sample with replacement while maintaining stratification
            pos_sample_idx = np.random.choice(pos_idx, size=n_pos, replace=True)
            neg_sample_idx = np.random.choice(neg_idx, size=n_neg, replace=True)
            
            # Combine and shuffle indices
            fold_indices = np.concatenate([pos_sample_idx, neg_sample_idx])
            np.random.shuffle(fold_indices)
            
            # Get bootstrapped samples for this fold
            y_true_fold_sample = fold_y_true[fold_indices]
            y_pred_proba_fold_sample = fold_y_pred_proba[fold_indices]
            y_pred_fold_sample = (y_pred_proba_fold_sample >= 0.5).astype(int)
            
            # Calculate metrics for this fold
            for metric in metrics_list:
                try:
                    if metric == 'Accuracy':
                        value = accuracy_score(y_true_fold_sample, y_pred_fold_sample)
                    elif metric == 'ROC AUC':
                        value = roc_auc_score(y_true_fold_sample, y_pred_proba_fold_sample)
                    elif metric == 'F1-Score':
                        value = f1_score(y_true_fold_sample, y_pred_fold_sample)
                    elif metric == 'AUC-PR':
                        value = average_precision_score(y_true_fold_sample, y_pred_proba_fold_sample)
                    fold_metrics[metric].append(value)
                except ValueError:
                    continue
        
        # Average metrics across folds for this bootstrap iteration
        for metric in metrics_list:
            if fold_metrics[metric]:  
                avg_metric = np.mean(fold_metrics[metric])
                bootstrapped_metrics[metric].append(avg_metric)
    
    # Compute confidence intervals
    ci_dict = {}
    for metric in metrics_list:
        values = np.array(bootstrapped_metrics[metric])
        values = values[~np.isnan(values)]  
        if len(values) > 0:
            lower = np.percentile(values, 2.5)
            upper = np.percentile(values, 97.5)
            ci_dict[metric] = (lower, upper)
        else:
            ci_dict[metric] = (np.nan, np.nan)
            print(f"Warning: No valid bootstrap samples for {metric}")
    
    return ci_dict

def calculate_stacked_model_ci(y_true, y_pred_proba, n_bootstraps=1000, random_state=42):
    np.random.seed(random_state)
    n_samples = len(y_true)
    metrics_list = ['Accuracy', 'ROC AUC', 'F1-Score', 'AUC-PR']
    bootstrapped_metrics = {metric: [] for metric in metrics_list}

    for _ in range(n_bootstraps):
        # Sample with replacement from the entire dataset
        sample_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_sample = y_true[sample_indices]
        y_pred_proba_sample = y_pred_proba[sample_indices]
        y_pred_sample = (y_pred_proba_sample >= 0.5).astype(int)

        # Calculate metrics for the bootstrap sample
        try:
            bootstrapped_metrics['Accuracy'].append(accuracy_score(y_true_sample, y_pred_sample))
            bootstrapped_metrics['ROC AUC'].append(roc_auc_score(y_true_sample, y_pred_proba_sample))
            bootstrapped_metrics['F1-Score'].append(f1_score(y_true_sample, y_pred_sample))
            bootstrapped_metrics['AUC-PR'].append(average_precision_score(y_true_sample, y_pred_proba_sample))
        except ValueError:
            continue  

    stacked_ci_dict = {}
    for metric, values in bootstrapped_metrics.items():
        if values:
            lower = np.percentile(values, 2.5)
            upper = np.percentile(values, 97.5)
            stacked_ci_dict[metric] = (lower, upper)
        else:
            stacked_ci_dict[metric] = (np.nan, np.nan)
            print(f"Warning: No valid bootstrap samples for {metric}")

    return stacked_ci_dict

y_trues = defaultdict(list)
y_probas = defaultdict(list)

# Execute the functions
cv_splits = create_cv_splits(X, y, n_splits=5, random_state=42)

# Process each fold
for fold, (train_idx, val_idx) in enumerate(cv_splits):
    print(f"\nProcessing fold {fold + 1}")

    # Split data into train and validation sets
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Preprocess the data
    X_train_proc, X_val_proc, feature_names = preprocess_data(
        X_train, X_val, numerical_features, categorical_features
    )
    del X_train, X_val  
    gc.collect()

    # Perform feature selection
    print("Performing feature selection...")

    # Filter-based selection
    print("Filter Method: Selecting features based on ANOVA F-test scores")
    X_filter, filter_features, filter_results_df = select_features_filter(
        X_train_proc, y_train, feature_names, k=20
    )

    # Embedded selection
    print("Embedded Method (Lasso): Selecting features with non-zero coefficients.")
    X_embedded, embedded_features, embedded_results_df = select_features_embedded(
        X_train_proc, y_train, feature_names, alpha=0.01
    )

    # Combine selected features using "union"
    print("Combining features selected by Filter and Embedded methods using 'union'.")
    feature_sets = [filter_features, embedded_features]
    X_train_selected, final_features, combined_results_df = combine_selected_features(
        X_train_proc, feature_names, feature_sets, method='union',
        filter_scores=filter_results_df['Score'].values,
        filter_pvalues=filter_results_df['P-Value'].values,
        embedded_coefs=embedded_results_df['Coefficient'].values
    )

    # Clean up and prepare validation data
    del X_train_proc
    gc.collect()

    # Prepare validation data with selected features
    X_val_selected = X_val_proc[:, np.isin(feature_names, final_features)]

    print(f"\nFinal number of selected features: {len(final_features)}")

    # Train individual models
    print("\nTraining base models...")
    models, probas, explainers, shap_values = train_base_models(
        X_train_selected, y_train, X_val_selected, y_val
    )

    # Collect validation labels and predictions
    for model_name, y_pred_proba in probas.items():
        y_trues[model_name].extend(y_val)
        y_probas[model_name].extend(y_pred_proba)

    # Evaluate and report performance for each model
    for model_name, y_pred_proba in probas.items():
        print(f"\nPerformance metrics for {model_name}:")
        y_pred = (y_pred_proba >= 0.5).astype(int)
        evaluate_and_report(model_name, y_val, y_pred, y_pred_proba)

    # Train stacked model
    print("\nTraining stacked model...")
    stack_clf, stack_probas = train_stacked_model(
        X_train_selected, y_train, X_val_selected
    )

    # Collect stacked model predictions
    y_trues['Stacked Model'].extend(y_val)
    y_probas['Stacked Model'].extend(stack_probas)

    del X_train_selected, y_train
    gc.collect()

    # Evaluate predictions for stacked model
    print(f"\nFold {fold + 1} Stacked Model Performance:")
    y_pred_stack = (stack_probas >= 0.5).astype(int)
    evaluate_and_report("Stacked Model", y_val, y_pred_stack, stack_probas)

    # Include stacked model's probabilities into probas dictionary
    probas['Stacked Model'] = stack_probas

    # Plot ROC curves for all models including stacked model
    print(f"\nPlotting ROC Curves for Fold {fold + 1} including Stacked Model...")
    plot_roc_curve(y_val, probas, fold=fold + 1)

    # Plot SHAP summary for LightGBM if available
    if shap_values.get('LightGBM', None) is not None:
        print(f"\nGenerating SHAP plots for Fold {fold + 1} (LightGBM)...")
        shap_vals = shap_values['LightGBM']
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        plot_shap_summary(shap_vals, X_val_selected, final_features, plot_type='summary')
        plot_shap_summary(shap_vals, X_val_selected, final_features, plot_type='violin')
        plot_shap_summary(shap_vals, X_val_selected, final_features, plot_type='bar')

    del probas  
    gc.collect()

    save_model(stack_clf, f'model_fold_{fold+1}.joblib', final_features)
    del stack_clf  
    gc.collect()

    del X_val_selected, y_val  
    gc.collect()

print("\nTraining completed!")

print("\nCalculating 95% Confidence Intervals for Base Models using Bootstrapping...")

for model_name in y_trues.keys():
    if model_name == 'Stacked Model':
        continue

    y_true_all = np.array(y_trues[model_name])
    y_pred_proba_all = np.array(y_probas[model_name])

    # Perform bootstrapping
    ci_dict = bootstrap_metrics(
        y_true_all,                 
        y_pred_proba_all,            
        n_folds=5,                   
        n_bootstraps=1000,           
        metrics_list=['Accuracy', 'ROC AUC', 'F1-Score', 'AUC-PR'],  
        random_state=42              
    )

    print(f"\n95% Confidence Intervals for {model_name}:")
    for metric, (lower, upper) in ci_dict.items():
        print(f"{metric}: {lower:.4f} - {upper:.4f}")        
        
        
# Perform bootstrapping for the Stacked Model
stacked_ci_dict = calculate_stacked_model_ci(
    y_true=np.array(y_trues['Stacked Model']),  
    y_pred_proba=np.array(y_probas['Stacked Model']),  
    n_bootstraps=1000,  
    random_state=42  
)

# Print the results for the stacked model
print("\n95% Confidence Intervals for Stacked Model:")
for metric, (lower, upper) in stacked_ci_dict.items():
    print(f"{metric}: {lower:.4f} - {upper:.4f}")

