import pandas as pd
import numpy as np
import joblib
import os
from skopt.plots import plot_convergence
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                           f1_score, roc_auc_score, confusion_matrix,
                           classification_report, roc_curve, precision_recall_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.metrics import roc_auc_score
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
from skopt.space import Integer, Real, Categorical
from sklearn.feature_selection import RFECV


def train_and_save_model(df: pd.DataFrame, 
                        target_col: str = 'outcome', 
                        top_n_features: int = 30,
                        use_bayesian: bool = False, 
                        bayesian_iter: int = 50):
    """
    Training function that uses parameters from grid_best_params.csv without search
    
    Args:
        df: Input DataFrame containing features and target
        target_col: Name of the target column
        top_n_features: Number of top features to select
        use_bayesian: Flag for Bayesian optimization (not used in this function)
        bayesian_iter: Number of iterations for Bayesian optimization (not used)
        
    Returns:
        Trained model pipeline
    """
    # Create artifacts directory structure
    os.makedirs('artifacts/plots', exist_ok=True)
    os.makedirs('artifacts/metrics', exist_ok=True)
    os.makedirs('artifacts/models', exist_ok=True)
    
    # 1. Prepare Data with time-series ordering
    print("Preparing time-series data...")
    if 'date' in df.columns:
        df = df.sort_values('date')
    y = df.loc[:, target_col].copy()
    X = prepare_features(df, target_col)
    
    # 2. Time-series split
    tscv = TimeSeriesSplit(n_splits=5)
    print(f"Using {tscv.n_splits} time-series splits")
    
    # 3. Identify feature types and ensure consistent data types
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Convert ALL categorical features to strings and handle mixed types
    for col in categorical_features:
        X.loc[:, col] = X[col].astype(str)
        if any(str(x).isdigit() for x in X[col] if not pd.isna(x)):
            X.loc[:, col] = X[col].apply(lambda x: f"num_{x}" if str(x).isdigit() else x)
    
    # 4. Define preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )

    # 5. Load parameters from CSV
    print("\nLoading parameters from grid_best_params.csv...")
    params_file = 'artifacts/metrics/grid_best_params.csv'
    specific_params = {}
    
    try:
        params_df = pd.read_csv(params_file)
        
        # Handle the specific format with unnamed first column
        if len(params_df.columns) >= 2:
            # Use first column as index, second column as values
            specific_params = dict(zip(
                params_df.iloc[:, 0].dropna().astype(str).str.strip(),
                params_df.iloc[:, 1].dropna().astype(str).str.strip()
            ))
            print("Successfully loaded parameters:")
            for param, val in specific_params.items():
                print(f"  {param}: {val}")
        else:
            print("Warning: CSV doesn't have enough columns. Using default parameters.")
            
    except Exception as e:
        print(f"Warning: Could not load parameters - {str(e)}. Using default parameters.")

    # 6. Create pipeline with loaded parameters
    print("\nCreating pipeline with loaded parameters...")
    
    # Convert parameter values to correct types
    param_types = {
        'classifier__n_estimators': int,
        'classifier__max_depth': lambda x: None if x == 'None' else int(x),
        'classifier__min_samples_split': int,
        'classifier__min_samples_leaf': int,
        'classifier__max_features': lambda x: None if x == 'None' else float(x) if '.' in x else int(x),
        'classifier__class_weight': lambda x: None if x == 'None' else x
    }
    
    classifier_params = {}
    for param, value in specific_params.items():
        if param in param_types:
            try:
                classifier_params[param.replace('classifier__', '')] = param_types[param](value)
            except (ValueError, TypeError) as e:
                print(f"Warning: Could not convert {param}={value}. Using default.")
    
    # Create base estimator with loaded parameters
    base_estimator = RandomForestClassifier(
        **classifier_params,
        random_state=42
    )
    
    # Create pipeline
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('variance_threshold', VarianceThreshold(threshold=0.01)),
        ('feature_selection', SelectFromModel(
            RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=50),
            max_features=top_n_features
        )),
        ('classifier', base_estimator)
    ])
    
    model = model_pipeline
    
    # 7. Time-series cross-validation with best model
    print("\nEvaluating best model with time-series cross-validation...")
    cv_metrics = []
    feature_importances = []
    class_names = ['Draw (0)', 'Home Win (1)', 'Away Win (2)']
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Verify shapes before probability prediction
        try:
            y_proba = model.predict_proba(X_test)
            assert len(y_test) == len(y_proba), "Shape mismatch between y_test and y_proba"
            
            metrics = {
                'fold': fold + 1,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision_macro': precision_score(y_test, y_pred, average='macro'),
                'recall_macro': recall_score(y_test, y_pred, average='macro'),
                'f1_macro': f1_score(y_test, y_pred, average='macro'),
                'roc_auc_ovr': roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro'),
                'roc_auc_ovo': roc_auc_score(y_test, y_proba, multi_class='ovo', average='macro'),
                'log_loss': log_loss(y_test, y_proba),
                'brier_score': multi_class_brier_score(y_test, y_proba),
                'confidence': np.mean(np.max(y_proba, axis=1)),
                'uncertainty_rate': np.mean(np.max(y_proba, axis=1) < 0.6)
            }
            
        except Exception as e:
            print(f"Error in fold {fold + 1}: {str(e)}")
            metrics = {
                'fold': fold + 1,
                'error': str(e)
            }
        
        cv_metrics.append(metrics)
        
        if fold == 0 and 'error' not in metrics:
            try:
                feature_names = get_feature_names_from_pipeline(model, numerical_features, categorical_features)
                selected_features = get_selected_features(model, feature_names)
                importances = model.named_steps['classifier'].feature_importances_
                feature_importances = list(zip(selected_features, importances))
            except Exception as e:
                print(f"Feature importance extraction failed: {str(e)}")
                feature_importances = []
        
        if 'error' not in metrics:
            print(f"Fold {fold + 1} - Accuracy: {metrics['accuracy']:.4f}, ROC-AUC (OvR): {metrics['roc_auc_ovr']:.4f}")
    
    # 8. Final training and evaluation
    print("\nTraining final model on full dataset...")
    model.fit(X, y)
    
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    
    # Calculate final metrics with error handling
    final_metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision_macro': precision_score(y, y_pred, average='macro'),
        'recall_macro': recall_score(y, y_pred, average='macro'),
        'f1_macro': f1_score(y, y_pred, average='macro'),
        'class_report': classification_report(y, y_pred, target_names=class_names, output_dict=True)
    }
    
    try:
        final_metrics.update({
            'roc_auc_ovr': roc_auc_score(y, y_proba, multi_class='ovr', average='macro'),
            'roc_auc_ovo': roc_auc_score(y, y_proba, multi_class='ovo', average='macro'),
            'log_loss': log_loss(y, y_proba),
            'brier_score': multi_class_brier_score(y, y_proba),
            'probability_metrics': {
                'mean_prob_class_0': np.mean(y_proba[:, 0]),
                'mean_prob_class_1': np.mean(y_proba[:, 1]),
                'mean_prob_class_2': np.mean(y_proba[:, 2]),
                'confidence': np.mean(np.max(y_proba, axis=1))
            }
        })
    except Exception as e:
        print(f"Error in final metrics calculation: {str(e)}")
        final_metrics['error'] = str(e)
    
    # 9. Save all artifacts
    print("\nSaving all artifacts...")
    save_all_artifacts(
        model=model,
        cv_metrics=cv_metrics,
        final_metrics=final_metrics,
        feature_importances=feature_importances,
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        class_names=class_names,
        X=X,
        y=y,
        is_bayesian=use_bayesian
    )
    
    return model


def multi_class_brier_score(y_true, y_proba):
    """Safe calculation of multi-class Brier score with shape validation"""
    if len(y_true) != len(y_proba):
        raise ValueError(f"Shape mismatch: y_true ({len(y_true)}) vs y_proba ({len(y_proba)})")
    
    try:
        n_classes = y_proba.shape[1]
        y_true_bin = np.zeros((len(y_true), n_classes))
        for i in range(n_classes):
            y_true_bin[y_true == i, i] = 1
        return np.mean(np.sum((y_proba - y_true_bin) ** 2, axis=1))
    except Exception as e:
        print(f"Brier score calculation failed: {str(e)}")
        return np.nan

def save_all_artifacts(
    model,
    cv_metrics,
    final_metrics,
    feature_importances,
    numerical_features,
    categorical_features,
    class_names,
    X,
    y,
    search_results=None,
    is_bayesian=False
):
    """Enhanced version saving to CSV with improved formatting"""
    
    # 1. Save the trained model (unchanged)
    model_path = 'artifacts/models/football_model.pkl'
    joblib.dump(model, model_path)
    
    # 2. Enhanced CSV saving with better formatting
    # Cross-validation results - add fold description
    cv_df = pd.DataFrame(cv_metrics)
    cv_df.insert(0, 'fold_description', ['Fold ' + str(i) for i in cv_df['fold']])
    cv_df.to_csv('artifacts/metrics/cross_validation_results.csv', index=False, float_format='%.4f')
    
    # Classification report - improved formatting
    class_report = final_metrics['class_report']
    class_report_df = pd.DataFrame(class_report).transpose().round(4)
    
    # Remove redundant accuracy row and add it to final metrics
    accuracy_row = class_report_df.loc['accuracy']
    class_report_df = class_report_df.drop('accuracy')
    
    # Save classification report
    class_report_df.to_csv('artifacts/metrics/classification_report.csv', float_format='%.4f')
    
    # 3. Enhanced final metrics with fold statistics
    final_metrics_flat = {
        'metric': [
            'accuracy', 'precision_macro', 'recall_macro', 'f1_macro',
            'roc_auc_ovr', 'roc_auc_ovo', 'log_loss', 'brier_score',
            'mean_prob_class_0', 'mean_prob_class_1', 'mean_prob_class_2',
            'confidence'
        ],
        'value': [
            final_metrics['accuracy'],
            final_metrics['precision_macro'],
            final_metrics['recall_macro'],
            final_metrics['f1_macro'],
            final_metrics['roc_auc_ovr'],
            final_metrics['roc_auc_ovo'],
            final_metrics['log_loss'],
            final_metrics['brier_score'],
            final_metrics['probability_metrics']['mean_prob_class_0'],
            final_metrics['probability_metrics']['mean_prob_class_1'],
            final_metrics['probability_metrics']['mean_prob_class_2'],
            final_metrics['probability_metrics']['confidence']
        ],
        'cv_mean': [
            cv_df['accuracy'].mean(),
            cv_df['precision_macro'].mean(),
            cv_df['recall_macro'].mean(),
            cv_df['f1_macro'].mean(),
            cv_df['roc_auc_ovr'].mean(),
            cv_df['roc_auc_ovo'].mean(),
            cv_df['log_loss'].mean(),
            cv_df['brier_score'].mean(),
            np.nan,  # Not available in CV
            np.nan,
            np.nan,
            cv_df['confidence'].mean()
        ],
        'cv_std': [
            cv_df['accuracy'].std(),
            cv_df['precision_macro'].std(),
            cv_df['recall_macro'].std(),
            cv_df['f1_macro'].std(),
            cv_df['roc_auc_ovr'].std(),
            cv_df['roc_auc_ovo'].std(),
            cv_df['log_loss'].std(),
            cv_df['brier_score'].std(),
            np.nan,
            np.nan,
            np.nan,
            cv_df['confidence'].std()
        ]
    }
    pd.DataFrame(final_metrics_flat).to_csv(
        'artifacts/metrics/final_metrics.csv',
        index=False,
        float_format='%.4f'
    )
    
    # 4. Feature importances with additional stats
    feature_imp_df = pd.DataFrame(feature_importances, columns=['feature', 'importance'])
    feature_imp_df['importance_pct'] = (feature_imp_df['importance'] / feature_imp_df['importance'].sum()) * 100
    feature_imp_df = feature_imp_df.sort_values('importance', ascending=False)
    feature_imp_df.to_csv('artifacts/metrics/feature_importances.csv', index=False, float_format='%.4f')
    
    # 5. Enhanced preprocessing info
    preprocessing_info = pd.DataFrame({
        'feature_name': numerical_features + categorical_features,
        'feature_type': ['numerical'] * len(numerical_features) + ['categorical'] * len(categorical_features),
        'n_unique': [X[col].nunique() for col in numerical_features] + 
                   [X[col].nunique() for col in categorical_features],
        'missing_pct': [(X[col].isna().mean() * 100) for col in numerical_features] + 
                      [(X[col].isna().mean() * 100) for col in categorical_features]
    })
    preprocessing_info.to_csv('artifacts/metrics/preprocessing_info.csv', index=False, float_format='%.2f')
    
    # 6. Best parameters with additional info
    if 'best_params' in final_metrics:
        best_params_df = pd.DataFrame({
            'parameter': final_metrics['best_params'].keys(),
            'value': final_metrics['best_params'].values(),
            'parameter_type': [p.split('__')[0] if '__' in p else 'base' for p in final_metrics['best_params'].keys()]
        })
        best_params_df.to_csv('artifacts/metrics/best_params.csv', index=False)
    
    # 7. Generate visualizations (unchanged)
    generate_visualizations(
        model=model,
        X=X,
        y=y,
        class_names=class_names,
        feature_imp_df=feature_imp_df,
        cv_metrics=cv_metrics,
        search_results=search_results,
        is_bayesian=is_bayesian
    )
    
    # Print summary
    print("\nSaved artifacts:")
    print(f"- Model: {model_path}")
    print("- Evaluation metrics (CSV):")
    print("  - Cross-validation: artifacts/metrics/cross_validation_results.csv")
    print("  - Final metrics: artifacts/metrics/final_metrics.csv")
    print("  - Classification report: artifacts/metrics/classification_report.csv")
    print("  - Feature importances: artifacts/metrics/feature_importances.csv")
    print("  - Preprocessing info: artifacts/metrics/preprocessing_info.csv")
    print("  - Best parameters: artifacts/metrics/best_params.csv")

def generate_multi_class_visualizations(model, X, class_names, feature_imp_df, cv_metrics):
    """Generate visualizations for multi-class problem"""
    plt.figure(figsize=(18, 12))
    
    # 1. Feature Importance Plot
    plt.subplot(2, 2, 1)
    top_features = feature_imp_df.sort_values('importance', ascending=False).head(20)
    sns.barplot(x='importance', y='feature', data=top_features)
    plt.title('Top 20 Feature Importances')
    
    # 2. CV Metrics Trend
    plt.subplot(2, 2, 2)
    cv_df = pd.DataFrame(cv_metrics)
    metrics_to_plot = ['accuracy', 'roc_auc_ovr', 'f1_macro']
    for metric in metrics_to_plot:
        plt.plot(cv_df['fold'], cv_df[metric], label=metric)
    plt.title('Cross-Validation Metrics')
    plt.xlabel('Fold')
    plt.ylabel('Score')
    plt.legend()
    
    # 3. ROC Curves for each class (OvR)
    plt.subplot(2, 2, 3)
    y_proba = model.predict_proba(X)
    y = model.predict(X)
    
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve((y == i).astype(int), y_proba[:, i])
        plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc_score((y == i).astype(int), y_proba[:, i]):.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curves (One-vs-Rest)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    
    # 4. Confusion Matrix
    plt.subplot(2, 2, 4)
    cm = confusion_matrix(y, model.predict(X))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig('artifacts/plots/multi_class_evaluation.png')
    plt.close()
    
    # Additional plot: Class prediction distribution
    plt.figure(figsize=(8, 6))
    pd.Series(y).value_counts().plot(kind='bar')
    plt.title('Class Distribution in Predictions')
    plt.xlabel('Outcome Class')
    plt.ylabel('Count')
    plt.xticks(ticks=[0, 1, 2], labels=class_names, rotation=45)
    plt.savefig('artifacts/plots/class_distribution.png')
    plt.close()

def generate_visualizations(
    model,
    X,
    y,
    class_names,
    feature_imp_df,
    cv_metrics,
    search_results=None,
    is_bayesian=False
):
    """
    Enhanced visualization function that handles both Bayesian and grid search results
    """
    plt.figure(figsize=(20, 15))
    plt.suptitle('Model Evaluation Summary', y=1.02, fontsize=16)
    
    # 1. Feature Importance Plot (fixed warning)
    plt.subplot(2, 2, 1)
    top_features = feature_imp_df.nlargest(15, 'importance').sort_values('importance', ascending=True)
    sns.barplot(x='importance', y='feature', data=top_features, hue='feature', palette='viridis', legend=False)
    plt.title('Top 15 Feature Importances', pad=20)
    plt.xlabel('Importance Score', labelpad=10)
    plt.ylabel('Features', labelpad=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 2. CV Metrics Trend
    plt.subplot(2, 2, 2)
    cv_df = pd.DataFrame(cv_metrics)
    metrics_to_plot = ['accuracy', 'roc_auc_ovr', 'f1_macro']
    colors = sns.color_palette("husl", len(metrics_to_plot))
    
    for metric, color in zip(metrics_to_plot, colors):
        plt.plot(cv_df['fold'], cv_df[metric], label=metric.replace('_', ' ').title(), 
                color=color, linewidth=2.5, marker='o')
    
    plt.title('Cross-Validation Performance', pad=20)
    plt.xlabel('Fold Number', labelpad=10)
    plt.ylabel('Score', labelpad=10)
    plt.legend(frameon=True, facecolor='white')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(0, 1)
    
    # 3. Hyperparameter Tuning Results
    plt.subplot(2, 2, 3)
    if search_results is not None:
        if is_bayesian:
            # Bayesian optimization convergence plot
            try:
                from skopt.plots import plot_convergence
                if hasattr(search_results, 'optimizer_results_') and search_results.optimizer_results_:
                    plot_convergence(search_results.optimizer_results_[0])
                else:
                    plot_convergence(search_results)
                plt.title('Bayesian Optimization Convergence', pad=20)
            except Exception as e:
                plt.text(0.5, 0.5, f'Could not plot convergence:\n{str(e)}', 
                        ha='center', va='center')
                plt.title('Optimization Results', pad=20)
        else:
            # Grid search heatmap
            results = pd.DataFrame(search_results.cv_results_)
            heatmap_data = results.pivot_table(
                index='param_classifier__max_depth',
                columns='param_classifier__n_estimators',
                values='mean_test_score'
            )
            sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlGnBu", 
                       cbar_kws={'label': 'ROC-AUC Score'})
            plt.title('Grid Search Results', pad=20)
            plt.xlabel('Number of Estimators', labelpad=10)
            plt.ylabel('Max Depth', labelpad=10)
    else:
        plt.text(0.5, 0.5, 'No Hyperparameter Tuning Results', 
                ha='center', va='center', fontsize=12)
        plt.title('Hyperparameter Tuning', pad=20)
        plt.axis('off')
    
    # 4. Confusion Matrix (fixed annotation error)
    plt.subplot(2, 2, 4)
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create annotation labels
    annot_labels = []
    for i in range(cm.shape[0]):
        row_labels = []
        for j in range(cm.shape[1]):
            row_labels.append(f"{cm[i,j]}\n({cm_percent[i,j]:.1f}%)")
        annot_labels.append(row_labels)
    
    sns.heatmap(cm, annot=annot_labels, fmt='', cmap='Blues', 
               xticklabels=class_names, yticklabels=class_names,
               cbar_kws={'label': 'Number of Predictions'})
    
    plt.title('Confusion Matrix', pad=20)
    plt.xlabel('Predicted Label', labelpad=10)
    plt.ylabel('True Label', labelpad=10)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    plt.tight_layout(pad=3.0)
    plt.savefig('artifacts/plots/model_evaluation.png', bbox_inches='tight', dpi=300)
    plt.close()

    # 5. Probability Calibration Plot (new)
    plt.subplot(3, 3, 5)
    plot_calibration_curve(model, X, y, class_names)
    
    # 6. Probability Distribution (new)
    plt.subplot(3, 3, 6)
    plot_probability_distribution(model, X, y, class_names)
    
    # 7. ROC Curves (enhanced)
    plt.subplot(3, 3, 7)
    plot_multiclass_roc(model, X, y, class_names)
    
    plt.tight_layout(pad=3.0)
    plt.savefig('artifacts/plots/model_evaluation.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Additional: Save feature importance plot separately
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=top_features, hue='feature', 
               palette='viridis', legend=False)
    plt.title('Top 15 Feature Importances', pad=20)
    plt.xlabel('Importance Score', labelpad=10)
    plt.ylabel('Features', labelpad=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('artifacts/plots/feature_importances.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_calibration_curve(model, X, y, class_names):
    """Plot calibration curves for each class"""
    y_proba = model.predict_proba(X)
    
    for i, class_name in enumerate(class_names):
        fraction_of_positives, mean_predicted_value = calibration_curve(
            (y == i).astype(int), y_proba[:, i], n_bins=10
        )
        plt.plot(mean_predicted_value, fraction_of_positives, "s-",
                label=f"{class_name}")
    
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    plt.ylabel("Fraction of positives")
    plt.xlabel("Mean predicted value")
    plt.title('Calibration Plot')
    plt.legend()
    plt.grid()

def plot_probability_distribution(model, X, y, class_names):
    """Plot distribution of predicted probabilities"""
    y_proba = model.predict_proba(X)
    
    for i, class_name in enumerate(class_names):
        sns.kdeplot(y_proba[:, i], label=class_name, shade=True)
    
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.title('Probability Distribution')
    plt.legend()
    plt.grid()

def plot_multiclass_roc(model, X, y, class_names):
    """Plot ROC curves for each class (OvR)"""
    y_proba = model.predict_proba(X)
    
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve((y == i).astype(int), y_proba[:, i])
        roc_auc = roc_auc_score((y == i).astype(int), y_proba[:, i])
        plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (One-vs-Rest)')
    plt.legend()
    plt.grid()

def get_feature_names_from_pipeline(model, numerical_features, categorical_features):
    """Get all feature names after preprocessing"""
    preprocessor = model.named_steps['preprocessor']
    feature_names = numerical_features.copy()
    
    # Handle categorical features
    if 'cat' in preprocessor.named_transformers_:
        ohe = preprocessor.named_transformers_['cat']
        if hasattr(ohe, 'get_feature_names_out'):
            cat_features = ohe.get_feature_names_out(categorical_features)
            feature_names.extend(cat_features)
    
    # Handle remainder features
    if hasattr(preprocessor, 'transformers_'):
        for name, trans, cols in preprocessor.transformers_:
            if name == 'remainder' and trans == 'passthrough':
                remainder_features = [f for f in cols if f not in numerical_features 
                                   and f not in categorical_features]
                feature_names.extend(remainder_features)
    
    return feature_names

def get_selected_features(model, all_feature_names):
    """Get names of selected features"""
    selector = model.named_steps['feature_selection']
    selected_indices = selector.get_support()
    return [all_feature_names[i] for i in range(len(selected_indices)) if selected_indices[i]]

def prepare_features(df, target_col):
    """
    Robust feature preparation with:
    - Proper pandas indexing to avoid warnings
    - Leakage prevention
    - Type safety
    - Formation handling
    """
    # Create a copy of the DataFrame to avoid chained indexing warnings
    X = df.copy()

    
    
    # Columns that could cause data leakage (expanded list)
    leakage_columns = {
        'fixture_id', 'league_id', 'date', 'home_team', 'away_team',
        'home_team_id', 'away_team_id', 'referee', target_col, 'halftime_difference',
        'extratime_home', 'extratime_away', 'home_winner', 'away_winner',
        'season', 'penalty_home', 'penalty_away', 'halftime_home', 'halftime_away',
        'fulltime_home', 'fulltime_away', 'home_goals', 'away_goals', 'total_goals', 
        'goal_difference', 'year', 'goals_home', 'goals_away', 'match_result',
        'ft_score', 'ht_score', 'final_score', 'result', 'halftime_total', 'second_half_home',
        'second_half_away', 'second_half_total', 'home_margin', 'goals_product', 'big_win',
        'high_scoring', 'close_game', 'comeback', 'blown_lead', 'draw_after_lead'
    } & set(X.columns)  # Only existing columns
    
    # Proper column dropping using .loc
    columns_to_keep = [col for col in X.columns if col not in leakage_columns]
    X = X.loc[:, columns_to_keep]
    
    # Handle formations safely for both home and away
    for side in ['home', 'away']:
        formation_col = f'formation_{side}'
        if formation_col in X.columns:
            # Convert to string safely using .loc
            X.loc[:, formation_col] = X[formation_col].astype(str)
            
            # Standardize formation formats
            valid_formats = {
                f for f in X[formation_col].unique() 
                if isinstance(f, str) and (f.count('-') == 2 or f == 'unknown')
            }
            
            # Create dummy variables safely
            formation_series = X[formation_col].where(
                X[formation_col].isin(valid_formats),
                'other'
            )
            dummies = pd.get_dummies(formation_series, prefix=f'formation_{side}')
            
            # Safely concatenate and drop
            X = pd.concat([
                X.drop(columns=[formation_col]), 
                dummies
            ], axis=1)
    
    # Convert remaining categorical columns safely
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        X.loc[:, col] = X[col].astype(str)
    
    # Fill NA values safely
    numeric_cols = X.select_dtypes(include=np.number).columns
    X.loc[:, numeric_cols] = X[numeric_cols].fillna(0)
    
    # For categorical columns, fill with 'missing'
    if len(cat_cols) > 0:
        X.loc[:, cat_cols] = X[cat_cols].fillna('missing')
    
    print(f"Final features after preparation: {list(X.columns)}")
    return X


if __name__ == "__main__":
    try:
        # Load data
        df = pd.read_csv('data/merged/final_merged_dataset.csv')
        
        # Train and save
        trained_model = train_and_save_model(df, top_n_features=30, use_bayesian=False)
        print("\nTraining and evaluation completed successfully!")
        
    except Exception as e:
        print(f"\nTraining failed: {str(e)}")
        raise


