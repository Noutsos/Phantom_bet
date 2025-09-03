import pandas as pd
import numpy as np
import joblib
import os
from skopt.plots import plot_convergence
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PoissonRegressor
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel, RFE, SelectKBest, f_classif, RFECV
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit,RandomizedSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                           f1_score, roc_auc_score, confusion_matrix,
                           classification_report, roc_curve, precision_recall_curve)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.calibration import calibration_curve
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
from skopt.space import Integer, Real, Categorical
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*use_label_encoder.*")

# Define model parameter grids
MODEL_PARAMS = {
    'random_forest': {
        'bayesian': {
            'classifier__n_estimators': Integer(50, 500),
            'classifier__max_depth': Integer(3, 30),
            'classifier__min_samples_split': Integer(2, 20),
            'classifier__min_samples_leaf': Integer(1, 10),
            'classifier__max_features': Categorical(['sqrt', 'log2', None]),
            'classifier__bootstrap': Categorical([True, False]),
            'classifier__criterion': Categorical(['gini', 'entropy'])
        },
        'grid': {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [None, 5, 10, 20],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4],
            'classifier__max_features': ['sqrt', 'log2', None],
            'classifier__bootstrap': [True, False],
            'classifier__criterion': ['gini', 'entropy']
        }
    },
    'logistic_regression': {
        'bayesian': {
            'classifier__C': Real(1e-4, 1e4, prior='log-uniform'),
            'classifier__penalty': Categorical(['l1', 'l2']),
            'classifier__solver': Categorical(['liblinear', 'saga']),
            'classifier__max_iter': Integer(100, 1000)
        },
        'grid': {
            'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'classifier__penalty': ['l1', 'l2'],
            'classifier__solver': ['liblinear', 'saga'],
            'classifier__max_iter': [100, 500, 1000]
        }
    },
    'xgboost': {
        'bayesian': {
            'classifier__n_estimators': Integer(50, 300),  # Reduced from 50–500
            'classifier__max_depth': Integer(3, 9),       # Reduced from 3–15
            'classifier__learning_rate': Real(0.01, 0.2, prior='log-uniform'),
            'classifier__subsample': Real(0.7, 1.0),     # Reduced from 0.5–1.0
            'classifier__colsample_bytree': Real(0.7, 1.0),
            'classifier__gamma': Real(0, 2),              # Reduced from 0–5
            'classifier__reg_alpha': Real(0, 5),          # Reduced from 0–10
            'classifier__reg_lambda': Real(0, 5)
        },
        'grid': {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [3, 6, 9],
            'classifier__learning_rate': [0.01, 0.1, 0.2],
            'classifier__subsample': [0.6, 0.8, 1.0],
            'classifier__colsample_bytree': [0.6, 0.8, 1.0],
            'classifier__gamma': [0, 1, 5],
            'classifier__reg_alpha': [0, 0.1, 1],
            'classifier__reg_lambda': [0, 0.1, 1]
        }
    },
    'poisson': {
        'bayesian': {
            'classifier__alpha': Real(1e-4, 1e4, prior='log-uniform'),
            'classifier__max_iter': Integer(100, 1000)
        },
        'grid': {
            'classifier__alpha': [0.001, 0.01, 0.1, 1, 10],
            'classifier__max_iter': [100, 500, 1000]
        }
    }
}

# Define default parameter sets for each model type
DEFAULT_PARAMS = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': None,  # Unlimited depth
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'bootstrap': True,
        'class_weight': 'balanced',
        'n_jobs': -1  # Use all cores
    },
    'logistic_regression': {
        'C': 1.0,
        'penalty': 'l2',
        'solver': 'lbfgs',
        'max_iter': 1000,
        'class_weight': 'balanced',
        'multi_class': 'auto'
    },
    'xgboost': {
        'n_estimators': 200,
        'max_depth': 9,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 1.0,
        'gamma': 5.0,
        'reg_alpha': 0,
        'reg_lambda': 0,
        'objective': 'multi:softprob',
        'n_jobs': -1,
        'tree_method': 'hist'  # Faster histogram method
    },
    'poisson': {
        'alpha': 1.0,
        'max_iter': 1000,
        'tol': 1e-4
    }
}

def get_model_class(model_type):
    """Return the appropriate model class based on model_type"""
    if model_type == 'random_forest':
        return RandomForestClassifier(random_state=42, class_weight='balanced')
    elif model_type == 'logistic_regression':
        return LogisticRegression(random_state=42, class_weight='balanced', multi_class='ovr')
    elif model_type == 'xgboost':
        return XGBClassifier(random_state=42,  
                           eval_metric='mlogloss',
                           objective='multi:softprob')
    elif model_type == 'poisson':
        return PoissonRegressor(max_iter=500)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def get_feature_selector(feature_selection_method, model_type, top_n_features, tscv, target_variance=0.9):
    """Return feature selector with RFECV support added"""
    if feature_selection_method == 'importance':
        return SelectFromModel(
            RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=50),
            max_features=top_n_features
        )
    elif feature_selection_method == 'rfe':
        return RFE(
            estimator=get_model_class(model_type),
            n_features_to_select=top_n_features,
            step=0.1
        )
    elif feature_selection_method == 'rfecv':  # NEW METHOD
        return RFECV(
            estimator=get_model_class(model_type),
            step=1,
            cv=tscv,
            scoring='accuracy',
            min_features_to_select=top_n_features,
            n_jobs=-1
        )
    elif feature_selection_method == 'pca':
        return PCA(n_components=target_variance, svd_solver='full')
    elif feature_selection_method == 'anova':
        return SelectKBest(score_func=f_classif, k=top_n_features)
    else:
        raise ValueError(f"Unknown method: {feature_selection_method}")

def train_and_save_model(df: pd.DataFrame, 
                       target_col: str = 'outcome', 
                       model_type: str = 'random_forest',
                       feature_selection_method: str = 'importance',
                       top_n_features: int = 30,
                       use_bayesian: bool = False, 
                       bayesian_iter: int = 50,
                       use_grid_search: bool = False,
                       use_random_search: bool = False,  # NEW: Add random search option
                       random_search_iter: int = 50,    # NEW: Number of iterations for random search
                       load_params: bool = False):
    """
    Enhanced training function with:
    - Multiple model types
    - Multiple feature selection methods
    - Parameter loading from CSV
    - Bayesian optimization
    - Grid search
    - Randomized search (NEW)
    
    Args:
        df: Input DataFrame
        target_col: Target column name
        model_type: Type of model ('random_forest', 'logistic_regression', 'xgboost', 'poisson')
        feature_selection_method: Feature selection method ('importance', 'rfe', 'pca', 'anova')
        top_n_features: Number of features/components to select
        use_bayesian: Use Bayesian optimization if True
        bayesian_iter: Number of Bayesian iterations
        use_grid_search: Use grid search if True
        use_random_search: Use randomized search if True (NEW)
        random_search_iter: Number of random search iterations (NEW)
        load_params: Load parameters from CSV file if True
        
    Returns:
        Trained model pipeline
    """
    # Setup directories
    base_path = f"artifacts/{model_type}/{feature_selection_method}"
    os.makedirs(f"{base_path}/plots", exist_ok=True)
    os.makedirs(f"{base_path}/metrics", exist_ok=True)
    os.makedirs(f"{base_path}/models", exist_ok=True)

    # 1. Prepare data
    print("Preparing time-series data...")
    if 'date' in df.columns:
        df = df.sort_values('date')
    y = df[target_col].copy()
    X = prepare_features(df, target_col)

    # 2. Time-series split
    tscv = TimeSeriesSplit(n_splits=5)
    print(f"Using {tscv.n_splits} time-series splits")

    # 3. Feature processing
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    for col in categorical_features:
        X[col] = X[col].astype(str)
        if any(str(x).isdigit() for x in X[col] if not pd.isna(x)):
            X[col] = X[col].apply(lambda x: f"num_{x}" if str(x).isdigit() else x)

    # 4. Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )

   # 5. Parameter selection logic - DEFAULT PARAMETERS VERSION
    classifier_params = {}
    search = None
    # Apply model-specific defaults
    if not (use_bayesian or use_grid_search or use_random_search):
        classifier_params = DEFAULT_PARAMS.get(model_type, {}).copy()
        
        # Apply any user overrides
        if isinstance(load_params, dict):  # Allow dict of parameter overrides
            classifier_params.update(load_params)
        
        print(f"\nUsing default parameters for {model_type}:")
        for k, v in classifier_params.items():
            print(f"{k}: {v}")

    # 6. Create base pipeline with feature selection
    feature_selector = get_feature_selector(
        feature_selection_method=feature_selection_method,
        model_type=model_type,
        top_n_features=top_n_features,
        tscv=tscv,
    )
    base_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('variance_threshold', VarianceThreshold(threshold=0.01)),
        ('feature_selection', feature_selector)
    ])



    # 7. Configure search based on selected method
    if use_bayesian:
        print(f"\nUsing Bayesian optimization for {model_type} parameter tuning...")
        search = BayesSearchCV(
            estimator=Pipeline([
                *base_pipeline.steps,
                ('classifier', get_model_class(model_type))
            ]),
            search_spaces=MODEL_PARAMS[model_type]['bayesian'],
            n_iter=bayesian_iter,
            cv=tscv,
            scoring='accuracy',
            n_jobs=-1,
            random_state=42
        )
        
    elif use_grid_search:
        print(f"\nUsing grid search for {model_type} parameter tuning...")
        search = GridSearchCV(
            estimator=Pipeline([
                *base_pipeline.steps,
                ('classifier', get_model_class(model_type))
            ]),
            param_grid=MODEL_PARAMS[model_type]['grid'],
            cv=tscv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=2
        )
        
    elif use_random_search:  # NEW: Randomized Search implementation
        print(f"\nUsing randomized search for {model_type} parameter tuning...")
        search = RandomizedSearchCV(
            estimator=Pipeline([
                *base_pipeline.steps,
                ('classifier', get_model_class(model_type))
            ]),
            param_distributions=MODEL_PARAMS[model_type]['grid'],  # Reuse grid space
            n_iter=random_search_iter,
            cv=tscv,
            scoring='accuracy',
            n_jobs=-1,
            random_state=42,
            verbose=2
        )
    else:
        print("\nUsing pre-loaded or default parameters...")
        model = Pipeline([
            *base_pipeline.steps,
            ('classifier', get_model_class(model_type).set_params(**classifier_params))
        ])


    # 8. Training logic - OPTIMIZED VERSION
    cv_metrics = []
    feature_importances = []
    class_names = ['Draw (0)', 'Home Win (1)', 'Away Win (2)']
    
    def evaluate_model(X_train, X_test, y_train, y_test, fold=None):
        """Helper function for standardized model evaluation"""
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision_macro': precision_score(y_test, y_pred, average='macro'),
            'recall_macro': recall_score(y_test, y_pred, average='macro'),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
        }
        
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
            metrics.update({
                'roc_auc_ovr': roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro'),
                'log_loss': log_loss(y_test, y_proba)
            })
        
        if fold is not None:
            metrics['fold'] = fold + 1
            print(f"Fold {fold + 1} - Accuracy: {metrics['accuracy']:.4f}")
        
        return metrics

    if search is not None:
        # Parameter search mode with optimized evaluation
        print(f"\nStarting {'Bayesian' if use_bayesian else 'Grid'} search...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            search.fit(X, y)
            
        print(f"Best parameters: {search.best_params_}")
        model = search.best_estimator_
        
        # Save search results with proper path organization
        os.makedirs(f'artifacts/{model_type}/{feature_selection_method}/metrics', exist_ok=True)
        pd.DataFrame(search.cv_results_).to_csv(
            f'artifacts/{model_type}/{feature_selection_method}/metrics/search_results.csv', 
            index=False
        )
        pd.DataFrame([search.best_params_]).to_csv(
            f'artifacts/{model_type}/{feature_selection_method}/metrics/best_params.csv', 
            index=False
        )
        
        # Evaluate best model on all folds
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            cv_metrics.append(evaluate_model(
                X.iloc[train_idx], X.iloc[test_idx],
                y.iloc[train_idx], y.iloc[test_idx],
                fold
            ))
    else:
        # Standard training mode with optimized feature processing
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            fold_metrics = evaluate_model(
                X.iloc[train_idx], X.iloc[test_idx],
                y.iloc[train_idx], y.iloc[test_idx],
                fold
            )
            cv_metrics.append(fold_metrics)
            
            # Feature processing only on first fold to save time
            if fold == 0:
                try:
                    feature_names = get_feature_names_from_pipeline(model, numerical_features, categorical_features)
                    selector = model.named_steps['feature_selection']
                    
                    # Handle PCA case
                    if hasattr(selector, 'components_'):
                        pca_loadings = pd.DataFrame(
                            selector.components_,
                            columns=feature_names,
                            index=[f'PC_{i+1}' for i in range(selector.n_components_)]
                        )
                        pca_loadings.to_csv(
                            f'artifacts/{model_type}/{feature_selection_method}/metrics/pca_loadings.csv'
                        )
                        selected_features = pca_loadings.index.tolist()
                    else:
                        # Handle other feature selection methods
                        selected_features = (
                            feature_names if not hasattr(selector, 'get_support')
                            else [f for f, m in zip(feature_names, selector.get_support()) if m]
                        )
                    
                    # Save selected features
                    pd.DataFrame({'feature': selected_features}).to_csv(
                        f'artifacts/{model_type}/{feature_selection_method}/metrics/selected_features.csv',
                        index=False
                    )
                    
                    # Save feature importances if available
                    if hasattr(model.named_steps['classifier'], 'feature_importances_'):
                        importances = model.named_steps['classifier'].feature_importances_
                        if len(importances) == len(selected_features):
                            feature_importances = list(zip(selected_features, importances))
                            pd.DataFrame(
                                feature_importances, 
                                columns=['feature', 'importance']
                            ).to_csv(
                                f'artifacts/{model_type}/{feature_selection_method}/metrics/feature_importances.csv',
                                index=False
                            )
                            
                except Exception as e:
                    print(f"Feature processing warning: {str(e)}")
                    cv_metrics[-1]['feature_processing_error'] = str(e)
                    
    # 9. Final training and evaluation
    print("\nTraining final model on full dataset...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)
    
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
    
    final_metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision_macro': precision_score(y, y_pred, average='macro'),
        'recall_macro': recall_score(y, y_pred, average='macro'),
        'f1_macro': f1_score(y, y_pred, average='macro'),
        'class_report': classification_report(y, y_pred, target_names=class_names, output_dict=True),
        'parameter_source': 'bayesian' if use_bayesian else 'grid' if use_grid_search else 'file',
        'model_type': model_type,
        'feature_selection_method': feature_selection_method
    }
    
    if y_proba is not None:
        final_metrics.update({
            'roc_auc_ovr': roc_auc_score(y, y_proba, multi_class='ovr', average='macro'),
            'log_loss': log_loss(y, y_proba),
            'probability_metrics': {
                'mean_prob_class_0': np.mean(y_proba[:, 0]),
                'mean_prob_class_1': np.mean(y_proba[:, 1]),
                'mean_prob_class_2': np.mean(y_proba[:, 2]),
                'confidence': np.mean(np.max(y_proba, axis=1))
            }
        })

    # 10. Save artifacts
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
        search_results=search if search is not None else None,
        is_bayesian=use_bayesian
    )
    print_classification_report(model_type, feature_selection_method)
    
    return model




def plot_rfecv_results(rfecv, path):
    """Save RFECV performance plot"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), 
             rfecv.cv_results_['mean_test_score'])
    plt.xlabel("Number of features selected")
    plt.ylabel("Mean CV accuracy")
    plt.title("RFECV Performance")
    plt.savefig(f"{path}/rfecv_performance.png")
    plt.close()

def save_feature_analysis(model, base_path, feature_names, numerical_features, categorical_features):
    """Save comprehensive feature analysis"""
    selector = model.named_steps['feature_selection']
    
    # Handle different selector types
    if hasattr(selector, 'cv_results_'):  # RFECV
        pd.DataFrame(selector.cv_results_).to_csv(f'{base_path}/metrics/rfecv_results.csv')
        selected_features = feature_names[selector.support_]
    elif hasattr(selector, 'components_'):  # PCA
        pca_loadings = pd.DataFrame(
            selector.components_,
            columns=feature_names,
            index=[f'PC_{i+1}' for i in range(selector.n_components_)]
        )
        pca_loadings.to_csv(f'{base_path}/metrics/pca_loadings.csv')
        selected_features = pca_loadings.index.tolist()
    else:  # Other selectors
        selected_features = feature_names[selector.get_support()] if hasattr(selector, 'get_support') else feature_names
    
    # Save selected features
    pd.DataFrame({'feature': selected_features}).to_csv(
        f'{base_path}/metrics/selected_features.csv', index=False)
    
    # Save feature importances if available
    if hasattr(model.named_steps['classifier'], 'feature_importances_'):
        importances = model.named_steps['classifier'].feature_importances_
        if len(importances) == len(selected_features):
            pd.DataFrame({
                'feature': selected_features,
                'importance': importances
            }).sort_values('importance', ascending=False).to_csv(
                f'{base_path}/metrics/feature_importances.csv', index=False)


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

def get_artifact_paths(model_type, feature_selection_method):
    """Generate paths for artifacts based on model and feature selection method"""
    base_path = f"artifacts/{model_type}/{feature_selection_method}"
    paths = {
        'base': base_path,
        'plots': f"{base_path}/plots",
        'metrics': f"{base_path}/metrics",
        'models': f"{base_path}/models"
    }
    # Create directories if they don't exist
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    return paths

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
    """Save all model artifacts with proper test set evaluation"""
    model_type = final_metrics.get('model_type', 'unknown')
    feature_selection_method = final_metrics.get('feature_selection_method', 'unknown')
    paths = get_artifact_paths(model_type, feature_selection_method)

    # 1. Ensure proper test set evaluation using the last time-split fold
    tscv = TimeSeriesSplit(n_splits=5)
    splits = list(tscv.split(X))
    train_idx, test_idx = splits[-1]  # Use last fold for test
    
    # Fit model on training data only
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    
    # Generate predictions on test data only
    y_test = y.iloc[test_idx]
    y_pred = model.predict(X.iloc[test_idx])
    
    # Generate proper classification report
    test_report = classification_report(
        y_test, y_pred, 
        target_names=class_names,
        output_dict=True
    )
    
    # Update final metrics with test set evaluation
    final_metrics.update({
        'test_accuracy': accuracy_score(y_test, y_pred),
        'test_precision_macro': precision_score(y_test, y_pred, average='macro'),
        'test_recall_macro': recall_score(y_test, y_pred, average='macro'),
        'test_f1_macro': f1_score(y_test, y_pred, average='macro'),
        'test_class_report': test_report,
        'evaluation_note': 'Final metrics computed on held-out test set'
    })

    # 2. Save the trained model
    model_path = f"{paths['models']}/{model_type}_model.pkl"
    joblib.dump(model, model_path)
    
    # 3. Save metrics to CSV
    pd.DataFrame(cv_metrics).to_csv(f"{paths['metrics']}/cross_validation_results.csv", index=False)
    
    # Prepare final metrics (now including test set metrics)
    final_metrics_flat = {
        'cv_accuracy': [np.mean([m['accuracy'] for m in cv_metrics])],
        'cv_f1_macro': [np.mean([m['f1_macro'] for m in cv_metrics])],
        'test_accuracy': [final_metrics['test_accuracy']],
        'test_f1_macro': [final_metrics['test_f1_macro']],
        'model_type': [model_type],
        'feature_selection_method': [feature_selection_method]
    }
    
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X.iloc[test_idx])
        final_metrics_flat.update({
            'test_roc_auc_ovr': [roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')],
            'test_log_loss': [log_loss(y_test, y_proba)]
        })
    
    pd.DataFrame(final_metrics_flat).to_csv(f"{paths['metrics']}/final_metrics.csv", index=False)
    
    # 4. Save test set classification report
    pd.DataFrame(test_report).transpose().to_csv(
        f"{paths['metrics']}/test_classification_report.csv",
        index=True
    )
    
    # 5. Save feature importances if available
    if feature_importances:
        pd.DataFrame(feature_importances, columns=['feature', 'importance'])\
            .to_csv(f"{paths['metrics']}/feature_importances.csv", index=False)
    
    # 6. Save preprocessing info
    pd.DataFrame({
        'feature_type': ['numerical']*len(numerical_features) + ['categorical']*len(categorical_features),
        'feature_name': numerical_features + categorical_features
    }).to_csv(f"{paths['metrics']}/preprocessing_info.csv", index=False)
    
    # 7. Save best parameters if available
    if search_results is not None:
        pd.DataFrame.from_dict(search_results.best_params_, orient='index', columns=['value'])\
            .to_csv(f"{paths['metrics']}/best_params.csv")
    
    # 8. Generate visualizations using test data
    generate_visualizations(
        model=model,
        X=X.iloc[test_idx],
        y=y.iloc[test_idx],
        class_names=class_names,
        feature_imp_df=pd.DataFrame(feature_importances, columns=['feature', 'importance']) if feature_importances else None,
        cv_metrics=cv_metrics,
        search_results=search_results,
        is_bayesian=is_bayesian,
        model_type=model_type,
        feature_selection_method=feature_selection_method,
        artifact_paths=paths
    )
    
    print("\nSaved artifacts:")
    print(f"- Model: {model_path}")
    print(f"- Cross-val metrics: {paths['metrics']}/cross_validation_results.csv")
    print(f"- Test set metrics: {paths['metrics']}/test_classification_report.csv")
    print(f"- Plots saved in: {paths['plots']}")

def plot_rfecv_results(selector, feature_names=None):
    """Plot RFECV performance vs number of features"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(selector.cv_results_['mean_test_score']) + 1), 
             selector.cv_results_['mean_test_score'])
    plt.fill_between(
        range(1, len(selector.cv_results_['mean_test_score']) + 1),
        selector.cv_results_['mean_test_score'] - selector.cv_results_['std_test_score'],
        selector.cv_results_['mean_test_score'] + selector.cv_results_['std_test_score'],
        alpha=0.1
    )
    plt.xlabel("Number of features selected")
    plt.ylabel("Mean test score (F1 macro)")
    plt.title("RFECV Performance")
    if feature_names:
        plt.axvline(x=len(selector.support_), color='r', linestyle='--')
    plt.show()

def generate_visualizations(
    model,
    X,
    y,
    class_names,
    feature_imp_df,
    cv_metrics,
    search_results,
    is_bayesian,
    model_type,
    feature_selection_method,
    artifact_paths
):
    """Generate visualizations and save to model/feature-selection specific folders"""
    # 1. Main evaluation plot
    plt.figure(figsize=(20, 15))
    plt.suptitle(f'{model_type.title()} ({feature_selection_method}) Evaluation', y=1.02, fontsize=16)
    
    # Feature Importance Plot
    if feature_imp_df is not None and not feature_imp_df.empty:
        plt.subplot(2, 2, 1)
        top_features = feature_imp_df.nlargest(15, 'importance').sort_values('importance', ascending=True)
        sns.barplot(x='importance', y='feature', data=top_features, palette='viridis')
        plt.title('Top 15 Feature Importances')
    
    # CV Metrics Trend
    plt.subplot(2, 2, 2)
    cv_df = pd.DataFrame(cv_metrics)
    metrics_to_plot = ['accuracy', 'f1_macro']
    if 'roc_auc_ovr' in cv_df.columns:
        metrics_to_plot.append('roc_auc_ovr')
    for metric in metrics_to_plot:
        plt.plot(cv_df['fold'], cv_df[metric], label=metric.replace('_', ' ').title())
    plt.title('Cross-Validation Performance')
    plt.legend()
    
    # Hyperparameter Tuning Results
    plt.subplot(2, 2, 3)
    if search_results is not None:
        if is_bayesian:
            plot_convergence(search_results.optimizer_results_[0] if hasattr(search_results, 'optimizer_results_') 
                           else search_results)
            plt.title('Bayesian Optimization Convergence')
        else:
            results = pd.DataFrame(search_results.cv_results_)
            param_cols = [col for col in results.columns if col.startswith('param_')]
            if len(param_cols) >= 2:
                heatmap_data = results.pivot_table(
                    index=param_cols[0],
                    columns=param_cols[1],
                    values='mean_test_score'
                )
                sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlGnBu")
                plt.title('Grid Search Results')
    
    # Confusion Matrix
    plt.subplot(2, 2, 4)
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig(f"{artifact_paths['plots']}/model_evaluation.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    # Additional probability plots if available
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X)
        
        # Calibration curve
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        for i, class_name in enumerate(class_names):
            fraction_of_positives, mean_predicted_value = calibration_curve(
                (y == i).astype(int), y_proba[:, i], n_bins=10
            )
            plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=class_name)
        plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
        plt.title('Calibration Plot')
        plt.legend()
        
        # Probability distribution
        plt.subplot(1, 3, 2)
        for i, class_name in enumerate(class_names):
            sns.kdeplot(y_proba[:, i], label=class_name, fill=True)
        plt.title('Probability Distribution')
        plt.legend()
        
        # ROC curves
        plt.subplot(1, 3, 3)
        for i, class_name in enumerate(class_names):
            fpr, tpr, _ = roc_curve((y == i).astype(int), y_proba[:, i])
            roc_auc = roc_auc_score((y == i).astype(int), y_proba[:, i])
            plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title('ROC Curves')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{artifact_paths['plots']}/probability_plots.png", bbox_inches='tight', dpi=300)
        plt.close()

def plot_calibration_curve(model, X, y, class_names):
    """Plot calibration curves for each class"""
    y_proba = model.predict_proba(X)
    
    plt.figure(figsize=(8, 6))
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
    
    plt.figure(figsize=(8, 6))
    for i, class_name in enumerate(class_names):
        sns.kdeplot(y_proba[:, i], label=class_name, fill=True)
    
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.title('Probability Distribution')
    plt.legend()
    plt.grid()

def plot_multiclass_roc(model, X, y, class_names):
    """Plot ROC curves for each class (OvR)"""
    y_proba = model.predict_proba(X)
    
    plt.figure(figsize=(8, 6))
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
    """Get all feature names after preprocessing, handling different feature selection methods"""
    try:
        # Get feature names after preprocessor
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
        
        # Handle variance threshold if present
        if 'variance_threshold' in model.named_steps:
            vt = model.named_steps['variance_threshold']
            if hasattr(vt, 'get_support'):
                vt_mask = vt.get_support()
                feature_names = [f for f, m in zip(feature_names, vt_mask) if m]
        
        return feature_names
    except Exception as e:
        print(f"Error getting feature names: {str(e)}")
        return []

def get_selected_features(model, all_feature_names):
    """Get names of selected features based on feature selection method"""
    try:
        selector = model.named_steps['feature_selection']
        
        if hasattr(selector, 'get_support'):
            # For SelectFromModel, RFE, SelectKBest
            selected_indices = selector.get_support()
            return [all_feature_names[i] for i in range(len(selected_indices)) if selected_indices[i]]
        elif hasattr(selector, 'components_'):
            # For PCA
            return [f'PC_{i+1}' for i in range(selector.n_components_)]
        else:
            # No feature selection applied
            return all_feature_names
    except Exception as e:
        print(f"Error getting selected features: {str(e)}")
        return []

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
        'fixture_id', 'league_id', 'date', 'home_team', 'away_team', 'season',
        'home_team_id', 'away_team_id', 'referee', target_col, 'venue_id', 'round',
        'extratime_home', 'extratime_away', 'home_winner', 'away_winner',
        'season', 'penalty_home', 'penalty_away', 'halftime_home', 'halftime_away',
        'fulltime_home', 'fulltime_away', 'home_goals', 'away_goals', 'total_goals', 
        'goal_difference', 'year', 'goals_home', 'goals_away', 'league',
        'league_name', 'league_flag', 'league_logo', 'venue_name', 'venue_city',
        'status', 'home_team_flag', 'away_team_flag', 'home_team_name', 'away_team_name',
        'home_coach_id', 'away_coach_id', 'home_player_id', 'away_player_id', 'timestamp',
        'maintime', 'first_half', 'second_half', 'country', 'extratime'
        
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

def print_classification_report(model_type, feature_selection_method):
    """
    Print comprehensive classification report including:
    - Cross-validation metrics
    - Test set performance
    - Proper handling of train/test separation
    
    Args:
        model_type: Type of model ('random_forest', 'logistic_regression', etc.)
        feature_selection_method: Feature selection method ('importance', 'pca', etc.)
    """
    base_path = f"artifacts/{model_type}/{feature_selection_method}/metrics"
    
    try:
        # Load all relevant files
        cv_results = pd.read_csv(f"{base_path}/cross_validation_results.csv")
        test_report = pd.read_csv(f"{base_path}/test_classification_report.csv", index_col=0)
        final_metrics = pd.read_csv(f"{base_path}/final_metrics.csv")
        
        # Print header with model info
        print(f"\nModel Performance: {model_type.upper()} ({feature_selection_method})")
        print("="*80)
        
        # Cross-validation summary
        print("\nCross-Validation Performance:")
        print("-"*80)
        print(f"Average Accuracy: {cv_results['accuracy'].mean():.2%} (±{cv_results['accuracy'].std():.2%})")
        print(f"Average F1 Score: {cv_results['f1_macro'].mean():.2%} (±{cv_results['f1_macro'].std():.2%})")
        print(f"Average ROC AUC: {cv_results['roc_auc_ovr'].mean():.2%} (±{cv_results['roc_auc_ovr'].std():.2%})")
        
        # Test set performance
        print("\nTest Set Evaluation:")
        print("-"*80)
        print(f"Accuracy: {final_metrics.iloc[0]['test_accuracy']:.2%}")
        if 'test_roc_auc_ovr' in final_metrics:
            print(f"ROC AUC: {final_metrics.iloc[0]['test_roc_auc_ovr']:.2%}")
        
        # Detailed classification report
        print("\nTest Set Classification Report:")
        print("-"*80)
        print(f"{'Class':<15}{'Precision':>10}{'Recall':>10}{'F1-Score':>10}{'Support':>10}")
        print("-"*80)
        
        # Print class-wise metrics
        for class_name in test_report.index:
            if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                row = test_report.loc[class_name]
                print(f"{class_name:<15}{row['precision']:>10.2f}{row['recall']:>10.2f}{row['f1-score']:>10.2f}{int(row['support']):>10}")
        
        # Print averages
        print("-"*80)
        for avg_type in ['macro avg', 'weighted avg']:
            if avg_type in test_report.index:
                row = test_report.loc[avg_type]
                print(f"{avg_type:<15}{row['precision']:>10.2f}{row['recall']:>10.2f}{row['f1-score']:>10.2f}{int(row['support']):>10}")
        
        print("="*80)
        
    except FileNotFoundError as e:
        print(f"\nError: Required files not found in {base_path}")
        print(f"Missing file: {str(e)}")
    except Exception as e:
        print(f"\nError generating report: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        # Load data
        df = pd.read_csv('data/final_processed_standings_h2h_rolling.csv')
        
        # Example usage:
        # Random Forest with Bayesian optimization and PCA feature selection
        trained_model = train_and_save_model(
            df, 
            model_type='random_forest',
            feature_selection_method='rfe',
            load_params=True,
            use_bayesian=False,
            use_grid_search=False,
            use_random_search=False,
            top_n_features=30,
            
 
           
        )
        print("\nTraining and evaluation completed successfully!")
        #print_classification_report('random_forest', 'importance')
        """""
        # Logistic Regression with grid search and ANOVA feature selection
        trained_model = train_and_save_model(
            df,
            model_type='logistic_regression',
            feature_selection_method='anova',
            top_n_features=15,
            use_grid_search=True
        )
        
        # 1. Random Forest with Bayesian optimization and PCA
        trained_model = train_and_save_model(
            df,
            model_type='random_forest',
            feature_selection_method='pca',
            top_n_features=20,
            use_bayesian=True,
            bayesian_iter=30
        )

        # 2. Logistic Regression with ANOVA feature selection
        trained_model = train_and_save_model(
            df,
            model_type='logistic_regression',
            feature_selection_method='anova',
            top_n_features=15,
            use_grid_search=True
        )

        # 3. XGBoost with RFE and loaded parameters
        trained_model = train_and_save_model(
            df,
            model_type='xgboost',
            feature_selection_method='rfe',
            top_n_features=25,
            load_params=True
        )

        # 4. Poisson regression with feature importance
        trained_model = train_and_save_model(
            df,
            model_type='poisson',
            feature_selection_method='importance',
            top_n_features=30
        )
        """

        
    except Exception as e:
        print(f"\nTraining failed: {str(e)}")
        raise


