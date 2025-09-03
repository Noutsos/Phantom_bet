# Core & Data Handling
import pandas as pd
import numpy as np
import os
import warnings
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Dict, List, Union, Optional, Tuple

# Models
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier)
from sklearn.linear_model import (LogisticRegression, PoissonRegressor)
from xgboost import XGBClassifier
from sklearn.svm import SVC
import xgboost as xgb

# Feature Selection
from sklearn.feature_selection import (
    mutual_info_classif, SelectFromModel, RFE, RFECV,
    SelectKBest, f_classif, VarianceThreshold, SequentialFeatureSelector
)
from sklearn.decomposition import PCA
# Preprocessing
from sklearn.preprocessing import (OneHotEncoder, StandardScaler, LabelEncoder, label_binarize)
from sklearn.compose import ColumnTransformer


# Model Selection & Evaluation
from sklearn.model_selection import (
    TimeSeriesSplit, RandomizedSearchCV, GridSearchCV, learning_curve
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    balanced_accuracy_score, average_precision_score, auc,
    precision_recall_curve, brier_score_loss, log_loss, roc_curve, make_scorer
)

# Calibration & Visualization
from sklearn.calibration import (CalibratedClassifierCV, calibration_curve)
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('default')  # Set a nice style
sns.set_palette("viridis")  # Set a nice color palette

# Bayesian Optimization
from skopt import BayesSearchCV
from skopt.space import (Integer, Real, Categorical)
from skopt.plots import plot_convergence

# Misc
from sklearn.exceptions import NotFittedError
from sklearn.base import clone
import warnings
warnings.filterwarnings('ignore')

# Configure XGBoost
xgb.set_config(verbosity=0)

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
    },
        'gradient_boosting': {
        'bayesian': {
            'classifier__n_estimators': Integer(50, 500),
            'classifier__learning_rate': Real(0.01, 0.2, prior='log-uniform'),
            'classifier__max_depth': Integer(3, 10),
            'classifier__min_samples_split': Integer(2, 20),
            'classifier__min_samples_leaf': Integer(1, 10),
            'classifier__max_features': Categorical(['sqrt', 'log2', None])
        },
        'grid': {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__learning_rate': [0.01, 0.1, 0.2],
            'classifier__max_depth': [3, 5, 7],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4],
            'classifier__max_features': ['sqrt', 'log2', None]
        }
    },
    'svm': {
        'bayesian': {
            'classifier__C': Real(1e-4, 1e4, prior='log-uniform'),
            'classifier__kernel': Categorical(['linear', 'rbf', 'poly']),
            'classifier__gamma': Real(1e-4, 1e4, prior='log-uniform'),
            'classifier__degree': Integer(2, 5)  # Only used for poly kernel
        },
        'grid': {
            'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'classifier__kernel': ['linear', 'rbf'],
            'classifier__gamma': ['scale', 'auto'] + [0.001, 0.01, 0.1, 1]
        }
    }
}

# Define default parameter sets for each model type
DEFAULT_PARAMS = {
    'random_forest': {
        'n_estimators': 300,
        'max_depth': None,  # Unlimited depth
        'min_samples_split': 5,
        'min_samples_leaf': 1,
        'max_features': 'sqrt',
        'bootstrap': False,
        'class_weight': 'balanced',
        'criterion': 'entropy',  # Default criterion
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
        'max_depth': 3,
        'learning_rate': 0.1,
        'subsample': 1.0,
        'colsample_bytree': 1.0,
        'gamma': 1.0,
        'reg_alpha': 0.1,
        'reg_lambda': 1,
        'objective': 'multi:softprob',
        'n_jobs': -1,
        'tree_method': 'hist'  # Faster histogram method
    },
    'poisson': {
        'alpha': 1.0,
        'max_iter': 1000,
        'tol': 1e-4
    },
    'gradient_boosting': {
        'n_estimators': 200,
        'learning_rate': 0.1,
        'max_depth': 3,
        'min_samples_split': 5,
        'min_samples_leaf': 1,
        'max_features': 'sqrt',
        'random_state': 42
    },
    'svm': {
        'C': 1.0,
        'kernel': 'rbf',
        'gamma': 'scale',
        'probability': True,
        'random_state': 42,
        'class_weight': 'balanced'
    }
}

def get_model_class(model_type):
    """Return the appropriate model class based on model_type with enhanced defaults"""
    model_classes = {
        'random_forest': RandomForestClassifier(
            random_state=42, 
            class_weight='balanced',
            n_jobs=-1
        ),
        'logistic_regression': LogisticRegression(
            random_state=42,
            class_weight='balanced',
            multi_class='ovr',
            max_iter=1000,
            solver='lbfgs'
        ),
        'xgboost': XGBClassifier(
            random_state=42,
            eval_metric='mlogloss',
            objective='multi:softprob',
            n_jobs=-1,
            tree_method='hist'
 
        ),
        'poisson': PoissonRegressor(
            max_iter=500,
            alpha=1.0
        ),
        'gradient_boosting': GradientBoostingClassifier(
            random_state=42
        ),
        'svm': SVC(
            random_state=42,
            probability=True,
            class_weight='balanced',
            kernel='linear',
        )
    }
    
    if model_type not in model_classes:
        raise ValueError(f"Unknown model type: {model_type}. Available options: {list(model_classes.keys())}")
    
    return model_classes[model_type]

def get_feature_selector(feature_selection_method, model_type, top_n_features, tscv, target_variance=0.9):
    """Return feature selector with additional methods and better defaults"""
    selectors = {
        'importance': SelectFromModel(
            RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=50),
            max_features=top_n_features,
            threshold=-np.inf
        ),
        'rfe': RFE(
            estimator=get_model_class(model_type),
            n_features_to_select=top_n_features,
            step=0.1,
            verbose=0
        ),
        'rfecv': RFECV(
            estimator=get_model_class(model_type),
            step=1,
            cv=tscv,
            scoring='accuracy',
            min_features_to_select=top_n_features,
            n_jobs=-1
        ),
        'pca': PCA(
            n_components=target_variance, 
            svd_solver='full',
            random_state=42
        ),
        'anova': SelectKBest(
            score_func=f_classif, 
            k=top_n_features
        ),
        'mutual_info': SelectKBest(
            score_func=mutual_info_classif,
            k=top_n_features
        ),
        'sequential': SequentialFeatureSelector(
            get_model_class(model_type),
            n_features_to_select=top_n_features,
            direction='forward',
            cv=tscv,
            n_jobs=-1
        )
    }
    
    if feature_selection_method not in selectors:
        raise ValueError(f"Unknown method: {feature_selection_method}. Available options: {list(selectors.keys())}")
    
    return selectors[feature_selection_method]


def setup_directories(model_type, feature_selection_method):
    """Create directory structure for artifacts"""
    paths = {
        'base': f"artifacts/{model_type}/{feature_selection_method}",
        'plots': f"artifacts/{model_type}/{feature_selection_method}/plots",
        'metrics': f"artifacts/{model_type}/{feature_selection_method}/metrics",
        'models': f"artifacts/{model_type}/{feature_selection_method}/models"
    }
    
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    
    return paths

def prepare_training_data(df, target_col):
    """Prepare features and target with proper validation"""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")
    
    df = df[df['status'].isin(['FT', 'AET', 'PEN'])]  # Filter completed matches
    df = df.sort_values('date') if 'date' in df.columns else df
    
    y = df[target_col].copy()
    le = LabelEncoder()
    y_encoded = pd.Series(le.fit_transform(y), index=y.index)
    
    print(f"Encoded classes: {dict(zip(le.classes_, range(len(le.classes_))))}")
    X = prepare_features(df, target_col)
    
    return X, y_encoded, le

def initialize_search(model_type, base_pipeline, use_bayesian, use_grid_search, 
                    use_random_search, tscv, bayesian_iter=50, random_search_iter=50):
    """Initialize the appropriate parameter search strategy"""
    if not any([use_bayesian, use_grid_search, use_random_search]):
        return None
    
    model_class = get_model_class(model_type)
    full_pipeline = Pipeline([
        *base_pipeline.steps,
        ('classifier', model_class)
    ])
    
    if use_bayesian:
        return BayesSearchCV(
            estimator=full_pipeline,
            search_spaces=MODEL_PARAMS[model_type]['bayesian'],
            n_iter=bayesian_iter,
            cv=tscv,
            scoring='accuracy',
            n_jobs=-1,
            random_state=42
        )
    elif use_grid_search:
        return GridSearchCV(
            estimator=full_pipeline,
            param_grid=MODEL_PARAMS[model_type]['grid'],
            cv=tscv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=2
        )
    else:  # random_search
        return RandomizedSearchCV(
            estimator=full_pipeline,
            param_distributions=MODEL_PARAMS[model_type]['grid'],
            n_iter=random_search_iter,
            cv=tscv,
            scoring='accuracy',
            n_jobs=-1,
            random_state=42,
            verbose=2
        )

def run_cross_validation(model, X, y, tscv, model_type, feature_selection_method):
    cv_metrics = []
    feature_importances = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        print(f"\nFold {fold + 1} Feature Selection:")
        print(f"- Initial features: {X.shape[1]}")
        
        fold_metrics = evaluate_model(
            model, 
            X.iloc[train_idx], X.iloc[test_idx],
            y.iloc[train_idx], y.iloc[test_idx],
            fold,
            class_names=['away_win', 'draw', 'home_win'],
            model_type=model_type,
            feature_selection_method=feature_selection_method
        )
        
        # Get selected features count
        try:
            selector = model.named_steps['feature_selection']
            if hasattr(selector, 'n_features_in_'):
                print(f"- Selected features: {selector.n_features_in_}")
        except Exception as e:
            print(f"- Could not determine selected features: {str(e)}")
        
        cv_metrics.append(fold_metrics)
    
    return cv_metrics, feature_importances

def train_final_model(model, X, y, class_names):
    """Train and evaluate final model on full dataset"""
    model.fit(X, y)
    y_pred = model.predict(X)
    
    final_metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision_macro': precision_score(y, y_pred, average='macro'),
        'recall_macro': recall_score(y, y_pred, average='macro'),
        'f1_macro': f1_score(y, y_pred, average='macro'),
        'class_report': classification_report(y, y_pred, target_names=class_names, output_dict=True)
    }
    
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X)
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
    
    return final_metrics

def evaluate_model(model, X_train, X_test, y_train, y_test, fold=None, class_names=None, model_type=None, feature_selection_method=None):
    """
    Enhanced model evaluation with comprehensive metrics and visualizations
    
    Args:
        model: Trained sklearn model or pipeline
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        fold: Fold number (for CV)
        class_names: List of class names
        model_type: Type of model
        feature_selection_method: Feature selection method used
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Setup paths if model_type and feature_selection_method provided
    paths = None
    if model_type and feature_selection_method:
        paths = setup_directories(model_type, feature_selection_method)
    
    # Initialize metrics dictionary
    metrics = {
        'fold': fold + 1 if fold is not None else None,
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test)
    }
    
    # Train and predict
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    
    # Basic classification metrics
    metrics.update({
        'train_accuracy': accuracy_score(y_train, y_pred_train),
        'test_accuracy': accuracy_score(y_test, y_pred),
        'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
    })
    
    # Class-specific metrics
    if class_names:
        class_report = classification_report(
            y_test, y_pred, 
            target_names=class_names, 
            output_dict=True,
            zero_division=0
        )
        metrics['class_report'] = class_report
        
        # Store per-class metrics
        for i, class_name in enumerate(class_names):
            if str(i) in class_report:  # Only if class exists in test set
                metrics.update({
                    f'precision_{class_name}': class_report[str(i)]['precision'],
                    f'recall_{class_name}': class_report[str(i)]['recall'],
                    f'f1_{class_name}': class_report[str(i)]['f1-score'],
                    f'support_{class_name}': class_report[str(i)]['support']
                })
    
    # Probability-based metrics
    if hasattr(model, 'predict_proba'):
        try:
            y_proba = model.predict_proba(X_test)
            n_classes = y_proba.shape[1]
            
            # Multi-class metrics
            if n_classes > 2:
                metrics.update({
                    'roc_auc_ovr': roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro'),
                    'roc_auc_ovo': roc_auc_score(y_test, y_proba, multi_class='ovo', average='macro'),
                    'log_loss': log_loss(y_test, y_proba),
                    'brier_score': multi_class_brier_score(y_test, y_proba)
                })
                
                # Class-specific AUC
                for i in range(n_classes):
                    if i in y_test.unique():  # Only if class exists
                        auc = roc_auc_score((y_test == i).astype(int), y_proba[:, i])
                        class_label = class_names[i] if class_names else str(i)
                        metrics[f'roc_auc_class_{class_label}'] = auc
            
            # Binary classification metrics
            elif n_classes == 2:
                metrics.update({
                    'roc_auc': roc_auc_score(y_test, y_proba[:, 1]),
                    'average_precision': average_precision_score(y_test, y_proba[:, 1]),
                    'log_loss': log_loss(y_test, y_proba),
                    'brier_score': brier_score_loss(y_test, y_proba[:, 1])
                })
                
                # Find optimal threshold
                precision, recall, thresholds = precision_recall_curve(y_test, y_proba[:, 1])
                f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
                optimal_idx = np.argmax(f1_scores)
                metrics['optimal_threshold'] = thresholds[optimal_idx]
                
        except Exception as e:
            metrics['probability_metrics_error'] = str(e)
    
    # Confusion matrix
    try:
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        if class_names and paths:
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names)
            plt.title(f"Confusion Matrix{' - Fold '+str(fold+1) if fold is not None else ''}")
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            #plt.savefig(f"{paths['plots']}/confusion_matrix{'_fold_'+str(fold+1) if fold is not None else ''}.png")
            plt.close()
    except Exception as e:
        metrics['confusion_matrix_error'] = str(e)
    
    # Calibration curve
    if hasattr(model, 'predict_proba') and class_names and paths:
        try:
            plt.figure(figsize=(10, 6))
            for i, class_name in enumerate(class_names):
                if i in y_test.unique():  # Only if class exists
                    prob_true, prob_pred = calibration_curve(
                        (y_test == i).ast(int), 
                        y_proba[:, i], 
                        n_bins=10, 
                        strategy='quantile'
                    )
                    plt.plot(prob_pred, prob_true, 's-', label=class_name)
            
            plt.plot([0, 1], [0, 1], 'k--', label="Perfectly calibrated")
            plt.xlabel('Mean predicted probability')
            plt.ylabel('Fraction of positives')
            plt.title('Calibration Plot')
            plt.legend()
            plt.savefig(f"{paths['plots']}/calibration{'_fold_'+str(fold+1) if fold is not None else ''}.png")
            plt.close()
        except Exception as e:
            metrics['calibration_error'] = str(e)
    
    # Feature importance tracking
    try:
        if hasattr(model.named_steps.get('classifier', model), 'feature_importances_'):
            importances = model.named_steps.get('classifier', model).feature_importances_
            metrics['feature_importances'] = importances.tolist()
            
            if paths and fold == 0:  # Only save for first fold
                feature_names = get_feature_names_from_pipeline(model)
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                importance_df.to_csv(f"{paths['metrics']}/feature_importances.csv", index=False)
                
                # Plot top features
                plt.figure(figsize=(10, 6))
                sns.barplot(x='importance', y='feature', 
                           data=importance_df.head(20))
                plt.title('Top 20 Feature Importances')
                plt.savefig(f"{paths['plots']}/top_features.png")
                plt.close()
    except Exception as e:
        metrics['feature_importance_error'] = str(e)
    
    if fold is not None:
        print(f"Fold {fold + 1} - Test Accuracy: {metrics['test_accuracy']:.4f} - F1 Macro: {metrics['f1_macro']:.4f}")
    
    return metrics


def train_and_save_model(df: pd.DataFrame, 
                       target_col: str = 'outcome', 
                       model_type: str = 'random_forest',
                       feature_selection_method: str = 'importance',
                       top_n_features: int = 30,
                       use_bayesian: bool = False, 
                       bayesian_iter: int = 50,
                       use_grid_search: bool = False,
                       use_random_search: bool = False,
                       random_search_iter: int = 50,
                       load_params: bool = False,
                       holdout_ratio: float = 0.2):  # Ratio for final holdout
    
    # 1. Setup and data preparation
    paths = setup_directories(model_type, feature_selection_method)
    X, y, le = prepare_training_data(df, target_col)
    joblib.dump(le, f"{paths['models']}/label_encoder.pkl")

    # 2. CREATE PROPER TIME SERIES HOLDOUT SET
    # Assuming your data is already sorted chronologically
    n_samples = len(X)
    holdout_size = int(n_samples * holdout_ratio)
    
    # Split into training (early data) and holdout (most recent data)
    X_train_full = X.iloc[:-holdout_size]
    y_train_full = y.iloc[:-holdout_size]
    X_holdout = X.iloc[-holdout_size:]
    y_holdout = y.iloc[-holdout_size:]
    
    print(f"Time Series Split:")
    print(f"- Training period: {X_train_full.shape[0]} samples")
    print(f"- Holdout period: {X_holdout.shape[0]} samples")
    print(f"- Holdout ratio: {holdout_ratio:.1%}")

    # 3. Initialize components using only training data
    tscv = TimeSeriesSplit(n_splits=5)
    numerical_features = X_train_full.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_train_full.select_dtypes(include=['object', 'category']).columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )
    
    feature_selector = get_feature_selector(
        feature_selection_method=feature_selection_method,
        model_type=model_type,
        top_n_features=top_n_features,
        tscv=tscv
    )
    
    base_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('variance_threshold', VarianceThreshold(threshold=0.01)),
        ('feature_selection', feature_selector)
    ])

    # 4. Training logic (use only training data!)
    if use_bayesian or use_grid_search or use_random_search:
        search = initialize_search(
            model_type=model_type,
            base_pipeline=base_pipeline,
            use_bayesian=use_bayesian,
            use_grid_search=use_grid_search,
            use_random_search=use_random_search,
            tscv=tscv,
            bayesian_iter=bayesian_iter,
            random_search_iter=random_search_iter
        )
        
        print(f"\nStarting parameter search on training data...")
        search.fit(X_train_full, y_train_full)  # Train on training period only!
        model = search.best_estimator_
        
        # Save search results
        pd.DataFrame(search.cv_results_).to_csv(f"{paths['metrics']}/search_results.csv", index=False)
        pd.DataFrame([search.best_params_]).to_csv(f"{paths['metrics']}/best_params.csv", index=False)
    else:
        classifier_params = DEFAULT_PARAMS.get(model_type, {}).copy()
        if isinstance(load_params, dict):
            classifier_params.update(load_params)
            
        model = Pipeline([
            *base_pipeline.steps,
            ('classifier', get_model_class(model_type).set_params(**classifier_params))
        ])

    # 5. Cross-validation (use only training data with TimeSeriesSplit!)
    class_names = ['away_win', 'draw', 'home_win']
    cv_metrics = []
    
    print(f"\nCross-validation on training period:")
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_train_full)):  # Use training data only!
        #print(f"\nFold {fold + 1}/{tscv.n_splits}")
        
        # Clone model for each fold to avoid contamination
        fold_model = clone(model)
        
        # Train and evaluate on training data splits
        metrics = evaluate_model(
            fold_model,
            X_train_full.iloc[train_idx], X_train_full.iloc[test_idx],  # Training period splits
            y_train_full.iloc[train_idx], y_train_full.iloc[test_idx],  # Training period splits
            fold=fold,
            class_names=class_names,
            model_type=model_type,
            feature_selection_method=feature_selection_method
        )
        cv_metrics.append(metrics)
        
        # Get selected features count
        try:
            selector = fold_model.named_steps['feature_selection']
            if hasattr(selector, 'n_features_in_'):
                print(f"- Selected features: {selector.n_features_in_}")
        except Exception as e:
            print(f"- Feature selection info not available: {str(e)}")

    # 6. Final training on full training period
    print("\nTraining final model on full training period...")
    model.fit(X_train_full, y_train_full)  # Train on full training period
    
    # 7. Evaluate on holdout period (most recent data)
    print("Evaluating on holdout period (most recent data)...")
    y_pred_holdout = model.predict(X_holdout)
    holdout_accuracy = accuracy_score(y_holdout, y_pred_holdout)
    
    # Generate comprehensive holdout metrics
    holdout_report = classification_report(
        y_holdout, y_pred_holdout, 
        target_names=class_names, 
        output_dict=True,
        zero_division=0
    )
    
    print(f"Holdout Period Performance:")
    print(f"- Accuracy: {holdout_accuracy:.4f}")
    print(f"- F1 Macro: {holdout_report['macro avg']['f1-score']:.4f}")

    # 8. Save artifacts with PROPER time series separation
    final_metrics = {
        'model_type': model_type,
        'feature_selection_method': feature_selection_method,
        'holdout_accuracy': holdout_accuracy,
        'holdout_precision_macro': holdout_report['macro avg']['precision'],
        'holdout_recall_macro': holdout_report['macro avg']['recall'],
        'holdout_f1_macro': holdout_report['macro avg']['f1-score'],
        'train_samples': len(X_train_full),
        'holdout_samples': len(X_holdout),
        'holdout_ratio': holdout_ratio
    }

    # Get feature importances if available
    feature_importances = []
    if hasattr(model.named_steps['classifier'], 'feature_importances_'):
        try:
            feature_names = get_feature_names_from_pipeline(model)
            importances = model.named_steps['classifier'].feature_importances_
            feature_importances = list(zip(feature_names, importances))
            
            # Save feature importances
            importance_df = pd.DataFrame(feature_importances, columns=['feature', 'importance'])
            importance_df = importance_df.sort_values('importance', ascending=False)
            importance_df.to_csv(f"{paths['metrics']}/feature_importances.csv", index=False)
            
        except Exception as e:
            print(f"Could not get feature importances: {str(e)}")

    # 9. Save artifacts with proper time series train/holdout separation
    save_all_artifacts(
        model=model,
        cv_metrics=cv_metrics,
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        class_names=class_names,
        X_train=X_train_full,  # Training period data
        y_train=y_train_full,  # Training period labels
        X_test=X_holdout,      # Holdout period data (most recent)
        y_test=y_holdout,      # Holdout period labels (most recent)
        final_metrics=final_metrics,
        feature_importances=feature_importances
    )
    
    print("\nTime Series Training completed successfully!")
    print(f"Final Model Performance:")
    print(f"- CV Average Accuracy: {np.mean([m['test_accuracy'] for m in cv_metrics]):.4f}")
    print(f"- Holdout Period Accuracy: {holdout_accuracy:.4f}")
    
    return model, X_holdout, y_holdout

def save_all_artifacts(
    model,
    cv_metrics,
    numerical_features,
    categorical_features,
    class_names,
    X_train,  # Explicit training data
    y_train,  # Explicit training labels  
    X_test,   # Explicit test data (HOLDOUT)
    y_test,   # Explicit test labels (HOLDOUT)
    final_metrics=None,
    feature_importances=None,
    search_results=None,
    is_bayesian=False
):
    """
    Enhanced artifact saving with proper test/train separation.
    Uses explicit holdout test set for final evaluation.
    """
    # Set defaults
    if final_metrics is None:
        final_metrics = {}
    if feature_importances is None:
        feature_importances = []
        
    try:
        # 1. Get model type and feature selection method
        model_type = final_metrics.get('model_type', 'unknown')
        feature_selection_method = final_metrics.get('feature_selection_method', 'unknown')
        paths = get_artifact_paths(model_type, feature_selection_method)

        # 2. Train on FULL training set (X_train, not the entire dataset)
        print("Training final model on full training set...")
        model_clone = clone(model)
        model_clone.fit(X_train, y_train)
        
        # 3. Evaluate on TRUE holdout test set (X_test, y_test)
        print("Evaluating on holdout test set...")
        y_pred = model_clone.predict(X_test)
        y_proba = model_clone.predict_proba(X_test) if hasattr(model_clone, 'predict_proba') else None
        
        # 4. Prepare comprehensive test metrics
        test_report = classification_report(
            y_test, y_pred, 
            target_names=class_names, 
            output_dict=True,
            zero_division=0
        )
        
        # Update final metrics with test performance
        final_metrics.update({
            'test_accuracy': accuracy_score(y_test, y_pred),
            'test_precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'test_recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
            'test_f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
            'test_balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
            'test_class_report': test_report,
        })
        
        # Add probability metrics if available
        if y_proba is not None:
            probability_metrics = {}
            n_classes = y_proba.shape[1]
            
            if n_classes > 2:
                probability_metrics.update({
                    'test_roc_auc_ovr': roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro'),
                    'test_roc_auc_ovo': roc_auc_score(y_test, y_proba, multi_class='ovo', average='macro'),
                    'test_log_loss': log_loss(y_test, y_proba),
                })
            elif n_classes == 2:
                probability_metrics.update({
                    'test_roc_auc': roc_auc_score(y_test, y_proba[:, 1]),
                    'test_average_precision': average_precision_score(y_test, y_proba[:, 1]),
                    'test_log_loss': log_loss(y_test, y_proba),
                    'test_brier_score': brier_score_loss(y_test, y_proba[:, 1]),
                })
            
            final_metrics.update(probability_metrics)

        # 5. Save all artifacts with checks
        save_model_artifacts(
            model=model_clone,
            paths=paths,
            model_type=model_type,
            cv_metrics=cv_metrics,
            final_metrics=final_metrics,
            feature_importances=feature_importances,
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            test_report=test_report,
            search_results=search_results
        )
        
        # 6. Generate visualizations using the HOLDOUT test set
        print("Generating visualizations from test set...")
        generate_visualizations(
            model=model_clone,
            X_train=X_train,  # Full training set for context
            y_train=y_train,
            X_test=X_test,    # Holdout test set for evaluation
            y_test=y_test,
            class_names=class_names,
            feature_imp_df=pd.DataFrame(feature_importances, columns=['feature', 'importance']) if feature_importances else None,
            cv_metrics=cv_metrics,
            search_results=search_results,
            is_bayesian=is_bayesian,
            model_type=model_type,
            feature_selection_method=feature_selection_method,
            artifact_paths=paths
        )

        print_artifact_summary(paths, model_type)
        
        return final_metrics
        
    except Exception as e:
        print(f"\n❌ Error saving artifacts: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def generate_visualizations(
    model,
    X_train,  # Added for learning curve potential
    y_train,   # Added for learning curve potential
    X_test,    # Explicit test set for evaluation plots
    y_test,    # Explicit test set for evaluation plots
    class_names,
    feature_imp_df,
    cv_metrics,
    search_results,
    is_bayesian,
    model_type,
    feature_selection_method,
    artifact_paths
):
    """Generate visualizations and save to model/feature-selection specific folders.
    Uses X_test and y_test for evaluation plots to ensure unbiased results.
    """
    print(f"Generating visualizations for {model_type} ({feature_selection_method})...")
    
    # 1. Main evaluation plot (Feature Importances, CV Trends, Confusion Matrix, Hyperparams)
    # Main evaluation plot
    plt.figure(figsize=(20, 15))
    plt.suptitle(f'{model_type.title()} ({feature_selection_method}) Evaluation', y=1.02, fontsize=16)
    
    # Feature Importance Plot
    if feature_imp_df is not None and not feature_imp_df.empty:
        plt.subplot(2, 2, 1)
        top_features = feature_imp_df.nlargest(15, 'importance').sort_values('importance', ascending=True)
        sns.barplot(x='importance', y='feature', data=top_features)
        plt.title('Top 15 Feature Importances')
    
    # CV Metrics Trend
    plt.subplot(2, 2, 2)
    cv_df = pd.DataFrame(cv_metrics)
    metrics_to_plot = ['test_accuracy', 'f1_macro']
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
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig(f"{artifact_paths['plots']}/model_evaluation.png", bbox_inches='tight', dpi=300)
    plt.close()

 

    # 2. PROBABILITY PLOTS - CRITICAL FIXES HERE
    if hasattr(model, 'predict_proba'):
        try:
            # Use the TEST set for all probability evaluations
            y_proba_test = model.predict_proba(X_test)
            n_classes = y_proba_test.shape[1]

            plt.figure(figsize=(15, 5))
            plt.suptitle(f'Probability Calibration & Performance on Test Set', fontsize=14)

            # --- Subplot 1: Calibration Curves ---
            plt.subplot(1, 3, 1)
            for i, class_name in enumerate(class_names):
                # Check if we have samples for this class in the TEST set
                class_mask = (y_test == i)
                if np.sum(class_mask) > 0: # Ensure there are samples to avoid errors
                    fraction_of_positives, mean_predicted_value = calibration_curve(
                        class_mask.astype(int),
                        y_proba_test[:, i],
                        n_bins=10,
                        strategy='quantile'
                    )
                    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=f'{class_name}', alpha=0.8)
            plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
            plt.xlabel('Mean Predicted Probability (Test Set)')
            plt.ylabel('Fraction of Positives (Test Set)')
            plt.title('Calibration Plot')
            plt.legend(loc='best', fontsize='small')
            plt.grid(True, alpha=0.3)

            # --- Subplot 2: Probability Distribution ---
            plt.subplot(1, 3, 2)
            # Plot distribution for each class
            for i, class_name in enumerate(class_names):
                class_probs = y_proba_test[y_test == i, i] # Probabilities for true class i
                if len(class_probs) > 0:
                    sns.kdeplot(class_probs, label=f'True {class_name}', fill=True, alpha=0.6)
            plt.xlabel('Predicted Probability for True Class')
            plt.ylabel('Density')
            plt.title('Probability Distribution (Test Set)')
            plt.legend(loc='best', fontsize='small')
            plt.grid(True, alpha=0.3)

            # --- Subplot 3: ROC Curves (Fixed for Multiclass) ---
            plt.subplot(1, 3, 3)
            # Binarize the output for OvR ROC curves
            y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
            
            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba_test[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            # Plot all ROC curves
            colors = plt.cm.get_cmap('tab10', n_classes)
            for i, color in zip(range(n_classes), colors(range(n_classes))):
                plt.plot(fpr[i], tpr[i], color=color, lw=2,
                         label='ROC {0} (AUC = {1:0.2f})'
                         .format(class_names[i], roc_auc[i]))

            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves (One-vs-Rest)')
            plt.legend(loc="lower right", fontsize='small')
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(f"{artifact_paths['plots']}/probability_plots_test_set.png", bbox_inches='tight', dpi=300)
            plt.close()
            print("  - Saved probability plots (calibration, distribution, ROC)")

        except Exception as e:
            print(f"  - Error generating probability plots: {e}")
            # Log the error for debugging
            import traceback
            traceback.print_exc()
    else:
        print("  - Skipping probability plots (model does not have predict_proba)")



    print("Visualization generation complete.\n")

def safe_classification_report(y_true, y_pred, class_names):
    """Generate classification report with error handling"""
    try:
        return classification_report(
            y_true, y_pred,
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
    except Exception as e:
        print(f"Error generating classification report: {str(e)}")
        return {}

def get_probability_metrics(model, X_test, y_test):
    """Calculate probability-based metrics if available"""
    metrics = {}
    if hasattr(model, 'predict_proba'):
        try:
            y_proba = model.predict_proba(X_test)
            if y_proba.shape[1] > 2:  # Multiclass
                metrics.update({
                    'test_roc_auc_ovr': roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro'),
                    'test_log_loss': log_loss(y_test, y_proba)
                })
            else:  # Binary
                metrics.update({
                    'test_roc_auc': roc_auc_score(y_test, y_proba[:, 1]),
                    'test_log_loss': log_loss(y_test, y_proba)
                })
        except Exception as e:
            print(f"Probability metrics failed: {str(e)}")
    return metrics

def save_model_artifacts(
    model, paths, model_type, cv_metrics, final_metrics,
    feature_importances, numerical_features, categorical_features,
    test_report, search_results
):
    """Save all model files with validation"""
    # 1. Save model
    model_path = f"{paths['models']}/{model_type}_model.pkl"
    joblib.dump(model, model_path)
    
    # 2. Save metrics - ensure cv_metrics has required fields
    safe_save_metrics(
        cv_metrics=cv_metrics,
        final_metrics=final_metrics,
        paths=paths
    )
    
    # 3. Save test report
    pd.DataFrame(test_report).transpose().to_csv(
        f"{paths['metrics']}/test_classification_report.csv",
        index=True
    )
    
    # 4. Save feature importances if available
    if feature_importances:
        pd.DataFrame(feature_importances, columns=['feature', 'importance'])\
            .sort_values('importance', ascending=False)\
            .to_csv(f"{paths['metrics']}/feature_importances.csv", index=False)
    
    # 5. Save preprocessing info
    pd.DataFrame({
        'feature_type': ['numerical']*len(numerical_features) + ['categorical']*len(categorical_features),
        'feature_name': numerical_features + categorical_features
    }).to_csv(f"{paths['metrics']}/preprocessing_info.csv", index=False)
    
    # 6. Save best parameters if available
    if search_results and hasattr(search_results, 'best_params_'):
        pd.DataFrame.from_dict(search_results.best_params_, orient='index', columns=['value'])\
            .to_csv(f"{paths['metrics']}/best_params.csv")

def safe_save_metrics(cv_metrics, final_metrics, paths):
    """Save metrics with validation"""
    # Ensure all CV metrics have required fields
    required_fields = ['test_accuracy', 'f1_macro']
    cleaned_cv_metrics = []
    
    for m in cv_metrics:
        clean_metric = {k: m.get(k, None) for k in required_fields}
        if 'fold' in m:
            clean_metric['fold'] = m['fold']
        cleaned_cv_metrics.append(clean_metric)
    
    # Save CV metrics
    pd.DataFrame(cleaned_cv_metrics).to_csv(
        f"{paths['metrics']}/cross_validation_results.csv",
        index=False
    )
    
    # Prepare final metrics
    final_metrics_flat = {
        'cv_accuracy': [np.mean([m['test_accuracy'] for m in cleaned_cv_metrics if m['test_accuracy'] is not None])],
        'cv_f1_macro': [np.mean([m['f1_macro'] for m in cleaned_cv_metrics if m['f1_macro'] is not None])],
        'test_accuracy': [final_metrics.get('test_accuracy')],
        'test_f1_macro': [final_metrics.get('test_f1_macro')],
        'model_type': [final_metrics.get('model_type')],
        'feature_selection_method': [final_metrics.get('feature_selection_method')]
    }
    
    # Add probability metrics if available
    for metric in ['test_roc_auc_ovr', 'test_roc_auc', 'test_log_loss']:
        if metric in final_metrics:
            final_metrics_flat[metric] = [final_metrics[metric]]
    
    pd.DataFrame(final_metrics_flat).to_csv(
        f"{paths['metrics']}/final_metrics.csv",
        index=False
    )

def print_artifact_summary(paths, model_type):
    """Print summary of saved artifacts"""
    print("\n=== Saved Artifacts Summary ===")
    print(f"Model Type: {model_type}")
    print(f"\nModel saved to: {paths['models']}/")
    print(f"Metrics saved to: {paths['metrics']}/")
    print(f"Visualizations saved to: {paths['plots']}/")
    print("\nKey Files:")
    print(f"- Trained model: {model_type}_model.pkl")
    print("- Evaluation metrics: cross_validation_results.csv")
    print("- Test set report: test_classification_report.csv")
    print("- Feature importances: feature_importances.csv")


    print("Visualization generation complete.\n")

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

def get_feature_names_from_pipeline(model):
    """Extract feature names from sklearn pipeline"""
    try:
        # Handle preprocessor
        preprocessor = model.named_steps['preprocessor']
        feature_names = []
        
        # Numerical features
        if 'num' in preprocessor.named_transformers_:
            num_features = preprocessor.named_transformers_['num'].get_feature_names_out()
            feature_names.extend(num_features)
        
        # Handle categorical features - only if fitted
        if 'cat' in preprocessor.named_transformers_:
            ohe = preprocessor.named_transformers_['cat']
            if hasattr(ohe, 'get_feature_names_out'):
                try:
                    cat_features = ohe.get_feature_names_out()
                    feature_names.extend(cat_features)
                except NotFittedError:
                    print("Warning: OneHotEncoder not fitted yet")
        # Remainder features
        if hasattr(preprocessor, 'transformers_'):
            for name, trans, cols in preprocessor.transformers_:
                if name == 'remainder' and trans == 'passthrough':
                    feature_names.extend(cols)
        
        # Handle variance threshold
        if 'variance_threshold' in model.named_steps:
            vt = model.named_steps['variance_threshold']
            if hasattr(vt, 'get_support'):
                mask = vt.get_support()
                feature_names = [f for f, m in zip(feature_names, mask) if m]
        
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
    """Enhanced feature preparation with more robust handling"""
    X = df.copy()
    X = X.sort_values(by='date')  # Ensure chronological order for time series
    # Expanded list of leakage columns
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
        'maintime', 'first_half', 'second_half', 'country', 'extratime', 'matchday'
    } & set(X.columns)
    
    # Drop leakage columns safely
    X = X.drop(columns=leakage_columns, errors='ignore')
    
    # Enhanced formation handling
    for side in ['home', 'away']:
        formation_col = f'formation_{side}'
        if formation_col in X.columns:
            # Standardize and one-hot encode formations
            X[formation_col] = X[formation_col].astype(str).str.replace(r'[^\d-]', '', regex=True)
            valid_formations = X[formation_col].str.match(r'^\d+-\d+-\d+$')
            X.loc[~valid_formations, formation_col] = 'other'
            
            # Create formation dummies
            dummies = pd.get_dummies(X[formation_col], prefix=formation_col)
            X = pd.concat([X.drop(columns=[formation_col]), dummies], axis=1)
    
    # Enhanced categorical handling
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        # Handle rare categories
        counts = X[col].value_counts()
        mask = X[col].isin(counts[counts < 10].index)
        X.loc[mask, col] = 'rare_category'
        X[col] = X[col].astype(str)
    
    # Enhanced numeric handling
    num_cols = X.select_dtypes(include=np.number).columns
    for col in num_cols:
        # Handle infinite values
        X[col] = X[col].replace([np.inf, -np.inf], np.nan)
        # Fill NA with median (more robust than mean)
        X[col] = X[col].fillna(X[col].median())
    
    # Fill remaining categorical NAs
    if len(cat_cols) > 0:
        X[cat_cols] = X[cat_cols].fillna('missing')
    
 
    print(f"\nFeature Preparation Summary:")
    print(f"- Initial features: {len(df.columns)}")
    print(f"- After leakage removal: {len(X.columns)}")
    print(f"- Final features after processing: {len(X.columns)}")
    print(f"- Final numeric features: {len(X.select_dtypes(include=np.number).columns)}")
    print(f"- Final categorical features: {len(X.select_dtypes(include=['object', 'category']).columns)}")
    
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
        print(f"Average Accuracy: {cv_results['test_accuracy'].mean():.2%} (±{cv_results['test_accuracy'].std():.2%})")
        print(f"Average F1 Score: {cv_results['f1_macro'].mean():.2%} (±{cv_results['f1_macro'].std():.2%})")
        #print(f"Average ROC AUC: {cv_results['roc_auc_ovr'].mean():.2%} (±{cv_results['roc_auc_ovr'].std():.2%})")
        
        # Test set performance
        print("\nTest Set Evaluation:")
        print("-"*80)
        print(f"Accuracy: {final_metrics.iloc[0]['test_accuracy']:.2%}")
        print(f"F1 Score: {final_metrics.iloc[0]['test_f1_macro']:.2%}")
        if 'test_roc_auc_ovr' in final_metrics:
            print(f"ROC AUC: {final_metrics.iloc[0]['test_roc_auc_ovr']:.2%}")
        if 'test_log_loss' in final_metrics:
            print(f"Log Loss: {final_metrics.iloc[0]['test_log_loss']:.4f}")
        
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

def multi_class_brier_score(y_true, y_proba):
    """Calculate multi-class Brier score"""
    n_classes = y_proba.shape[1]
    y_true_bin = np.zeros((len(y_true), n_classes))
    for i in range(n_classes):
        y_true_bin[y_true == i, i] = 1
    return np.mean(np.sum((y_proba - y_true_bin) ** 2, axis=1))

def plot_learning_curve(estimator, X, y, cv=None, n_jobs=-1, 
                       train_sizes=np.linspace(0.1, 1.0, 5), 
                       scoring='accuracy', title="Learning Curve"):
    """
    Generate a basic learning curve to visualize model performance vs training size.
    
    Parameters:
    -----------
    estimator : model object
        The machine learning model to evaluate
    X : array-like
        Feature matrix
    y : array-like
        Target vector
    cv : cross-validation generator, default=None
        Cross-validation strategy
    n_jobs : int, default=-1
        Number of jobs to run in parallel
    train_sizes : array-like, default=np.linspace(0.1, 1.0, 5)
        Relative or absolute numbers of training examples
    scoring : str, default='accuracy'
        Scoring metric to use
    title : str, default="Learning Curve"
        Plot title
        
    Returns:
    --------
    matplotlib.figure.Figure
        The learning curve plot
    """
    if cv is None:
        cv = TimeSeriesSplit(n_splits=5)
    
    # Generate learning curve data
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,
        scoring=scoring, random_state=42, shuffle=False
    )
    
    # Calculate means and standard deviations
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot learning curve
    ax.grid(True, alpha=0.3)
    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                   train_scores_mean + train_scores_std, alpha=0.1, color="r")
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                   test_scores_mean + test_scores_std, alpha=0.1, color="g")
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r", 
           label="Training score", linewidth=2, markersize=6)
    ax.plot(train_sizes, test_scores_mean, 'o-', color="g", 
           label="Cross-validation score", linewidth=2, markersize=6)
    
    # Add final score annotations
    ax.annotate(f'Train: {train_scores_mean[-1]:.3f}',
               xy=(train_sizes[-1], train_scores_mean[-1]),
               xytext=(10, -20), textcoords='offset points',
               ha='left', va='top', fontsize=10,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.2))
    
    ax.annotate(f'CV: {test_scores_mean[-1]:.3f}',
               xy=(train_sizes[-1], test_scores_mean[-1]),
               xytext=(10, 10), textcoords='offset points',
               ha='left', va='bottom', fontsize=10,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.2))
    
    ax.set_xlabel("Training examples")
    ax.set_ylabel(f"{scoring.title()} Score")
    ax.set_title(title)
    ax.legend(loc="best")
    
    # Set y-axis limits for better visualization
    y_min = min(np.min(train_scores_mean - train_scores_std), 
               np.min(test_scores_mean - test_scores_std))
    y_max = max(np.max(train_scores_mean + train_scores_std), 
               np.max(test_scores_mean + test_scores_std))
    ax.set_ylim(y_min - 0.05, min(y_max + 0.05, 1.05))
    
    plt.tight_layout()
    return fig

def plot_comprehensive_learning_curve(estimator, X, y, cv=None, model_type=None, 
                                    feature_selection_method=None, n_jobs=-1):
    """
    Generate comprehensive learning curves with multiple metrics for detailed model diagnosis.
    
    Parameters:
    -----------
    estimator : model object
        The machine learning model to evaluate
    X : array-like
        Feature matrix
    y : array-like
        Target vector
    cv : cross-validation generator, default=None
        Cross-validation strategy
    model_type : str, default=None
        Type of model for title
    feature_selection_method : str, default=None
        Feature selection method for title
    n_jobs : int, default=-1
        Number of jobs to run in parallel
        
    Returns:
    --------
    matplotlib.figure.Figure
        The comprehensive learning curve plot
    """
    if cv is None:
        cv = TimeSeriesSplit(n_splits=5)
    
    # Define metrics to plot - handle multiclass vs binary appropriately
    unique_classes = np.unique(y)
    if len(unique_classes) > 2:
        # Multiclass metrics
        metrics = {
            'accuracy': make_scorer(accuracy_score),
            'f1_macro': make_scorer(f1_score, average='macro', zero_division=0),
            'precision_macro': make_scorer(precision_score, average='macro', zero_division=0),
            'recall_macro': make_scorer(recall_score, average='macro', zero_division=0)
        }
    else:
        # Binary classification metrics
        metrics = {
            'accuracy': make_scorer(accuracy_score),
            'f1': make_scorer(f1_score, zero_division=0),
            'precision': make_scorer(precision_score, zero_division=0),
            'recall': make_scorer(recall_score, zero_division=0),
            'roc_auc': make_scorer(roc_auc_score, needs_proba=True)
        }
    
    # Create subplots
    n_metrics = len(metrics)
    n_cols = 2
    n_rows = (n_metrics + 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.ravel() if n_metrics > 1 else [axes]
    
    train_sizes = np.linspace(0.1, 1.0, 5)
    colors = plt.cm.Set2(np.linspace(0, 1, len(metrics)))
    
    for idx, ((metric_name, scorer), color) in enumerate(zip(metrics.items(), colors)):
        ax = axes[idx]
        
        try:
            # Generate learning curve data for this metric
            train_sizes, train_scores, test_scores = learning_curve(
                estimator, X, y, cv=cv, train_sizes=train_sizes,
                scoring=scorer, n_jobs=n_jobs, random_state=42, shuffle=False
            )
            
            # Calculate means and standard deviations
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            test_mean = np.mean(test_scores, axis=1)
            test_std = np.std(test_scores, axis=1)
            
            # Plot learning curve
            ax.grid(True, alpha=0.3)
            ax.fill_between(train_sizes, train_mean - train_std,
                         train_mean + train_std, alpha=0.2, color=color)
            ax.fill_between(train_sizes, test_mean - test_std,
                         test_mean + test_std, alpha=0.2, color=color, hatch='//')
            ax.plot(train_sizes, train_mean, 'o-', color=color, 
                   label="Training", linewidth=2, markersize=5)
            ax.plot(train_sizes, test_mean, 's--', color=color, 
                   label="CV", linewidth=2, markersize=5)
            
            # Add score annotations
            gap = test_mean[-1] - train_mean[-1]
            gap_text = f"Gap: {gap:+.3f}"
            
            ax.annotate(f'Train: {train_mean[-1]:.3f}',
                       xy=(train_sizes[-1], train_mean[-1]),
                       xytext=(10, -25), textcoords='offset points',
                       ha='left', va='top', fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
            
            ax.annotate(f'CV: {test_mean[-1]:.3f}\n{gap_text}',
                       xy=(train_sizes[-1], test_mean[-1]),
                       xytext=(10, 10), textcoords='offset points',
                       ha='left', va='bottom', fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
            
            ax.set_xlabel("Training examples")
            ax.set_ylabel(metric_name.replace('_', ' ').title())
            ax.set_title(f'Learning Curve ({metric_name})')
            ax.legend(loc="best")
            
            # Set appropriate y-axis limits
            all_scores = np.concatenate([train_scores.flatten(), test_scores.flatten()])
            y_min, y_max = np.nanmin(all_scores), np.nanmax(all_scores)
            y_range = y_max - y_min
            ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
            
        except Exception as e:
            # Handle errors gracefully
            ax.text(0.5, 0.5, f"Error plotting {metric_name}:\n{str(e)[:100]}...", 
                   ha='center', va='center', transform=ax.transAxes, fontsize=9)
            ax.set_title(f'Learning Curve ({metric_name}) - Error')
            ax.set_facecolor('#FFF0F0')  # Light red background for errors
    
    # Hide empty subplots if any
    for idx in range(len(metrics), len(axes)):
        axes[idx].set_visible(False)
    
    # Add main title
    title_parts = []
    if model_type:
        title_parts.append(model_type.title())
    if feature_selection_method:
        title_parts.append(f"({feature_selection_method})")
    title_parts.append("Learning Curves")
    
    main_title = " ".join(title_parts)
    plt.suptitle(main_title, fontsize=16, y=0.98)
    plt.tight_layout()
    
    return fig

def get_estimator_for_learning_curve(pipeline):
    """
    Extract the appropriate estimator from a pipeline for learning curve analysis.
    Handles the case where feature selection might be part of the pipeline.
    
    Returns:
    - The final estimator if it's a single model
    - The full pipeline if feature selection is involved (to maintain feature compatibility)
    """
    # If it's not a pipeline, return as is
    if not hasattr(pipeline, 'named_steps') and not hasattr(pipeline, 'steps'):
        return pipeline
    
    # Check if this is a full pipeline with preprocessing + feature selection + classifier
    try:
        # Get all step names
        if hasattr(pipeline, 'named_steps'):
            step_names = list(pipeline.named_steps.keys())
        else:
            step_names = [name for name, _ in pipeline.steps]
        
        # If we have feature selection in the pipeline, we need to use the FULL pipeline
        # because the feature selection transforms the data
        feature_selection_keywords = ['feature_selection', 'selector', 'select', 'feature']
        has_feature_selection = any(any(keyword in step_name.lower() 
                                      for keyword in feature_selection_keywords) 
                              for step_name in step_names)
        
        if has_feature_selection:
            print("  - Pipeline contains feature selection, using full pipeline for learning curves")
            return pipeline
        
        # Otherwise, try to extract the final estimator
        if hasattr(pipeline, 'named_steps'):
            if 'classifier' in pipeline.named_steps:
                return pipeline.named_steps['classifier']
            elif 'regressor' in pipeline.named_steps:
                return pipeline.named_steps['regressor']
        
        # Return the last step (assuming it's the estimator)
        return pipeline.steps[-1][1]
        
    except (AttributeError, IndexError, TypeError) as e:
        print(f"  - Could not extract estimator from pipeline: {e}, using full pipeline")
        return pipeline
    
if __name__ == "__main__":
    try:
        # Load data
        df = pd.read_csv('data/final_processed.csv')
        
        # Example usage:
        # Random Forest with Bayesian optimization and PCA feature selection
        trained_model = train_and_save_model(
            df, 
            model_type='xgboost',
            feature_selection_method='rfe',
            load_params=True,
            use_bayesian=False,
            use_grid_search=False,
            use_random_search=False,
            top_n_features=40,
        
        )

        
    except Exception as e:
        print(f"\nTraining failed: {str(e)}")
        raise