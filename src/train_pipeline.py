# ==================== IMPORTS ====================
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn imports
from sklearn.model_selection import TimeSeriesSplit, learning_curve, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, 
    roc_curve, confusion_matrix, classification_report, log_loss, brier_score_loss, 
    average_precision_score, precision_recall_curve, balanced_accuracy_score, 
    mean_squared_error, mean_absolute_error, r2_score,
    explained_variance_score, auc
)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from sklearn.calibration import calibration_curve
from sklearn.pipeline import Pipeline
#from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel, RFE, RFECV, SelectKBest, SequentialFeatureSelector, f_classif, mutual_info_classif, f_regression, mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.base import clone
from sklearn.utils import check_X_y
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import label_binarize
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, PoissonRegressor, Ridge, Lasso, ElasticNet
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor

# Bayesian optimization
from skopt import BayesSearchCV
from skopt.plots import plot_convergence
from skopt.space import Real, Integer, Categorical

# XGBoost
from xgboost import XGBClassifier, XGBRegressor

from src.utils import get_feature_names_from_pipeline

# ==================== MODEL CONFIGURATIONS ====================
# Define model parameter grids
CLASSIFICATION_MODEL_PARAMS = {
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
            'classifier__n_estimators': Integer(50, 300),
            'classifier__max_depth': Integer(3, 9),
            'classifier__learning_rate': Real(0.01, 0.2, prior='log-uniform'),
            'classifier__subsample': Real(0.7, 1.0),
            'classifier__colsample_bytree': Real(0.7, 1.0),
            'classifier__gamma': Real(0, 2),
            'classifier__reg_alpha': Real(0, 5),
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
            'classifier__degree': Integer(2, 5)
        },
        'grid': {
            'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'classifier__kernel': ['linear', 'rbf'],
            'classifier__gamma': ['scale', 'auto'] + [0.001, 0.01, 0.1, 1]
        }
    }
}

REGRESSION_MODEL_PARAMS = {
    'random_forest': {
        'bayesian': {
            'regressor__n_estimators': Integer(50, 500),
            'regressor__max_depth': Integer(3, 30),
            'regressor__min_samples_split': Integer(2, 20),
            'regressor__min_samples_leaf': Integer(1, 10),
            'regressor__max_features': Categorical(['sqrt', 'log2', None]),
            'regressor__bootstrap': Categorical([True, False])
        },
        'grid': {
            'regressor__n_estimators': [100, 200, 300],
            'regressor__max_depth': [None, 5, 10, 20],
            'regressor__min_samples_split': [2, 5, 10],
            'regressor__min_samples_leaf': [1, 2, 4],
            'regressor__max_features': ['sqrt', 'log2', None],
            'regressor__bootstrap': [True, False]
        }
    },
    'linear_regression': {
        'bayesian': {
            'regressor__alpha': Real(1e-4, 1e4, prior='log-uniform'),
            'regressor__max_iter': Integer(100, 1000)
        },
        'grid': {
            'regressor__alpha': [0.001, 0.01, 0.1, 1, 10],
            'regressor__max_iter': [100, 500, 1000]
        }
    },
    'xgboost': {
        'bayesian': {
            'regressor__n_estimators': Integer(50, 300),
            'regressor__max_depth': Integer(3, 9),
            'regressor__learning_rate': Real(0.01, 0.2, prior='log-uniform'),
            'regressor__subsample': Real(0.7, 1.0),
            'regressor__colsample_bytree': Real(0.7, 1.0),
            'regressor__gamma': Real(0, 2),
            'regressor__reg_alpha': Real(0, 5),
            'regressor__reg_lambda': Real(0, 5)
        },
        'grid': {
            'regressor__n_estimators': [100, 200, 300],
            'regressor__max_depth': [3, 6, 9],
            'regressor__learning_rate': [0.01, 0.1, 0.2],
            'regressor__subsample': [0.6, 0.8, 1.0],
            'regressor__colsample_bytree': [0.6, 0.8, 1.0],
            'regressor__gamma': [0, 1, 5],
            'regressor__reg_alpha': [0, 0.1, 1],
            'regressor__reg_lambda': [0, 0.1, 1]
        }
    },
    'gradient_boosting': {
        'bayesian': {
            'regressor__n_estimators': Integer(50, 500),
            'regressor__learning_rate': Real(0.01, 0.2, prior='log-uniform'),
            'regressor__max_depth': Integer(3, 10),
            'regressor__min_samples_split': Integer(2, 20),
            'regressor__min_samples_leaf': Integer(1, 10),
            'regressor__max_features': Categorical(['sqrt', 'log2', None])
        },
        'grid': {
            'regressor__n_estimators': [100, 200, 300],
            'regressor__learning_rate': [0.01, 0.1, 0.2],
            'regressor__max_depth': [3, 5, 7],
            'regressor__min_samples_split': [2, 5, 10],
            'regressor__min_samples_leaf': [1, 2, 4],
            'regressor__max_features': ['sqrt', 'log2', None]
        }
    },
    'svm': {
        'bayesian': {
            'regressor__C': Real(1e-4, 1e4, prior='log-uniform'),
            'regressor__kernel': Categorical(['linear', 'rbf', 'poly']),
            'regressor__gamma': Real(1e-4, 1e4, prior='log-uniform'),
            'regressor__degree': Integer(2, 5)
        },
        'grid': {
            'regressor__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'regressor__kernel': ['linear', 'rbf'],
            'regressor__gamma': ['scale', 'auto'] + [0.001, 0.01, 0.1, 1]
        }
    }
}

# Define default parameter sets for each model type
CLASSIFICATION_DEFAULT_PARAMS = {
    'random_forest': {
        'n_estimators': 300,
        'max_depth': None,
        'min_samples_split': 5,
        'min_samples_leaf': 1,
        'max_features': 'sqrt',
        'bootstrap': False,
        'class_weight': 'balanced',
        'criterion': 'entropy',
        'n_jobs': -1
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
        'tree_method': 'hist',
          # Add these for class imbalance:
        'scale_pos_weight': None,  # For binary, use custom for multi-class
        'class_weight': 'balanced',  # This needs custom implementation
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

REGRESSION_DEFAULT_PARAMS = {
    'random_forest': {
        'n_estimators': 300,
        'max_depth': None,
        'min_samples_split': 5,
        'min_samples_leaf': 1,
        'max_features': 'sqrt',
        'bootstrap': False,
        'n_jobs': -1
    },
    'linear_regression': {
        'alpha': 1.0,
        'max_iter': 1000,
        'tol': 1e-4
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
        'n_jobs': -1,
        'tree_method': 'hist'
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
        'random_state': 42
    }
}

class TrainPipeline:
    """A comprehensive machine learning pipeline for both classification and regression tasks"""
    
    def __init__(self, task_type='auto', random_state=42, log_level=logging.INFO):
        """
        Initialize the ML pipeline
        
        Parameters:
        -----------
        task_type : str, default='auto'
            Type of ML task: 'classification', 'regression', or 'auto' for automatic detection
        random_state : int, default=42
            Random seed for reproducibility
        log_level : int, default=logging.INFO
            Logging level
        """
        self.task_type = task_type
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        
        # Initialize attributes
        self.model = None
        self.X = None
        self.y = None
        self.le = None
        self.feature_names = None
        self.cv_metrics = None
        self.final_metrics = None
        self.paths = None
        
        # Load model configurations
        self.model_params = CLASSIFICATION_MODEL_PARAMS if task_type == 'classification' else REGRESSION_MODEL_PARAMS
        self.default_params = CLASSIFICATION_DEFAULT_PARAMS if task_type == 'classification' else REGRESSION_DEFAULT_PARAMS
        
        self.logger.info(f"Initialized MLPipeline for {task_type} task")
    
    def _initialize(self):
        """Initialize the trainer only when needed"""
        from src.train_pipeline import TrainPipeline
        self._trainer = TrainPipeline(
            task_type=self.config.get('task_type', 'auto'),
            random_state=self.config.get('random_state', 42)
        )
    
    
    def setup_directories(self, model_type, feature_selection_method, target_col=None):
        """Create directory structure for artifacts with target-specific naming"""
        # Include target column in path if provided
        if target_col:
            base_path = f"artifacts/{target_col}/{model_type}/{feature_selection_method}"
            os.makedirs(base_path, exist_ok=True)
        else:
            base_path = f"artifacts/{model_type}/{feature_selection_method}"
            os.makedirs(base_path, exist_ok=True)
        
        self.paths = {
            'base': base_path,
            'plots': f"{base_path}/plots",
            'metrics': f"{base_path}/metrics", 
            'models': f"{base_path}/models",
            'logs': "logs/train"
        }
        
        for path in self.paths.values():
            os.makedirs(path, exist_ok=True)
        
        # Setup file logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"train_{target_col}_{timestamp}.log" if target_col else f"train_{timestamp}.log"
        file_handler = logging.FileHandler(f"{self.paths['logs']}/{log_filename}")
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(file_handler)
        
        return self.paths 
 
    def prepare_data(self, df, target_col):
        """Prepare features and target with proper validation"""
        self.logger.info(f"Preparing data with target column: {target_col}")
        
        if target_col not in df.columns:
            error_msg = f"Target column '{target_col}' not found in DataFrame"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Filter completed matches
        if 'status' in df.columns:
            df = df[df['status'].isin(['FT', 'AET', 'PEN'])]
            self.logger.info(f"Filtered to {len(df)} completed matches")
        
        # Sort by date if available
        if 'date' in df.columns:
            df = df.sort_values('date')
            self.logger.info("Sorted data chronologically by date")
        
        # Prepare target
        y = df[target_col].copy()
        
        # Auto-detect task type if needed
        if self.task_type == 'auto':
            self.task_type = detect_task_type(y)
            self.logger.info(f"Auto-detected task type: {self.task_type}")
            
            # Update model configurations based on detected task type
            if self.task_type == 'classification':
                self.model_params = CLASSIFICATION_MODEL_PARAMS
                self.default_params = CLASSIFICATION_DEFAULT_PARAMS
            else:
                self.model_params = REGRESSION_MODEL_PARAMS
                self.default_params = REGRESSION_DEFAULT_PARAMS
        
        # Encode target for classification
        if self.task_type == 'classification':
            self.le = LabelEncoder()
            y_encoded = pd.Series(self.le.fit_transform(y), index=y.index)
            self.logger.info(f"Encoded classes: {dict(zip(self.le.classes_, range(len(self.le.classes_))))}")
        else:
            y_encoded = y
        
        # Prepare features
        X = self.prepare_features(df, target_col)
        
        self.X = X
        self.y = y_encoded
        
        self.logger.info(f"Data preparation complete. Features: {X.shape[1]}, Samples: {X.shape[0]}")
        return X, y_encoded
    
    def prepare_features(self, df, target_col):
        """Enhanced feature preparation with more robust handling"""
        self.logger.info("Preparing features...")
        
        X = df.copy()
        
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
            'maintime', 'first_half', 'second_half', 'country', 'extratime', 'matchday', 'odds_home_win', 'odds_draw', 'odds_away_win',
            'total_yellow_cards', 'total_red_cards', 'total_corners', 'outcome'
        } & set(X.columns)
        
        # Drop leakage columns safely
        X = X.drop(columns=leakage_columns, errors='ignore')
        self.logger.info(f"Removed {len(leakage_columns)} potential leakage columns")
        
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
                self.logger.info(f"Processed {formation_col} into {dummies.shape[1]} dummy variables")
        
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
        
        self.logger.info(f"Feature preparation complete. Final features: {len(X.columns)}")
        return X
    
 
    def build_pipeline(self, model_type, feature_selection_method, top_n_features=30):
        """Build the complete ML pipeline"""
        self.logger.info(f"Building pipeline for {model_type} with {feature_selection_method} feature selection")
        
        # Identify feature types
        numerical_features = self.X.select_dtypes(include=np.number).columns.tolist()
        categorical_features = self.X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        self.logger.info(f"Numerical features: {len(numerical_features)}, Categorical features: {len(categorical_features)}")
        
        # Create a list of transformers dynamically
        transformers = []
        
        # Always add the numerical transformer
        if numerical_features:
            transformers.append(('num', StandardScaler(), numerical_features))
        else:
            self.logger.warning("No numerical features found!")
        
        # Only add the categorical transformer if categorical features exist
        if categorical_features:
            transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features))
        else:
            self.logger.info("No categorical features found. Skipping OneHotEncoder.")
        
        # Check if we have any transformers at all
        if not transformers:
            raise ValueError("No features found for preprocessing! Check your dataset.")
        
        # Create preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='passthrough'  # This will handle any other dtypes you might have missed
        )
        
        # Get feature selector
        tscv = TimeSeriesSplit(n_splits=5)
        feature_selector = get_feature_selector(
            feature_selection_method=feature_selection_method,
            model_type=model_type,
            top_n_features=top_n_features,
            tscv=tscv,
            task_type=self.task_type
        )
        
        # Build base pipeline
        base_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('variance_threshold', VarianceThreshold(threshold=0.01)),
            ('feature_selection', feature_selector)
        ])
        
        return base_pipeline

    def initialize_search(self, model_type, base_pipeline, use_bayesian=False, 
                         use_grid_search=False, use_random_search=False, 
                         bayesian_iter=50, random_search_iter=50):
        """Initialize the appropriate parameter search strategy"""
        self.logger.info("Initializing parameter search")
        
        if not any([use_bayesian, use_grid_search, use_random_search]):
            self.logger.info("No parameter search selected, using default parameters")
            return None
        
        # Get model class and parameters
        model_class = get_model_class(model_type, self.task_type)
        model_params, _ = get_model_params(model_type, self.task_type)
        
        # Create full pipeline
        estimator_name = 'classifier' if self.task_type == 'classification' else 'regressor'
        full_pipeline = Pipeline([
            *base_pipeline.steps,
            (estimator_name, model_class)
        ])
        
        # Initialize appropriate search
        if use_bayesian:
            search_space = model_params.get('bayesian', {})
            self.logger.info(f"Initializing Bayesian search with {bayesian_iter} iterations")
            return BayesSearchCV(
                estimator=full_pipeline,
                search_spaces=search_space,
                n_iter=bayesian_iter,
                cv=TimeSeriesSplit(n_splits=5),
                scoring='accuracy' if self.task_type == 'classification' else 'r2',
                n_jobs=-1,
                random_state=self.random_state
            )
        elif use_grid_search:
            search_space = model_params.get('grid', {})
            self.logger.info("Initializing Grid search")
            return GridSearchCV(
                estimator=full_pipeline,
                param_grid=search_space,
                cv=TimeSeriesSplit(n_splits=5),
                scoring='accuracy' if self.task_type == 'classification' else 'r2',
                n_jobs=-1,
                verbose=2
            )
        else:  # random_search
            search_space = model_params.get('grid', {})
            self.logger.info(f"Initializing Random search with {random_search_iter} iterations")
            return RandomizedSearchCV(
                estimator=full_pipeline,
                param_distributions=search_space,
                n_iter=random_search_iter,
                cv=TimeSeriesSplit(n_splits=5),
                scoring='accuracy' if self.task_type == 'classification' else 'r2',
                n_jobs=-1,
                random_state=self.random_state,
                verbose=2
            )
    

    def train_2(self, df, target_col='outcome', model_type='random_forest',
            feature_selection_method='importance', top_n_features=30,
            use_bayesian=False, bayesian_iter=50, use_grid_search=False,
            use_random_search=False, random_search_iter=50, load_params=False,
            holdout_ratio=0.2, handle_class_imbalance=True, class_weights_dict=None,  
            use_smote=False, smote_strategy=None):
        """
        Train the model with the specified configuration
        
        Parameters:
        -----------
        handle_class_imbalance : bool, default=True
            Whether to handle class imbalance for classification tasks
            
        Returns:
        --------
        model : trained model
        X_holdout : holdout features
        y_holdout : holdout targets
        """
        try:
            # Setup directories and logging
            self.setup_directories(model_type, feature_selection_method, target_col)
            self.logger.info(f"Starting training for {target_col} with model: {model_type}, feature selection: {feature_selection_method} with {top_n_features} features")
            
            # Prepare data
            X, y = self.prepare_data(df, target_col)
            
            # CREATE PROPER TIME SERIES HOLDOUT SET
            n_samples = len(X)
            holdout_size = int(n_samples * holdout_ratio)
            
            # Split into training (early data) and holdout (most recent data)
            X_train_full = X.iloc[:-holdout_size]
            y_train_full = y.iloc[:-holdout_size]
            X_holdout = X.iloc[-holdout_size:]
            y_holdout = y.iloc[-holdout_size:]
            
            self.logger.info(f"Time Series Split - Training: {X_train_full.shape[0]}, Holdout: {X_holdout.shape[0]}")
            
            # Handle class imbalance for classification tasks
            sample_weights = None
            class_weights = None


            
            # Build pipeline
            base_pipeline = self.build_pipeline(model_type, feature_selection_method, top_n_features)
            
            # Training logic
            if use_bayesian or use_grid_search or use_random_search:
                search = self.initialize_search(
                    model_type=model_type,
                    base_pipeline=base_pipeline,
                    use_bayesian=use_bayesian,
                    use_grid_search=use_grid_search,
                    use_random_search=use_random_search,
                    bayesian_iter=bayesian_iter,
                    random_search_iter=random_search_iter
                )
                
                self.logger.info("Starting parameter search...")
                # Don't pass sample weights if using SMOTE
                if use_smote:
                    search.fit(X_train_full, y_train_full)
                else:
                    search.fit(X_train_full, y_train_full, classifier__sample_weight=sample_weights if sample_weights is not None else None)
                self.model = search.best_estimator_
                
                # Save search results
                search_results_df = pd.DataFrame(search.cv_results_)
                search_results_df.to_csv(f"{self.paths['metrics']}/search_results.csv", index=False)
                pd.DataFrame([search.best_params_]).to_csv(f"{self.paths['metrics']}/best_params.csv", index=False)
                self.logger.info(f"Best parameters: {search.best_params_}")
            else:
                # Use default parameters
                _, default_params = get_model_params(model_type, self.task_type)
                classifier_params = default_params.copy()
                
                if isinstance(load_params, dict):
                    classifier_params.update(load_params)
                
                # Remove class weight parameter if using SMOTE
                if use_smote and 'class_weight' in classifier_params:
                    classifier_params.pop('class_weight', None)
                    self.logger.info("Removed class_weight parameter due to SMOTE usage")
                
                estimator_name = 'classifier' if self.task_type == 'classification' else 'regressor'
                model_instance = get_model_class(model_type, self.task_type)()
                model_instance.set_params(**classifier_params)
                
                # Build the final pipeline with or without SMOTE
                if use_smote:
                    from imblearn.pipeline import Pipeline as ImbPipeline
                    # Extract steps from base_pipeline and add SMOTE + classifier
                    pipeline_steps = base_pipeline.steps.copy()
                    pipeline_steps.append(('smote', SMOTE(
                        sampling_strategy=smote_strategy, 
                        random_state=42,
                        k_neighbors=min(5, len(np.unique(y_train_full)) - 1)
                    )))
                    pipeline_steps.append((estimator_name, model_instance))
                    self.model = ImbPipeline(pipeline_steps)
                else:
                    # Regular pipeline without SMOTE
                    pipeline_steps = base_pipeline.steps.copy()
                    pipeline_steps.append((estimator_name, model_instance))
                    self.model = Pipeline(pipeline_steps)
            
            # Cross-validation with sample weights (except when using SMOTE)
            self.logger.info("Starting cross-validation...")
            tscv = TimeSeriesSplit(n_splits=5)
            self.cv_metrics = []
            
            for fold, (train_idx, test_idx) in enumerate(tscv.split(X_train_full)):
                self.logger.info(f"Processing fold {fold + 1}/{tscv.n_splits}")
                
                # Clone model for each fold
                fold_model = clone(self.model)
                
                # Get sample weights for this fold (skip if using SMOTE)
                fold_sample_weights = None
                if not use_smote and sample_weights is not None:
                    fold_sample_weights = sample_weights[train_idx]
                
                # Train and evaluate
                if self.task_type == 'classification':
                    class_names = ['away_win', 'draw', 'home_win'] if self.le else None
                    metrics = evaluate_classification_model(
                        fold_model,
                        X_train_full.iloc[train_idx], X_train_full.iloc[test_idx],
                        y_train_full.iloc[train_idx], y_train_full.iloc[test_idx],
                        self.logger,
                        fold=fold,
                        class_names=class_names,
                        sample_weight=fold_sample_weights,  # Don't pass weights if using SMOTE
                        use_smote=use_smote,
                        smote_strategy=smote_strategy
                    )
                else:
                    metrics = evaluate_regression_model(
                        fold_model,
                        X_train_full.iloc[train_idx], X_train_full.iloc[test_idx],
                        y_train_full.iloc[train_idx], y_train_full.iloc[test_idx],
                        self.logger,
                        fold=fold
                    )
                
                self.cv_metrics.append(metrics)
            
            # Final training on full training period
            self.logger.info("Training final model on full training period...")
            if not use_smote and sample_weights is not None:
                self.model.fit(X_train_full, y_train_full, classifier__sample_weight=sample_weights)
            else:
                self.model.fit(X_train_full, y_train_full)
            
            # Evaluate on holdout period
            self.logger.info("Evaluating on holdout period...")
            if self.task_type == 'classification':
                y_pred_holdout = self.model.predict(X_holdout)
                holdout_accuracy = accuracy_score(y_holdout, y_pred_holdout)
                
                # Generate comprehensive holdout metrics
                class_names = ['away_win', 'draw', 'home_win'] if self.le else None
                holdout_report = classification_report(
                    y_holdout, y_pred_holdout, 
                    target_names=class_names, 
                    output_dict=True,
                    zero_division=0
                )
                
                self.final_metrics = {
                    'model_type': model_type,
                    'feature_selection_method': feature_selection_method,
                    'holdout_accuracy': holdout_accuracy,
                    'holdout_precision_macro': holdout_report['macro avg']['precision'],
                    'holdout_recall_macro': holdout_report['macro avg']['recall'],
                    'holdout_f1_macro': holdout_report['macro avg']['f1-score'],
                    'train_samples': len(X_train_full),
                    'holdout_samples': len(X_holdout),
                    'holdout_ratio': holdout_ratio,
                    'handled_class_imbalance': handle_class_imbalance,
                    'used_sample_weights': sample_weights is not None,
                    'used_smote': use_smote,
                    'smote_strategy': str(smote_strategy) if use_smote else None,
                }
                
                # Add class-specific metrics if available
                if 'draw' in holdout_report:
                    self.final_metrics['draw_recall'] = holdout_report['draw']['recall']
                    self.final_metrics['draw_precision'] = holdout_report['draw']['precision']
                
                self.logger.info(f"Holdout Accuracy: {holdout_accuracy:.4f}")
            else:
                y_pred_holdout = self.model.predict(X_holdout)
                holdout_rmse = np.sqrt(mean_squared_error(y_holdout, y_pred_holdout))
                holdout_r2 = r2_score(y_holdout, y_pred_holdout)
                
                self.final_metrics = {
                    'model_type': model_type,
                    'feature_selection_method': feature_selection_method,
                    'holdout_rmse': holdout_rmse,
                    'holdout_r2': holdout_r2,
                    'holdout_mae': mean_absolute_error(y_holdout, y_pred_holdout),
                    'train_samples': len(X_train_full),
                    'holdout_samples': len(X_holdout),
                    'holdout_ratio': holdout_ratio,
                    'handled_class_imbalance': handle_class_imbalance,
                    'used_sample_weights': sample_weights is not None,
                    'used_smote': use_smote,
                    'smote_strategy': str(smote_strategy) if use_smote else None,
                }
                
                self.logger.info(f"Holdout RMSE: {holdout_rmse:.4f}, R²: {holdout_r2:.4f}")
            
            # Save model and artifacts
            self.save_artifacts(X_train_full, y_train_full, X_holdout, y_holdout, holdout_report, target_col)

            self.generate_visualizations(X_holdout, y_holdout, target_col)
            
            self.logger.info("Training completed successfully!")
            self.print_model_report(model_type, feature_selection_method, target_col)

            return self.model, X_holdout, y_holdout
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}", exc_info=True)
            raise

    def train(self, df, target_col='outcome', model_type='random_forest',
            feature_selection_method='importance', top_n_features=30,
            use_bayesian=False, bayesian_iter=50, use_grid_search=False,
            use_random_search=False, random_search_iter=50, load_params=False,
            holdout_ratio=0.2, handle_class_imbalance=True, imbalance_method='auto',
            class_weights_dict=None, use_smote=False, smote_strategy=None):
        """
        Train the model with the specified configuration
        
        Parameters:
        -----------
        handle_class_imbalance : bool, default=True
            Whether to handle class imbalance for classification tasks
            
        Returns:
        --------
        model : trained model
        X_holdout : holdout features
        y_holdout : holdout targets
        """
        try:
            # Setup directories and logging
            self.setup_directories(model_type, feature_selection_method, target_col)
            self.logger.info(f"Starting training for {target_col} with model: {model_type}, feature selection: {feature_selection_method} with {top_n_features} features")
            
            # Prepare data
            X, y = self.prepare_data(df, target_col)
            
            # CREATE PROPER TIME SERIES HOLDOUT SET
            n_samples = len(X)
            holdout_size = int(n_samples * holdout_ratio)
            
            # Split into training (early data) and holdout (most recent data)
            X_train_full = X.iloc[:-holdout_size]
            y_train_full = y.iloc[:-holdout_size]
            X_holdout = X.iloc[-holdout_size:]
            y_holdout = y.iloc[-holdout_size:]
            
            self.logger.info(f"Time Series Split - Training: {X_train_full.shape[0]}, Holdout: {X_holdout.shape[0]}")
            
            # Handle class imbalance for classification tasks
            imbalance_config = None
            sample_weights = None
            class_weights = None
            smote_instance = None

            if self.task_type == 'classification' and handle_class_imbalance:
                # Validate configuration
                is_valid, error_msg = validate_imbalance_config(
                    imbalance_method, use_smote, class_weights_dict
                )
                if not is_valid:
                    raise ValueError(f"Invalid imbalance configuration: {error_msg}")
                
                # Determine actual method based on parameters
                actual_method = imbalance_method
                if use_smote:
                    actual_method = 'smote'
                elif class_weights_dict is not None:
                    actual_method = 'custom_weights'
                
                # Apply imbalance handling
                imbalance_config = handle_class_imbalance(
                    y_train=y_train_full,
                    method=actual_method,
                    custom_weights=class_weights_dict,
                    smote_strategy=smote_strategy,
                    random_state=self.random_state
                )
                
                sample_weights = imbalance_config['sample_weights']
                class_weights = imbalance_config['class_weights']
                smote_instance = imbalance_config['smote_instance']
                
                self.logger.info(f"Class imbalance handling: {imbalance_config['method']}")
                self.logger.info(f"Class distribution: {imbalance_config['class_distribution']}")
            
            # Build base pipeline (without the final estimator)
            base_pipeline = self.build_pipeline(model_type, feature_selection_method, top_n_features)
            
            # Get model instance and parameters
            estimator_name = 'classifier' if self.task_type == 'classification' else 'regressor'
            model_class = get_model_class(model_type, self.task_type)
            model_instance = model_class()
            
            # Set up parameters
            if use_bayesian or use_grid_search or use_random_search:
                # For hyperparameter search, parameters will be set by the search
                classifier_params = {}
            else:
                # Use default or loaded parameters
                _, default_params = get_model_params(model_type, self.task_type)
                classifier_params = default_params.copy()
                
                if isinstance(load_params, dict):
                    classifier_params.update(load_params)
                
                # Remove class weight parameter if using SMOTE
                if smote_instance is not None and 'class_weight' in classifier_params:
                    classifier_params.pop('class_weight', None)
                    self.logger.info("Removed class_weight parameter due to SMOTE usage")
                
                model_instance.set_params(**classifier_params)
            
            # Build the complete pipeline with SMOTE if applicable
            pipeline_steps = base_pipeline.steps.copy()
            
            if smote_instance is not None:
                from imblearn.pipeline import Pipeline as ImbPipeline
                pipeline_steps.append(('smote', smote_instance))
                pipeline_steps.append((estimator_name, model_instance))
                self.model = ImbPipeline(pipeline_steps)
            else:
                pipeline_steps.append((estimator_name, model_instance))
                self.model = Pipeline(pipeline_steps)
            
            # Training logic
            if use_bayesian or use_grid_search or use_random_search:
                search = self.initialize_search(
                    model_type=model_type,
                    base_pipeline=self.model,  # Pass the complete pipeline
                    use_bayesian=use_bayesian,
                    use_grid_search=use_grid_search,
                    use_random_search=use_random_search,
                    bayesian_iter=bayesian_iter,
                    random_search_iter=random_search_iter
                )
                
                self.logger.info("Starting parameter search...")
                # Don't pass sample weights if using SMOTE
                if smote_instance is not None:
                    search.fit(X_train_full, y_train_full)
                else:
                    search.fit(X_train_full, y_train_full, 
                            classifier__sample_weight=sample_weights if sample_weights is not None else None)
                self.model = search.best_estimator_
                
                # Save search results
                search_results_df = pd.DataFrame(search.cv_results_)
                search_results_df.to_csv(f"{self.paths['metrics']}/search_results.csv", index=False)
                pd.DataFrame([search.best_params_]).to_csv(f"{self.paths['metrics']}/best_params.csv", index=False)
                self.logger.info(f"Best parameters: {search.best_params_}")
            
            # Cross-validation with sample weights (except when using SMOTE)
            self.logger.info("Starting cross-validation...")
            tscv = TimeSeriesSplit(n_splits=5)
            self.cv_metrics = []
            
            for fold, (train_idx, test_idx) in enumerate(tscv.split(X_train_full)):
                self.logger.info(f"Processing fold {fold + 1}/{tscv.n_splits}")
                
                # Clone model for each fold
                fold_model = clone(self.model)
                
                # Get sample weights for this fold (skip if using SMOTE)
                fold_sample_weights = None
                if smote_instance is None and sample_weights is not None:
                    fold_sample_weights = sample_weights[train_idx]
                
                # Train and evaluate
                if self.task_type == 'classification':
                    class_names = ['away_win', 'draw', 'home_win'] if self.le else None
                    metrics = evaluate_classification_model(
                        fold_model,
                        X_train_full.iloc[train_idx], X_train_full.iloc[test_idx],
                        y_train_full.iloc[train_idx], y_train_full.iloc[test_idx],
                        self.logger,
                        fold=fold,
                        class_names=class_names,
                        sample_weight=fold_sample_weights,
                        use_smote=(smote_instance is not None),
                        smote_strategy=smote_strategy
                    )
                else:
                    metrics = evaluate_regression_model(
                        fold_model,
                        X_train_full.iloc[train_idx], X_train_full.iloc[test_idx],
                        y_train_full.iloc[train_idx], y_train_full.iloc[test_idx],
                        self.logger,
                        fold=fold
                    )
                
                self.cv_metrics.append(metrics)
            
            # Final training on full training period
            self.logger.info("Training final model on full training period...")
            if smote_instance is None and sample_weights is not None:
                self.model.fit(X_train_full, y_train_full, classifier__sample_weight=sample_weights)
            else:
                self.model.fit(X_train_full, y_train_full)
            
            # Evaluate on holdout period
            self.logger.info("Evaluating on holdout period...")
            if self.task_type == 'classification':
                y_pred_holdout = self.model.predict(X_holdout)
                holdout_accuracy = accuracy_score(y_holdout, y_pred_holdout)
                
                # Generate comprehensive holdout metrics
                class_names = ['away_win', 'draw', 'home_win'] if self.le else None
                holdout_report = classification_report(
                    y_holdout, y_pred_holdout, 
                    target_names=class_names, 
                    output_dict=True,
                    zero_division=0
                )
                
                self.final_metrics = {
                    'model_type': model_type,
                    'feature_selection_method': feature_selection_method,
                    'holdout_accuracy': holdout_accuracy,
                    'holdout_precision_macro': holdout_report['macro avg']['precision'],
                    'holdout_recall_macro': holdout_report['macro avg']['recall'],
                    'holdout_f1_macro': holdout_report['macro avg']['f1-score'],
                    'train_samples': len(X_train_full),
                    'holdout_samples': len(X_holdout),
                    'holdout_ratio': holdout_ratio,
                    'handled_class_imbalance': handle_class_imbalance,
                    'imbalance_method': imbalance_config['method'] if imbalance_config else 'none',
                    'class_distribution': imbalance_config['class_distribution'] if imbalance_config else None,
                    'used_sample_weights': sample_weights is not None,
                    'used_smote': smote_instance is not None,
                    'smote_strategy': str(smote_strategy) if smote_instance is not None else None,
                }
                
                # Add class-specific metrics if available
                if 'draw' in holdout_report:
                    self.final_metrics['draw_recall'] = holdout_report['draw']['recall']
                    self.final_metrics['draw_precision'] = holdout_report['draw']['precision']
                
                self.logger.info(f"Holdout Accuracy: {holdout_accuracy:.4f}")
            else:
                y_pred_holdout = self.model.predict(X_holdout)
                holdout_rmse = np.sqrt(mean_squared_error(y_holdout, y_pred_holdout))
                holdout_r2 = r2_score(y_holdout, y_pred_holdout)
                
                self.final_metrics = {
                    'model_type': model_type,
                    'feature_selection_method': feature_selection_method,
                    'holdout_rmse': holdout_rmse,
                    'holdout_r2': holdout_r2,
                    'holdout_mae': mean_absolute_error(y_holdout, y_pred_holdout),
                    'train_samples': len(X_train_full),
                    'holdout_samples': len(X_holdout),
                    'holdout_ratio': holdout_ratio,
                    'handled_class_imbalance': handle_class_imbalance,
                    'used_sample_weights': sample_weights is not None,
                    'used_smote': smote_instance is not None,
                    'smote_strategy': str(smote_strategy) if smote_instance is not None else None,
                }
                
                self.logger.info(f"Holdout RMSE: {holdout_rmse:.4f}, R²: {holdout_r2:.4f}")
            
            # Save model and artifacts
            self.save_artifacts(X_train_full, y_train_full, X_holdout, y_holdout, holdout_report, target_col)

            self.generate_visualizations(X_holdout, y_holdout, target_col)
            
            self.logger.info("Training completed successfully!")
            self.print_model_report(model_type, feature_selection_method, target_col)

            return self.model, X_holdout, y_holdout
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}", exc_info=True)
            raise

    def train_hierarchical(self, df, target_col='outcome', model_type='random_forest',
                        feature_selection_method='importance', top_n_features=30,
                        use_bayesian=False, bayesian_iter=50, use_grid_search=False,
                        use_random_search=False, random_search_iter=50, load_params=False,
                        holdout_ratio=0.2, handle_class_imbalance=True, draw_threshold=0.25,
                        draw_model_type='logistic_regression', win_model_type='random_forest'):
        """
        Train using hierarchical two-stage approach
        
        Parameters:
        -----------
        draw_threshold : float, default=0.25
            Threshold for draw prediction (lower values catch more draws)
        draw_model_type : str, default='logistic_regression'
            Model type for draw detection stage
        win_model_type : str, default='random_forest'
            Model type for home/away win prediction stage
        """
        try:
            # Setup directories and logging
            self.setup_directories(f"{model_type}_hierarchical", feature_selection_method)
            self.logger.info(f"Starting hierarchical training with draw_model: {draw_model_type}, win_model: {win_model_type}")
            
            # Prepare data
            X, y = self.prepare_data(df, target_col)
            
            # Create time series holdout set
            n_samples = len(X)
            holdout_size = int(n_samples * holdout_ratio)
            
            X_train_full = X.iloc[:-holdout_size]
            y_train_full = y.iloc[:-holdout_size]
            X_holdout = X.iloc[-holdout_size:]
            y_holdout = y.iloc[-holdout_size:]
            
            self.logger.info(f"Time Series Split - Training: {X_train_full.shape[0]}, Holdout: {X_holdout.shape[0]}")
            
            # Build preprocessing pipeline (once)
            base_pipeline = self.build_pipeline(model_type, feature_selection_method, top_n_features)
            
            # Get models for each stage
            _, draw_default_params = get_model_params(draw_model_type, 'classification')
            draw_model_class = get_model_class(draw_model_type, 'classification')
            
            # For XGBoost, ensure we use binary objective
            if draw_model_type == 'xgboost':
                draw_default_params = draw_default_params.copy()
                draw_default_params['objective'] = 'binary:logistic'
            
            draw_model = draw_model_class(**draw_default_params)
            
            # Get win model
            _, win_default_params = get_model_params(win_model_type, 'classification')
            win_model_class = get_model_class(win_model_type, 'classification')
            
            if win_model_type == 'xgboost':
                win_default_params = win_default_params.copy()
                win_default_params['objective'] = 'binary:logistic'
            
            win_model = win_model_class(**win_default_params)
            
            # Create hierarchical classifier
            hierarchical_model = HierarchicalClassifier(
                win_model=win_model,  # For win prediction
                draw_threshold=draw_threshold
            )
            
            # Override the draw classifier with specialized model
            hierarchical_model.draw_classifier = draw_model
            
            # Create full pipeline
            full_pipeline = Pipeline([
                *base_pipeline.steps,
                ('hierarchical_classifier', hierarchical_model)
            ])
            
            # Cross-validation
            self.logger.info("Starting hierarchical cross-validation...")
            tscv = TimeSeriesSplit(n_splits=5)
            self.cv_metrics = []
            
            for fold, (train_idx, test_idx) in enumerate(tscv.split(X_train_full)):
                self.logger.info(f"Processing fold {fold + 1}/{tscv.n_splits}")
                
                # Clone model for each fold
                fold_model = clone(full_pipeline)
                
                # Train and evaluate
                class_names = ['away_win', 'draw', 'home_win'] if self.le else None
                metrics = evaluate_classification_model(
                    fold_model,
                    X_train_full.iloc[train_idx], X_train_full.iloc[test_idx],
                    y_train_full.iloc[train_idx], y_train_full.iloc[test_idx],
                    self.logger,
                    fold=fold,
                    class_names=class_names
                )
                
                self.cv_metrics.append(metrics)
            
            # Final training on full training period
            self.logger.info("Training final hierarchical model on full training period...")
            full_pipeline.fit(X_train_full, y_train_full)
            self.model = full_pipeline
            
            # Evaluate on holdout period
            self.logger.info("Evaluating hierarchical model on holdout period...")
            y_pred_holdout = self.model.predict(X_holdout)
            holdout_accuracy = accuracy_score(y_holdout, y_pred_holdout)
            
            # Generate comprehensive holdout metrics
            class_names = ['away_win', 'draw', 'home_win'] if self.le else None
            holdout_report = classification_report(
                y_holdout, y_pred_holdout, 
                target_names=class_names, 
                output_dict=True,
                zero_division=0
            )
            
            self.final_metrics = {
                'model_type': f"{model_type}_hierarchical",
                'feature_selection_method': feature_selection_method,
                'draw_model_type': draw_model_type,
                'win_model_type': win_model_type,
                'draw_threshold': draw_threshold,
                'holdout_accuracy': holdout_accuracy,
                'holdout_precision_macro': holdout_report['macro avg']['precision'],
                'holdout_recall_macro': holdout_report['macro avg']['recall'],
                'holdout_f1_macro': holdout_report['macro avg']['f1-score'],
                'train_samples': len(X_train_full),
                'holdout_samples': len(X_holdout),
                'holdout_ratio': holdout_ratio,
            }
            
            # Add draw metrics if available
            if 'draw' in holdout_report:
                self.final_metrics['draw_recall'] = holdout_report['draw']['recall']
                self.final_metrics['draw_precision'] = holdout_report['draw']['precision']
            
            self.logger.info(f"Hierarchical Holdout Accuracy: {holdout_accuracy:.4f}")
            if 'draw_recall' in self.final_metrics:
                self.logger.info(f"Hierarchical Draw Recall: {self.final_metrics['draw_recall']:.4f}")
            
            # Save model and artifacts
            self.save_artifacts(X_train_full, y_train_full, X_holdout, y_holdout)
            
            self.logger.info("Hierarchical training completed successfully!")
            return self.model, X_holdout, y_holdout
            
        except Exception as e:
            self.logger.error(f"Hierarchical training failed: {str(e)}", exc_info=True)
            raise

    
    def save_artifacts(self, X_train, y_train, X_test, y_test, holdout_report, target_col=None):
        """Save all model artifacts with target-specific naming"""
        self.logger.info("Saving model artifacts...")
        
        # Include target in model filename
        if target_col:
            model_filename = f"{target_col}_{self.final_metrics['model_type']}_model.pkl"
        else:
            model_filename = f"{self.final_metrics['model_type']}_model.pkl"
        
        model_path = f"{self.paths['models']}/{model_filename}"
        joblib.dump(self.model, model_path)
        
        # Save label encoder for classification tasks
        if self.task_type == 'classification' and self.le is not None:
            encoder_filename = f"{target_col}_label_encoder.pkl" if target_col else "label_encoder.pkl"
            joblib.dump(self.le, f"{self.paths['models']}/{encoder_filename}")
        
        # Save metrics
        cv_df = pd.DataFrame(self.cv_metrics)
        cv_df.to_csv(f"{self.paths['metrics']}/cross_validation_results.csv", index=False)
        
        final_metrics_df = pd.DataFrame([self.final_metrics])
        final_metrics_df.to_csv(f"{self.paths['metrics']}/final_metrics.csv", index=False)

        # Save holdout classification report if available
        if holdout_report:
            holdout_report_df = pd.DataFrame(holdout_report).transpose()
            holdout_report_filename =("test_classification_report.csv")
            holdout_report_df.to_csv(f"{self.paths['metrics']}/{holdout_report_filename}", index=True)
        
        # Save feature importances if available
        try:
            feature_names = get_feature_names_from_pipeline(self.model)
            estimator = self.model.named_steps.get('classifier', 
                        self.model.named_steps.get('regressor', self.model))
            
            if hasattr(estimator, 'feature_importances_'):
                importances = estimator.feature_importances_
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                importance_filename = ("feature_importances.csv")
                importance_df.to_csv(f"{self.paths['metrics']}/{importance_filename}", index=False)
        except Exception as e:
            self.logger.warning(f"Could not save feature importances: {str(e)}")
        
        self.logger.info("Artifacts saved successfully!")

    
    def predict_proba(self, X):
        """Make probability predictions (for classification only)"""
        if self.task_type != 'classification':
            self.logger.error("Probability predictions only available for classification tasks")
            raise ValueError("Probability predictions only available for classification tasks")
        
        if self.model is None:
            self.logger.error("Model not trained yet. Call train() first.")
            raise ValueError("Model not trained yet. Call train() first.")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            self.logger.warning("Model does not support probability predictions")
            return None
    
    def get_feature_importance(self):
        """Get feature importance from the trained model"""
        if self.model is None:
            self.logger.error("Model not trained yet. Call train() first.")
            raise ValueError("Model not trained yet. Call train() first.")
        
        try:
            feature_names = get_feature_names_from_pipeline(self.model)
            estimator = self.model.named_steps.get('classifier', 
                         self.model.named_steps.get('regressor', self.model))
            
            if hasattr(estimator, 'feature_importances_'):
                importances = estimator.feature_importances_
                return pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False)
            else:
                self.logger.warning("Model does not support feature importance")
                return None
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {str(e)}")
            return None
    
    def score(self, X, y):
        """Score the model on given data"""
        if self.model is None:
            self.logger.error("Model not trained yet. Call train() first.")
            raise ValueError("Model not trained yet. Call train() first.")
        
        return self.model.score(X, y)
    
    def generate_visualizations(self, X_test, y_test, target_col='outcome'):
        """
        Generate visualizations for the trained model
        """
        if self.model is None:
            self.logger.error("Model not trained yet. Call train() first.")
            raise ValueError("Model not trained yet. Call train() first.")
        
        if self.paths is None:
            self.logger.error("Paths not set up. Call train() first.")
            raise ValueError("Paths not set up. Call train() first.")
        
        # Create a wrapper that has access to both the model and the feature importance method
        class PipelineWrapper:
            def __init__(self, model, pipeline_instance):
                self.model = model
                self.final_metrics = pipeline_instance.final_metrics
                self.task_type = pipeline_instance.task_type
                self.le = getattr(pipeline_instance, 'le', None)
                self._pipeline_instance = pipeline_instance  # Store reference to the TrainPipeline instance
                
            def get_feature_importance(self):
                # Delegate to the TrainPipeline instance's method
                return self._pipeline_instance.get_feature_importance()
                
            # Add predict and predict_proba methods that delegate to the model
            def predict(self, X):
                return self.model.predict(X)
                
            def predict_proba(self, X):
                if hasattr(self.model, 'predict_proba'):
                    return self.model.predict_proba(X)
                raise AttributeError("Model does not have predict_proba method")
                
            def transform(self, X):
                if hasattr(self.model, 'transform'):
                    return self.model.transform(X)
                raise AttributeError("Model does not have transform method")
        
        # Create the wrapper
        wrapper = PipelineWrapper(self.model, self)
        
        # Generate visualizations using the wrapper
        generate_basic_visualizations(
            pipeline=wrapper,  # Pass the wrapper instead of just the model
            X_test=X_test,
            y_test=y_test,
            artifact_paths=self.paths,
            target_col=target_col,
            # Pass the metadata from the TrainPipeline instance
            model_type=self.final_metrics['model_type'],
            feature_selection_method=self.final_metrics['feature_selection_method'],
            task_type=self.task_type,
            le=getattr(self, 'le', None),
            final_metrics=self.final_metrics
        )


    def print_model_report(self, model_type, feature_selection_method, target_col=None):
        """
        Print comprehensive model report for both classification and regression
        
        Args:
            model_type: Type of model
            feature_selection_method: Feature selection method
            target_col: Target column name
        """
        if target_col:
            base_path = f"artifacts/{target_col}/{model_type}/{feature_selection_method}/metrics"
        else:
            base_path = f"artifacts/{model_type}/{feature_selection_method}/metrics"
        
        try:
            # Load metrics
            final_metrics = pd.read_csv(f"{base_path}/final_metrics.csv")
            cv_results = pd.read_csv(f"{base_path}/cross_validation_results.csv")
            test_report = pd.read_csv(f"{base_path}/test_classification_report.csv", index_col=0)
            
            task_type = 'classification' if 'holdout_accuracy' in final_metrics.columns else 'regression'
            
            print(f"\n{'='*80}")
            print(f"MODEL PERFORMANCE REPORT: {target_col.upper() if target_col else 'MODEL'}")
            print(f"Model: {model_type.upper()} | Features: {feature_selection_method.upper()} | Task: {task_type.upper()}")
            print(f"{'='*80}")
            
            if task_type == 'classification':
                self.print_classification_report(test_report, cv_results, base_path)
            else:
                self.print_regression_report(final_metrics, cv_results)
                
            print(f"{'='*80}")
            
        except FileNotFoundError as e:
            print(f"\nError: Required files not found in {base_path}")
            print(f"Missing file: {str(e)}")
        except Exception as e:
            print(f"\nError generating report: {str(e)}")
            import traceback
            traceback.print_exc()

    def print_classification_report(self, final_metrics, cv_results, base_path):
        """Print classification-specific report"""
        # Cross-validation summary
        print("\nCROSS-VALIDATION PERFORMANCE:")
        print("-" * 60)
        print(f"Average Accuracy: {cv_results['test_accuracy'].mean():.3f} (±{cv_results['test_accuracy'].std():.3f})")
        print(f"Average F1 Macro: {cv_results['f1_macro'].mean():.3f} (±{cv_results['f1_macro'].std():.3f})")
        
        # Holdout performance
        print("\nHOLDOUT SET EVALUATION:")
        print("-" * 60)
        print(f"Accuracy:    {final_metrics.iloc[0]['accuracy']:.3f}")
        print(f"F1 Macro:    {final_metrics.iloc[0]['f1-score']:.3f}")
        print(f"Precision:   {final_metrics.iloc[0]['precision']:.3f}")
        print(f"Recall:      {final_metrics.iloc[0]['recall']:.3f}")
        
        # Try to load detailed classification report
        try:
            test_report = pd.read_csv(f"{base_path}/test_classification_report.csv", index_col=0)
            print("\nDETAILED CLASSIFICATION REPORT:")
            print("-" * 60)
            print(f"{'Class':<15}{'Precision':>10}{'Recall':>10}{'F1-Score':>10}{'Support':>10}")
            print("-" * 60)
            
            for class_name in test_report.index:
                if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                    row = test_report.loc[class_name]
                    print(f"{class_name:<15}{row['precision']:>10.3f}{row['recall']:>10.3f}{row['f1-score']:>10.3f}{int(row['support']):>10}")
        except:
            print("\nDetailed classification report not available")

    def print_regression_report(self, final_metrics, cv_results):
        """Print regression-specific report"""
        # Cross-validation summary
        print("\nCROSS-VALIDATION PERFORMANCE:")
        print("-" * 60)
        print(f"Average RMSE: {cv_results['rmse'].mean():.3f} (±{cv_results['rmse'].std():.3f})")
        print(f"Average R²:   {cv_results['r2'].mean():.3f} (±{cv_results['r2'].std():.3f})")
        print(f"Average MAE:  {cv_results['mae'].mean():.3f} (±{cv_results['mae'].std():.3f})")
        
        # Holdout performance
        print("\nHOLDOUT SET EVALUATION:")
        print("-" * 60)
        print(f"RMSE: {final_metrics.iloc[0]['holdout_rmse']:.3f}")
        print(f"R²:   {final_metrics.iloc[0]['holdout_r2']:.3f}")
        print(f"MAE:  {final_metrics.iloc[0]['holdout_mae']:.3f}")
        
        # Sample information
        print(f"\nSAMPLES:")
        print("-" * 60)
        print(f"Training:  {final_metrics.iloc[0]['train_samples']}")
        print(f"Holdout:   {final_metrics.iloc[0]['holdout_samples']}")
        print(f"Total:     {final_metrics.iloc[0]['train_samples'] + final_metrics.iloc[0]['holdout_samples']}")        
    
    # ==================== TASK TYPE DETECTION ====================


def detect_task_type(y):
    """
    Detect if the task is classification or regression based on target variable
    
    Parameters:
    -----------
    y : array-like
        Target variable
        
    Returns:
    --------
    str : 'classification' or 'regression'
    """
    # If target is not numeric, it's classification
    if not pd.api.types.is_numeric_dtype(y):
        return 'classification'
    
    # Convert to numeric to handle string numbers
    y_numeric = pd.to_numeric(y, errors='coerce')
    
    # Remove NaN values for analysis
    y_clean = y_numeric.dropna()
    
    if len(y_clean) == 0:
        return 'classification'  # Default to classification if no valid numeric data
    
    # If target has many unique values (more than 20) and wide range, it's regression
    unique_values = len(y_clean.unique())
    total_values = len(y_clean)
    
    # Heuristic for regression: many unique values or wide value range
    value_range = y_clean.max() - y_clean.min()
    
    # Specific case for sports scores: if range > 5 and unique values > 10, treat as regression
    if (unique_values > 15 or value_range > 8) and unique_values / total_values > 0.1:
        return 'regression'
    
    # If target represents counts (like goals, cards, corners) with reasonable range, treat as regression
    if value_range >= 5 and unique_values >= 8:  # At least 8 different values spanning 5+ units
        return 'regression'
    
    # Default to classification for limited discrete values
    return 'classification'

# ==================== FEATURE SELECTOR ====================
def get_feature_selector(feature_selection_method, model_type, top_n_features, tscv, 
                        target_variance=0.9, task_type='classification'):
    """
    Return feature selector with appropriate scoring for task type
    
    Parameters:
    -----------
    feature_selection_method : str
        Method for feature selection
    model_type : str
        Type of model being used
    top_n_features : int
        Number of top features to select
    tscv : TimeSeriesSplit
        Time series cross-validation object
    target_variance : float, default=0.9
        Target variance for PCA
    task_type : str, default='classification'
        Type of ML task
        
    Returns:
    --------
    sklearn feature selector object
    """
    logger = logging.getLogger(__name__)
    
    # Get an instantiated model for the feature selector
    model_class = get_model_class(model_type, task_type)
    estimator = model_class()  # ✅ Instantiate the model
    
    # Choose appropriate score function based on task type
    if task_type == 'classification':
        score_func = f_classif
        mutual_func = mutual_info_classif
        scoring_metric = 'accuracy'
    else:
        score_func = f_regression
        mutual_func = mutual_info_regression
        scoring_metric = 'r2'
    
    selectors = {
        'importance': SelectFromModel(
            RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=50) 
            if task_type == 'classification' 
            else RandomForestRegressor(random_state=42, n_estimators=50),
            max_features=top_n_features,
            threshold=-np.inf
        ),
        'rfe': RFE(
            estimator=estimator,
            n_features_to_select=top_n_features,
            step=0.1,
            verbose=0
        ),
        'rfecv': RFECV(
            estimator=estimator,
            step=1,
            cv=tscv,
            scoring=scoring_metric,
            min_features_to_select=min(10, top_n_features),
            n_jobs=-1
        ),
        'pca': PCA(
            n_components=target_variance, 
            svd_solver='full',
            random_state=42
        ),
        'anova': SelectKBest(
            score_func=score_func, 
            k=min(top_n_features, 100)  # Limit to avoid memory issues
        ),
        'mutual_info': SelectKBest(
            score_func=mutual_func,
            k=min(top_n_features, 50)  # Mutual info can be computationally expensive
        ),
        'sequential': SequentialFeatureSelector(
            estimator,
            n_features_to_select=top_n_features,
            direction='forward',
            cv=tscv,
            n_jobs=-1,
            scoring=scoring_metric
        )
    }
    
    if feature_selection_method not in selectors:
        error_msg = f"Unknown feature selection method: {feature_selection_method}. Available options: {list(selectors.keys())}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info(f"Using feature selection method: {feature_selection_method} with top {top_n_features} features")
    return selectors[feature_selection_method]


# ==================== MODEL CLASS SELECTION ====================
def get_model_class(model_type, task_type='classification'):
    """
    Return the appropriate model class based on model_type and task_type
    
    Parameters:
    -----------
    model_type : str
        Type of model to use
    task_type : str, default='classification'
        Type of ML task
        
    Returns:
    --------
    sklearn model class
    """
    logger = logging.getLogger(__name__)
    
    if task_type == 'classification':
        model_classes = {
            'random_forest': RandomForestClassifier,
            'logistic_regression': LogisticRegression,
            'xgboost': XGBClassifier,
            'gradient_boosting': GradientBoostingClassifier,
            'svm': SVC
        }
    else:  # regression
        model_classes = {
            'random_forest': RandomForestRegressor,
            'linear_regression': Ridge,
            'xgboost': XGBRegressor,
            'gradient_boosting': GradientBoostingRegressor,
            'svm': SVR
        }
    
    if model_type not in model_classes:
        error_msg = f"Unknown model type: {model_type}. Available options: {list(model_classes.keys())}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info(f"Selected model class: {model_type} for {task_type}")
    return model_classes[model_type]


# ==================== MODEL PARAMETERS ====================
def get_model_params(model_type, task_type='classification'):
    """
    Get model parameters based on task type
    
    Parameters:
    -----------
    model_type : str
        Type of model to use
    task_type : str, default='classification'
        Type of ML task
        
    Returns:
    --------
    tuple : (search_params, default_params)
    """
    logger = logging.getLogger(__name__)
    
    if task_type == 'classification':
        search_params = CLASSIFICATION_MODEL_PARAMS.get(model_type, {})
        default_params = CLASSIFICATION_DEFAULT_PARAMS.get(model_type, {})
    else:
        search_params = REGRESSION_MODEL_PARAMS.get(model_type, {})
        default_params = REGRESSION_DEFAULT_PARAMS.get(model_type, {})
    
    if not search_params or not default_params:
        warning_msg = f"No parameters found for model type: {model_type}. Using empty parameters."
        logger.warning(warning_msg)
        return {}, {}
    
    logger.info(f"Retrieved parameters for model: {model_type}")
    return search_params, default_params


# ==================== CLASSIFICATION EVALUATION ====================


def evaluate_classification_model(model, X_train, X_test, y_train, y_test, logger, 
                                fold=None, class_names=None, model_type=None, 
                                feature_selection_method=None, sample_weight=None, use_smote=False, smote_strategy=None):
    """
    Evaluate classification model with comprehensive metrics
    
    Parameters:
    -----------
    model : sklearn model
        Trained model to evaluate
    X_train : array-like
        Training features
    X_test : array-like
        Test features
    y_train : array-like
        Training labels
    y_test : array-like
        Test labels
    logger : logging.Logger
        Logger instance
    fold : int, default=None
        Fold number for cross-validation
    class_names : list, default=None
        Names of classes for reporting
    model_type : str, default=None
        Type of model used
    feature_selection_method : str, default=None
        Feature selection method used
    sample_weight : array-like, default=None
        Sample weights for handling class imbalance
        
    Returns:
    --------
    dict : Evaluation metrics
    """
    metrics = {
        'fold': fold + 1 if fold is not None else None,
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test),
        'used_sample_weights': sample_weight is not None
    }
    
    try:
        
        # Apply SMOTE/oversampling
        if use_smote and smote_strategy:
            try:
                # Check if we have enough samples for SMOTE
                class_counts = np.bincount(y_train)
                min_class_count = min(class_counts)
                
                if min_class_count < 2:
                    logger.warning(f"Not enough samples in minority class ({min_class_count}) for SMOTE")
                    metrics['smote_skipped'] = "Insufficient minority class samples"
                else:
                    sample_weight = None  # Disable sample weights when using SMOTE
                    logger.info("Using SMOTE oversampling - sample weights disabled to avoid size mismatch")
                    
                    # Calculate safe k_neighbors value
                    safe_k_neighbors = min(5, min_class_count - 1)
                    
                    smote = SMOTE(
                        sampling_strategy=smote_strategy,
                        random_state=42 + fold if fold else 42,
                        k_neighbors=safe_k_neighbors
                    )
                    X_train, y_train = smote.fit_resample(X_train, y_train)
                        
                    logger.info(f"SMOTE applied: {X_train.shape[0]} samples (from {metrics['n_train_samples']})")
                    metrics['used_smote'] = True
                    metrics['smote_strategy'] = str(smote_strategy)
                    metrics['n_train_samples_after_smote'] = len(X_train)
            except Exception as e:
                logger.warning(f"SMOTE oversampling failed: {str(e)}")
                metrics['smote_error'] = str(e)

        # Fit the model with appropriate parameters
        if sample_weight is not None:
            logger.info(f"Using sample weights: {sample_weight.shape if hasattr(sample_weight, 'shape') else 'custom weights'}")
            
            # Try different approaches for passing sample weights
            try:
                # First try: standard sample_weight parameter
                if hasattr(model, 'fit'):
                    model.fit(X_train, y_train, sample_weight=sample_weight)
            except TypeError as e:
                logger.warning(f"Standard sample_weight failed: {str(e)}. Trying pipeline-specific approach.")
                try:
                    # Second try: for pipeline with classifier step
                    if hasattr(model, 'steps'):
                        model.fit(X_train, y_train, classifier__sample_weight=sample_weight)
                    else:
                        # Final fallback: fit without sample weights
                        model.fit(X_train, y_train)
                        metrics['sample_weight_warning'] = "Model doesn't support sample_weight parameter"
                except Exception as inner_e:
                    logger.warning(f"Pipeline sample_weight also failed: {str(inner_e)}. Fitting without sample weights.")
                    model.fit(X_train, y_train)
                    metrics['sample_weight_error'] = str(inner_e)
        else:
            # Standard fitting without sample weights
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
                logger.warning(f"Probability metrics failed: {str(e)}")
        
        # Confusion matrix
        try:
            cm = confusion_matrix(y_test, y_pred)
            metrics['confusion_matrix'] = cm.tolist()
            
            # Calculate class distribution for imbalance analysis
            class_distribution = {class_names[i]: count for i, count in enumerate(np.bincount(y_test)) if count > 0}
            metrics['class_distribution'] = class_distribution
            
        except Exception as e:
            metrics['confusion_matrix_error'] = str(e)
            logger.warning(f"Confusion matrix failed: {str(e)}")
        
        # Log class imbalance metrics if sample weights were used
        if sample_weight is not None and class_names:
            draw_idx = class_names.index('draw') if 'draw' in class_names else None
            if draw_idx is not None and f'recall_draw' in metrics:
                logger.info(f"Draw recall with sample weights: {metrics['recall_draw']:.4f}")
        
        if fold is not None:
            logger.info(f"Fold {fold + 1} - Test Accuracy: {metrics['test_accuracy']:.4f} - F1 Macro: {metrics['f1_macro']:.4f}")
        else:
            logger.info(f"Test Accuracy: {metrics['test_accuracy']:.4f} - F1 Macro: {metrics['f1_macro']:.4f}")
            
    except Exception as e:
        error_msg = f"Model evaluation failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        metrics['error'] = error_msg
    
    return metrics

# ==================== REGRESSION EVALUATION ====================
def evaluate_regression_model(model, X_train, X_test, y_train, y_test, logger, 
                            fold=None, model_type=None, feature_selection_method=None):
    """
    Evaluate regression model with comprehensive metrics
    
    Parameters:
    -----------
    model : sklearn model
        Trained model to evaluate
    X_train : array-like
        Training features
    X_test : array-like
        Test features
    y_train : array-like
        Training labels
    y_test : array-like
        Test labels
    logger : logging.Logger
        Logger instance
    fold : int, default=None
        Fold number for cross-validation
    model_type : str, default=None
        Type of model used
    feature_selection_method : str, default=None
        Feature selection method used
        
    Returns:
    --------
    dict : Evaluation metrics
    """
    metrics = {
        'fold': fold + 1 if fold is not None else None,
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test)
    }
    
    try:
        # Train and predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_train = model.predict(X_train)
        
        # Calculate regression metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        explained_variance = explained_variance_score(y_test, y_pred)
        
        # Training metrics
        train_mse = mean_squared_error(y_train, y_pred_train)
        train_rmse = np.sqrt(train_mse)
        train_r2 = r2_score(y_train, y_pred_train)
        
        metrics.update({
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'explained_variance': explained_variance,
            'train_mse': train_mse,
            'train_rmse': train_rmse,
            'train_r2': train_r2,
        })
        
        # Calculate additional metrics if needed
        try:
            metrics['mape'] = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        except:
            metrics['mape'] = None  # Handle division by zero
        
        if fold is not None:
            logger.info(f"Fold {fold + 1} - Test RMSE: {rmse:.4f} - R²: {r2:.4f}")
        else:
            logger.info(f"Test RMSE: {rmse:.4f} - R²: {r2:.4f}")
            
    except Exception as e:
        error_msg = f"Regression model evaluation failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        metrics['error'] = error_msg
    
    return metrics



    
# ==================== VISUALIZATION FUNCTIONS ====================

def generate_basic_visualizations(pipeline, X_test, y_test, artifact_paths, target_col=None, 
                                 model_type=None, feature_selection_method=None, task_type=None, 
                                 le=None, final_metrics=None):
    """
    Generate basic visualizations for model evaluation with target-specific naming
    """
    logger = logging.getLogger(__name__)
    logger.info("Generating basic visualizations...")
    
    try:
        model = pipeline  # Now pipeline is the wrapper object
        
        # USE THE PASSED PARAMETERS INSTEAD OF TRYING TO GET THEM FROM PIPELINE
        model_type = model_type or 'unknown'
        feature_selection_method = feature_selection_method or 'unknown'
        task_type = task_type or 'unknown'
        
        # Keep as DataFrames for pipeline compatibility
        X_test_df = X_test
        y_test_array = y_test.values if hasattr(y_test, 'values') else np.array(y_test)
        
        # Create title with target information
        if target_col:
            main_title = f'{target_col.upper()} - {model_type.title()} ({feature_selection_method}) Evaluation'
            filename_prefix = f"{target_col}_"
        else:
            main_title = f'{model_type.title()} ({feature_selection_method}) Evaluation'
            filename_prefix = ""
        
        # Get class names for classification - USE THE PASSED le PARAMETER
        class_names = None
        if task_type == 'classification' and le is not None:
            class_names = list(le.classes_)
        
        # Create main evaluation plot
        plt.figure(figsize=(15, 10))
        plt.suptitle(main_title, y=0.98, fontsize=16, fontweight='bold')
        
        # Get predictions using the model pipeline
        try:
            y_pred = model.predict(X_test_df)
            y_pred_array = np.array(y_pred)
        except Exception as e:
            logger.error(f"Model prediction failed: {e}")
            # Create error visualization
            plt.text(0.5, 0.5, 'Prediction failed\nCannot generate visualizations', 
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
            plt.tight_layout()
            plt.savefig(f"{artifact_paths['plots']}/{filename_prefix}model_evaluation_error.png", 
                       bbox_inches='tight', dpi=300, facecolor='white')
            plt.close()
            return
        
        if task_type == 'classification':
            # Classification-specific visualizations
            
            # 1. Confusion Matrix
            plt.subplot(2, 2, 1)
            try:
                cm = confusion_matrix(y_test_array, y_pred_array)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=class_names, yticklabels=class_names)
                plt.title('Confusion Matrix', fontweight='bold')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
            except Exception as e:
                plt.text(0.5, 0.5, 'Confusion matrix\nfailed', 
                        ha='center', va='center', transform=plt.gca().transAxes)
                plt.title('Confusion Matrix', fontweight='bold')
            
            # 2. Feature Importance - USE THE get_feature_importance METHOD FROM WRAPPER
            plt.subplot(2, 2, 2)
            try:
                # Use the get_feature_importance method from the wrapper
                if hasattr(pipeline, 'get_feature_importance'):
                    feature_importances = pipeline.get_feature_importance()
                else:
                    feature_importances = None
                
                if feature_importances is not None and not feature_importances.empty:
                    top_features = feature_importances.head(10)
                    plt.barh(range(len(top_features)), top_features['importance'])
                    plt.yticks(range(len(top_features)), top_features['feature'])
                    plt.title('Top 10 Feature Importances', fontweight='bold')
                    plt.xlabel('Importance')
                else:
                    plt.text(0.5, 0.5, 'Feature importances\nnot available', 
                            ha='center', va='center', transform=plt.gca().transAxes)
                    plt.title('Feature Importances', fontweight='bold')
            except Exception as e:
                logger.warning(f"Error generating feature importance: {str(e)}")
                plt.text(0.5, 0.5, f'Error generating\nfeature importance', 
                        ha='center', va='center', transform=plt.gca().transAxes)
                plt.title('Feature Importances', fontweight='bold')
            
            # 3. ROC Curve (moved to position 3)
            plt.subplot(2, 2, 3)
            try:
                # Get probabilities safely
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test_df)
                    
                    # Handle different return types from predict_proba
                    if isinstance(y_proba, list):
                        y_proba = np.array(y_proba)
                    
                    # Ensure y_proba is 2D array for multi-class classification
                    if len(y_proba.shape) == 1:
                        y_proba = y_proba.reshape(-1, 1)
                        if class_names and len(class_names) > 1:
                            y_proba = np.column_stack([1 - y_proba, y_proba])
                    
                    n_classes = y_proba.shape[1] if len(y_proba.shape) > 1 else 1
                    
                    if n_classes == 2:
                        fpr, tpr, _ = roc_curve(y_test_array, y_proba[:, 1])
                        roc_auc = auc(fpr, tpr)
                        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
                    else:
                        y_test_bin = label_binarize(y_test_array, classes=np.unique(y_test_array))
                        for i in range(n_classes):
                            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
                            roc_auc = auc(fpr, tpr)
                            plt.plot(fpr, tpr, label=f'Class {class_names[i]} (AUC = {roc_auc:.2f})')
                    
                    plt.plot([0, 1], [0, 1], 'k--')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC Curves', fontweight='bold')
                    plt.legend(loc='lower right', fontsize='small')
                else:
                    plt.text(0.5, 0.5, 'Probability predictions\nnot available', 
                            ha='center', va='center', transform=plt.gca().transAxes)
                    plt.title('ROC Curves', fontweight='bold')
            except Exception as e:
                plt.text(0.5, 0.5, 'ROC curve\nnot available', 
                        ha='center', va='center', transform=plt.gca().transAxes)
                plt.title('ROC Curves', fontweight='bold')
            
            # 4. Classification Report Heatmap (NEW - replaces Error Distribution)
            plt.subplot(2, 2, 4)
            try:
                report = classification_report(y_test_array, y_pred_array, 
                                             target_names=class_names, 
                                             output_dict=True)
                metrics_df = pd.DataFrame(report).transpose()
                
                # Select only the classification metrics (exclude accuracy, macro avg, weighted avg)
                if class_names:
                    metrics_to_plot = metrics_df.loc[class_names, ['precision', 'recall', 'f1-score']]
                else:
                    # If no class names, use numeric indices
                    class_indices = [str(i) for i in range(len(np.unique(y_test_array))) if str(i) in metrics_df.index]
                    metrics_to_plot = metrics_df.loc[class_indices, ['precision', 'recall', 'f1-score']]
                
                # Create heatmap
                sns.heatmap(metrics_to_plot, annot=True, fmt='.3f', cmap='YlOrRd', 
                           cbar_kws={'label': 'Score'}, center=0.5, vmin=0, vmax=1)
                plt.title('Classification Metrics by Class', fontweight='bold')
                
            except Exception as e:
                plt.text(0.5, 0.5, 'Classification report\nnot available', 
                        ha='center', va='center', transform=plt.gca().transAxes)
                plt.title('Classification Metrics', fontweight='bold')
        
        else:
            # Regression-specific visualizations
            # Calculate regression metrics for display
            rmse = np.sqrt(mean_squared_error(y_test_array, y_pred_array))
            r2 = r2_score(y_test_array, y_pred_array)
            mae = mean_absolute_error(y_test_array, y_pred_array)
            
            # 1. Actual vs Predicted Scatter Plot with metrics
            plt.subplot(2, 2, 1)
            plt.scatter(y_test_array, y_pred_array, alpha=0.6, s=30)
            max_val = max(np.max(y_test_array), np.max(y_pred_array))
            min_val = min(np.min(y_test_array), np.min(y_pred_array))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            
            # Add metrics text box
            textstr = f'RMSE: {rmse:.2f}\nR²: {r2:.3f}\nMAE: {mae:.2f}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, 
                        fontsize=10, verticalalignment='top', bbox=props)
            
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title('Actual vs Predicted', fontweight='bold')
            plt.grid(True, alpha=0.3)
            
            # 2. Prediction Error Plot (NEW)
            plt.subplot(2, 2, 2)
            errors = y_pred_array - y_test_array
            plt.scatter(y_test_array, errors, alpha=0.6, s=30)
            plt.axhline(y=0, color='r', linestyle='-', linewidth=2)
            
            # Add error statistics lines
            mean_error = np.mean(errors)
            std_error = np.std(errors)
            plt.axhline(y=mean_error, color='g', linestyle='--', label=f'Mean Error: {mean_error:.2f}')
            plt.axhline(y=mean_error + std_error, color='orange', linestyle=':', label='±1 STD')
            plt.axhline(y=mean_error - std_error, color='orange', linestyle=':')
            
            plt.xlabel('Actual Values')
            plt.ylabel('Prediction Error')
            plt.title('Prediction Error Analysis', fontweight='bold')
            plt.legend(fontsize=8)
            plt.grid(True, alpha=0.3)
            
            # 3. Feature Importance
            plt.subplot(2, 2, 3)
            try:
                # Use the get_feature_importance method from the wrapper
                if hasattr(pipeline, 'get_feature_importance'):
                    feature_importances = pipeline.get_feature_importance()
                else:
                    feature_importances = None
                
                if feature_importances is not None and not feature_importances.empty:
                    top_features = feature_importances.head(10)
                    colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
                    bars = plt.barh(range(len(top_features)), top_features['importance'], color=colors)
                    
                    # Add value labels on bars
                    for i, (idx, row) in enumerate(top_features.iterrows()):
                        plt.text(row['importance'] + 0.001, i, f'{row["importance"]:.3f}', 
                                va='center', fontsize=8)
                    
                    plt.yticks(range(len(top_features)), top_features['feature'], fontsize=9)
                    plt.title('Top 10 Feature Importances', fontweight='bold')
                    plt.xlabel('Importance Score')
                    plt.grid(True, alpha=0.3, axis='x')
                else:
                    plt.text(0.5, 0.5, 'Feature importances\nnot available', 
                            ha='center', va='center', transform=plt.gca().transAxes)
                    plt.title('Feature Importances', fontweight='bold')
            except Exception as e:
                plt.text(0.5, 0.5, f'Error generating\nfeature importance', 
                        ha='center', va='center', transform=plt.gca().transAxes)
                plt.title('Feature Importances', fontweight='bold')
            
            # 4. Residuals vs Most Important Feature (NEW)
            plt.subplot(2, 2, 4)
            try:
                if feature_importances is not None and not feature_importances.empty:
                    # Get the most important feature
                    top_feature_name = feature_importances.iloc[0]['feature']
                    
                    # Try to find this feature in the test data
                    if hasattr(X_test_df, 'columns') and top_feature_name in X_test_df.columns:
                        top_feature_values = X_test_df[top_feature_name]
                        residuals = y_test_array - y_pred_array
                        
                        plt.scatter(top_feature_values, residuals, alpha=0.6, s=30)
                        plt.axhline(y=0, color='r', linestyle='-', linewidth=2)
                        plt.xlabel(f'Most Important Feature: {top_feature_name}')
                        plt.ylabel('Residuals')
                        plt.title('Residuals vs Top Feature', fontweight='bold')
                        plt.grid(True, alpha=0.3)
                    else:
                        # Fallback: show prediction distribution
                        plt.hist(y_pred_array, bins=20, alpha=0.7, edgecolor='black')
                        plt.axvline(x=np.mean(y_pred_array), color='r', linestyle='--', 
                                label=f'Mean: {np.mean(y_pred_array):.2f}')
                        plt.xlabel('Predicted Values')
                        plt.ylabel('Frequency')
                        plt.title('Prediction Distribution', fontweight='bold')
                        plt.legend()
                else:
                    # Show prediction distribution if no feature importances
                    plt.hist(y_pred_array, bins=20, alpha=0.7, edgecolor='black')
                    plt.axvline(x=np.mean(y_pred_array), color='r', linestyle='--', 
                            label=f'Mean: {np.mean(y_pred_array):.2f}')
                    plt.xlabel('Predicted Values')
                    plt.ylabel('Frequency')
                    plt.title('Prediction Distribution', fontweight='bold')
                    plt.legend()
            except Exception as e:
                plt.text(0.5, 0.5, 'Residual analysis\nfailed', 
                        ha='center', va='center', transform=plt.gca().transAxes)
                plt.title('Residual Analysis', fontweight='bold')
        
        plt.tight_layout()
        
        # Save with target-specific filename
        plt.savefig(f"{artifact_paths['plots']}/{filename_prefix}model_evaluation.png", 
                   bbox_inches='tight', dpi=300, facecolor='white')
        plt.close()
        
        # Generate additional specialized plots
        if task_type == 'classification' and hasattr(model, 'predict_proba'):
            try:
                generate_probability_plots(model, X_test_df, y_test_array, class_names, artifact_paths, target_col, logger)
            except Exception as e:
                logger.warning(f"Probability plots failed: {str(e)}")
        elif task_type == 'regression':
            try:
                generate_regression_plots(model, X_test_df, y_test_array, artifact_paths, target_col, logger)
            except Exception as e:
                logger.warning(f"Additional regression plots failed: {str(e)}")
        
        logger.info("Basic visualizations generated successfully!")
        
    except Exception as e:
        logger.error(f"Error generating visualizations: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

def generate_regression_plots(model, X_test, y_test, artifact_paths, target_col=None, logger=None):
    """
    Generate additional regression-specific plots
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    filename_prefix = f"{target_col}_" if target_col else ""
    
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred
    
    plt.figure(figsize=(12, 4))
    
    # 1. Residuals vs Features analysis
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            if len(importances) > 0:
                top_feature_idx = np.argmax(importances)
                # Use the original feature index, not column names
                plt.subplot(1, 2, 1)
                plt.scatter(X_test[:, top_feature_idx], residuals, alpha=0.6)
                plt.axhline(y=0, color='r', linestyle='--')
                plt.xlabel(f'Top Feature (Index {top_feature_idx})')
                plt.ylabel('Residuals')
                plt.title('Residuals vs Top Feature', fontweight='bold')
                plt.grid(True, alpha=0.3)
    except Exception as e:
        logger.warning(f"Feature importance analysis failed: {str(e)}")
        # If feature importance fails, create a simple residual plot
        plt.subplot(1, 2, 1)
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residuals vs Predicted', fontweight='bold')
        plt.grid(True, alpha=0.3)
    
    # 2. Cumulative error distribution
    plt.subplot(1, 2, 2)
    sorted_errors = np.sort(np.abs(residuals))
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    plt.plot(sorted_errors, cumulative, marker='.', linestyle='none', alpha=0.6)
    plt.xlabel('Absolute Error')
    plt.ylabel('Cumulative Proportion')
    plt.title('Cumulative Error Distribution', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{artifact_paths['plots']}/{filename_prefix}regression_analysis.png", 
               bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()



    
def generate_probability_plots(model, X_test, y_test, class_names, artifact_paths, target_col=None, logger=None):
    """
    Generate probability calibration and distribution plots for classification
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    try:
        # Get predictions and ensure proper format
        y_proba = model.predict_proba(X_test)
        
        # Handle different return types from predict_proba
        if isinstance(y_proba, list):
            y_proba = np.array(y_proba)
        
        # Ensure y_proba is 2D array for multi-class classification
        if len(y_proba.shape) == 1:
            # Binary classification case - reshape to 2D
            y_proba = y_proba.reshape(-1, 1)
            if len(class_names) > 1:
                # For binary classification, we need probabilities for both classes
                y_proba = np.column_stack([1 - y_proba, y_proba])
        
        n_classes = y_proba.shape[1] if len(y_proba.shape) > 1 else 1
        
        # Create filename with target prefix
        filename_prefix = f"{target_col}_" if target_col else ""
        
        # Create title with target information
        if target_col:
            main_title = f'{target_col.upper()} - Probability Calibration and Distribution'
        else:
            main_title = 'Probability Calibration and Distribution'
        
        plt.figure(figsize=(14, 6))
        plt.suptitle(main_title, fontsize=16, fontweight='bold', y=0.98)
        
        # 1. Calibration Curve
        plt.subplot(1, 2, 1)
        calibration_data = {}
        
        for i, class_name in enumerate(class_names):
            if i >= n_classes:  # Safety check
                break
                
            # Check if we have samples for this class
            class_mask = (y_test == i)
            if np.sum(class_mask) > 10:  # Minimum samples for calibration curve
                try:
                    fraction_of_positives, mean_predicted_value = calibration_curve(
                        class_mask.astype(int),
                        y_proba[:, i],
                        n_bins=10,
                        strategy='quantile'
                    )
                    
                    # Calculate calibration error
                    calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
                    calibration_data[class_name] = calibration_error
                    
                    plt.plot(mean_predicted_value, fraction_of_positives, "o-", 
                            label=f'{class_name} (Error: {calibration_error:.3f})', 
                            markersize=5, linewidth=2, alpha=0.8)
                    
                except Exception as e:
                    logger.warning(f"Calibration curve failed for {class_name}: {str(e)}")
                    continue
        
        plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration", linewidth=2)
        plt.xlabel('Mean Predicted Probability', fontweight='bold')
        plt.ylabel('Fraction of Positives', fontweight='bold')
        plt.title('Calibration Curves', fontweight='bold')
        
        # Only show legend if we have reasonable number of classes
        if len(calibration_data) <= 10:
            plt.legend(loc='best', fontsize='small', frameon=True, fancybox=True, shadow=True)
        else:
            # For many classes, show summary instead
            avg_error = np.mean(list(calibration_data.values())) if calibration_data else 0
            plt.text(0.05, 0.95, f'Avg Calibration Error: {avg_error:.3f}', 
                    transform=plt.gca().transAxes, bbox=dict(boxstyle="round,pad=0.3", 
                    facecolor="yellow", alpha=0.7))
        
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        
        # 2. Probability Distribution with enhanced visualization
        plt.subplot(1, 2, 2)
        
        # Create color palette
        colors = plt.cm.Set3(np.linspace(0, 1, min(len(class_names), 12)))
        
        distribution_data = {}
        valid_classes = 0
        
        for i, class_name in enumerate(class_names):
            if i >= n_classes:  # Safety check
                break
                
            # Ensure y_test is numpy array for indexing
            y_test_array = np.array(y_test)
            class_mask = (y_test_array == i)
            
            if np.sum(class_mask) > 5:  # Minimum samples for distribution
                class_probs = y_proba[class_mask, i]
                color = colors[valid_classes % len(colors)]
                
                # Ensure class_probs is numpy array
                class_probs_array = np.array(class_probs)
                
                # Plot KDE
                sns.kdeplot(class_probs_array, label=f'{class_name}', fill=True, 
                           alpha=0.5, color=color, linewidth=2)
                
                # Calculate statistics
                mean_prob = np.mean(class_probs_array)
                median_prob = np.median(class_probs_array)
                distribution_data[class_name] = {
                    'mean': mean_prob,
                    'median': median_prob,
                    'count': len(class_probs_array)
                }
                
                # Add vertical lines for mean and median
                plt.axvline(x=mean_prob, color=color, linestyle='--', alpha=0.8, 
                           label=f'{class_name} Mean: {mean_prob:.2f}')
                plt.axvline(x=median_prob, color=color, linestyle=':', alpha=0.6)
                
                valid_classes += 1
        
        if valid_classes > 0:
            plt.xlabel('Predicted Probability for True Class', fontweight='bold')
            plt.ylabel('Density', fontweight='bold')
            plt.title('Probability Distributions by True Class', fontweight='bold')
            
            # Adjust legend for many classes
            if valid_classes <= 8:
                plt.legend(loc='best', fontsize='small', frameon=True, 
                          fancybox=True, shadow=True, ncol=1)
            else:
                # Show summary statistics instead of full legend
                stats_text = "\n".join([f"{cls}: μ={data['mean']:.2f}, n={data['count']}" 
                                      for cls, data in list(distribution_data.items())[:6]])
                if len(distribution_data) > 6:
                    stats_text += f"\n... and {len(distribution_data) - 6} more classes"
                
                plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
                        fontsize=8, verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        else:
            plt.text(0.5, 0.5, 'Insufficient data for\nprobability distribution', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Probability Distribution', fontweight='bold')
        
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 1)
        
        plt.tight_layout()
        
        # Save with target-specific filename
        plt.savefig(f"{artifact_paths['plots']}/{filename_prefix}probability_plots.png", 
                   bbox_inches='tight', dpi=300, facecolor='white')
        plt.close()
        
        logger.info("Probability plots generated successfully!")
        
    except Exception as e:
        logger.warning(f"Error generating probability plots: {str(e)}")
        import traceback
        logger.warning(traceback.format_exc())


def handle_class_imbalance(y_train, method='auto', custom_weights=None, smote_strategy='auto', random_state=42):
    """
    Handle class imbalance using various methods
    
    Parameters:
    -----------
    y_train : array-like
        Training target labels
    method : str, default='auto'
        Method to handle imbalance: 'auto', 'smote', 'class_weights', 'custom_weights', 'none'
    custom_weights : dict, optional
        Custom class weights dictionary {class: weight}
    smote_strategy : str or dict, default='auto'
        SMOTE sampling strategy
    random_state : int, default=42
        Random state for reproducibility
        
    Returns:
    --------
    dict: Configuration with method details and parameters
    """
    
    # Check if it's a classification task with imbalance
    unique_classes = np.unique(y_train)
    if len(unique_classes) <= 1:
        return {
            'method': 'none',
            'reason': 'Single class or regression task',
            'sample_weights': None,
            'class_weights': None,
            'smote_instance': None
        }
    
    # Calculate class distribution
    class_counts = np.bincount(y_train)
    class_distribution = {cls: count for cls, count in zip(unique_classes, class_counts)}
    
    # Check if imbalance exists (more than 2:1 ratio between largest and smallest class)
    max_count = max(class_counts)
    min_count = min(class_counts)
    is_imbalanced = (max_count / min_count) > 2 if min_count > 0 else True
    
    if not is_imbalanced:
        return {
            'method': 'none',
            'reason': 'Classes are balanced',
            'class_distribution': class_distribution,
            'sample_weights': None,
            'class_weights': None,
            'smote_instance': None
        }
    
    # Determine method if 'auto'
    if method == 'auto':
        # Auto logic: Use SMOTE if minority class has enough samples, otherwise class weights
        if min_count >= 5:  # Enough samples for SMOTE
            method = 'smote'
        else:
            method = 'class_weights'
    
    # Apply the selected method
    if method == 'smote':
        return _apply_smote(y_train, smote_strategy, class_distribution, random_state)
    
    elif method == 'class_weights':
        return _apply_class_weights(y_train, class_distribution, 'balanced')
    
    elif method == 'custom_weights':
        return _apply_custom_weights(y_train, custom_weights, class_distribution)
    
    elif method == 'none':
        return {
            'method': 'none',
            'reason': 'Explicitly disabled',
            'class_distribution': class_distribution,
            'sample_weights': None,
            'class_weights': None,
            'smote_instance': None
        }
    
    else:
        raise ValueError(f"Unknown imbalance handling method: {method}")

def _apply_smote(y_train, smote_strategy, class_distribution, random_state):
    """Apply SMOTE oversampling"""
    from imblearn.over_sampling import SMOTE
    
    # Calculate safe k_neighbors
    min_class_count = min(np.bincount(y_train))
    k_neighbors = min(5, min_class_count - 1)
    k_neighbors = max(1, k_neighbors)  # Ensure at least 1
    
    # Create SMOTE instance
    smote = SMOTE(
        sampling_strategy=smote_strategy,
        random_state=random_state,
        k_neighbors=k_neighbors
    )
    
    return {
        'method': 'smote',
        'class_distribution': class_distribution,
        'sample_weights': None,  # No sample weights when using SMOTE
        'class_weights': None,   # No class weights when using SMOTE
        'smote_instance': smote,
        'k_neighbors': k_neighbors,
        'smote_strategy': smote_strategy
    }

def _apply_class_weights(y_train, class_distribution, weight_type='balanced'):
    """Apply automatic class weights"""
    from sklearn.utils.class_weight import compute_class_weight
    
    unique_classes = np.unique(y_train)
    
    if weight_type == 'balanced':
        class_weights = compute_class_weight(
            'balanced', 
            classes=unique_classes, 
            y=y_train
        )
    else:
        # Custom weight calculation logic could be added here
        class_weights = np.ones(len(unique_classes))
    
    weight_dict = {cls: weight for cls, weight in zip(unique_classes, class_weights)}
    sample_weights = compute_sample_weights(y_train, weight_dict)
    
    return {
        'method': 'class_weights',
        'class_distribution': class_distribution,
        'sample_weights': sample_weights,
        'class_weights': weight_dict,
        'smote_instance': None
    }

def _apply_custom_weights(y_train, custom_weights, class_distribution):
    """Apply custom class weights"""
    if custom_weights is None:
        raise ValueError("Custom weights method requires custom_weights parameter")
    
    # Validate custom weights
    unique_classes = np.unique(y_train)
    for cls in unique_classes:
        if cls not in custom_weights:
            raise ValueError(f"Class {cls} not found in custom_weights")
    
    sample_weights = compute_sample_weights(y_train, custom_weights)
    
    return {
        'method': 'custom_weights',
        'class_distribution': class_distribution,
        'sample_weights': sample_weights,
        'class_weights': custom_weights.copy(),
        'smote_instance': None
    }

def compute_sample_weights(y, class_weights):
    """Compute sample weights for each instance"""
    return np.array([class_weights[label] for label in y])

def validate_imbalance_config(method, use_smote, class_weights_dict):
    """
    Validate that imbalance handling configuration is consistent
    
    Returns:
    --------
    bool: True if configuration is valid, False otherwise
    str: Error message if invalid, None if valid
    """
    
    if use_smote and class_weights_dict is not None:
        return False, "Cannot use both SMOTE and custom class weights simultaneously"
    
    if method == 'custom_weights' and class_weights_dict is None:
        return False, "Custom weights method requires class_weights_dict parameter"
    
    if method == 'smote' and not use_smote:
        return False, "SMOTE method requires use_smote=True"
    
    return True, None



class HierarchicalClassifier:
    """Optimized hierarchical classifier with better draw recall"""
    
    def __init__(self, draw_model=None, win_model=None, draw_threshold=0.25, random_state=42):
        self.random_state = random_state
        
        # Store the model classes/instances, don't instantiate them here
        self.draw_model_class = draw_model
        self.win_model_class = win_model
        self.draw_threshold = draw_threshold
        self.classes_ = np.array([0, 1, 2])
        
        # These will be set in the fit method
        self.draw_model_ = None
        self.win_model_ = None
    
    def fit(self, X, y):
        # First stage: draw vs non-draw
        y_draw = (y == 1).astype(int)
        
        # Initialize draw model if not provided
        if self.draw_model_class is None:
            self.draw_model_ = LogisticRegression(
                class_weight='balanced', 
                penalty='l1',
                solver='liblinear',
                random_state=self.random_state
            )
        else:
            # Clone the model to avoid modifying the original
            self.draw_model_ = clone(self.draw_model_class)
        
        # Check if we have both classes
        if len(np.unique(y_draw)) == 1:
            from sklearn.dummy import DummyClassifier
            self.draw_model_ = DummyClassifier(strategy='constant', constant=np.unique(y_draw)[0])
        
        self.draw_model_.fit(X, y_draw)
        
        # Second stage: home vs away for non-draws
        non_draw_mask = (y != 1)
        if np.any(non_draw_mask):
            X_non_draw = X[non_draw_mask]
            y_non_draw = y[non_draw_mask]
            y_win = (y_non_draw == 2).astype(int)
            
            # Initialize win model if not provided
            if self.win_model_class is None:
                self.win_model_ = RandomForestClassifier(
                    n_estimators=100,
                    class_weight='balanced',
                    random_state=self.random_state
                )
            else:
                self.win_model_ = clone(self.win_model_class)
            
            if len(np.unique(y_win)) > 1:
                self.win_model_.fit(X_non_draw, y_win)
            else:
                from sklearn.dummy import DummyClassifier
                self.win_model_ = DummyClassifier(strategy='constant', constant=0)
                self.win_model_.fit(X_non_draw, y_win)
        else:
            # Handle case with no non-draw samples
            from sklearn.dummy import DummyClassifier
            self.win_model_ = DummyClassifier(strategy='constant', constant=0)
            # Fit on some data (even if it's just one sample)
            self.win_model_.fit(X[:1], np.array([0]))
        
        return self
    
    def predict(self, X):
        # Get draw probabilities
        if hasattr(self.draw_model_, 'predict_proba'):
            draw_proba = self.draw_model_.predict_proba(X)[:, 1]
        else:
            # For dummy classifiers
            draw_pred = self.draw_model_.predict(X)
            draw_proba = draw_pred.astype(float)
        
        draw_pred = (draw_proba > self.draw_threshold).astype(int)
        
        # Get win predictions
        win_pred = self.win_model_.predict(X)
        
        # Combine predictions
        final_pred = np.zeros(len(X), dtype=int)
        for i in range(len(X)):
            if draw_pred[i] == 1:
                final_pred[i] = 1  # Draw
            else:
                final_pred[i] = 2 if win_pred[i] == 1 else 0  # Home or away win
                
        return final_pred
    
    def predict_proba(self, X):
        # Get draw probabilities
        if hasattr(self.draw_model_, 'predict_proba'):
            draw_proba = self.draw_model_.predict_proba(X)
        else:
            draw_pred = self.draw_model_.predict(X)
            draw_proba = np.column_stack([1 - draw_pred, draw_pred])
        
        # Get win probabilities
        if hasattr(self.win_model_, 'predict_proba'):
            win_proba = self.win_model_.predict_proba(X)
        else:
            win_pred = self.win_model_.predict(X)
            win_proba = np.column_stack([1 - win_pred, win_pred])
        
        # Combine probabilities
        final_proba = np.zeros((len(X), 3))
        for i in range(len(X)):
            # Probability of draw
            final_proba[i, 1] = draw_proba[i, 1]
            
            # Probability of home win = prob(not draw) * prob(home|not draw)
            final_proba[i, 2] = draw_proba[i, 0] * win_proba[i, 1]
            
            # Probability of away win = prob(not draw) * prob(away|not draw)
            final_proba[i, 0] = draw_proba[i, 0] * win_proba[i, 0]
            
        return final_proba


def find_optimal_draw_threshold(model, X_val, y_val):
    """Find the optimal threshold for draw prediction"""
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_val)
        draw_proba = y_proba[:, 1]  # Probability of draw
        
        thresholds = np.linspace(0.1, 0.5, 20)
        best_threshold = 0.3
        best_recall = 0
        
        for threshold in thresholds:
            draw_pred = (draw_proba > threshold).astype(int)
            recall = recall_score((y_val == 1).astype(int), draw_pred, zero_division=0)
            
            if recall > best_recall:
                best_recall = recall
                best_threshold = threshold
        
        return best_threshold, best_recall
    return 0.3, 0  # Default if no probabilities available

# Use this to set the optimal threshold
#optimal_threshold, optimal_recall = find_optimal_draw_threshold(model, X_holdout, y_holdout)
#print(f"Optimal draw threshold: {optimal_threshold:.3f}, Recall: {optimal_recall:.3f}")

def predict_target(self, df, target_col, **kwargs):
    """
    Train a model for any target (automatically detects classification/regression)
    Reuses your existing train() function with automatic task detection
    
    Parameters:
    -----------
    target_col : str
        Column name for the target (outcome, total_goals, total_cards, etc.)
    **kwargs : additional arguments to pass to train()
    
    Returns:
    --------
    model : trained model
    X_holdout : holdout features
    y_holdout : holdout targets
    """
    try:
        # Use your existing train function - it will auto-detect task type!
        model, X_holdout, y_holdout = self.train(
            df=df,
            target_col=target_col,
            **kwargs
        )
        
        self.logger.info(f"Model for '{target_col}' trained successfully! "
                        f"Task type: {self.task_type}")
        return model, X_holdout, y_holdout
        
    except Exception as e:
        self.logger.error(f"Prediction for '{target_col}' failed: {str(e)}", exc_info=True)
        raise

def train_multiple_targets(self, df, target_columns, common_config=None, specific_configs=None):
    """
    Train models for multiple targets using your existing train() function
    
    Parameters:
    -----------
    target_columns : list
        List of target column names to train models for
    common_config : dict
        Configuration applied to all targets
    specific_configs : dict
        Target-specific configurations: {'target_name': {config}}
    
    Returns:
    --------
    dict: Dictionary containing all trained models and results
    """
    if common_config is None:
        common_config = {}
    if specific_configs is None:
        specific_configs = {}
    
    results = {}
    
    for target_col in target_columns:
        try:
            self.logger.info(f"Training model for target: {target_col}")
            
            # Merge common config with target-specific config
            config = common_config.copy()
            config.update(specific_configs.get(target_col, {}))
            
            # Use your existing train function
            model, X_holdout, y_holdout = self.train(
                df=df,
                target_col=target_col,
                **config
            )
            
            # Store results
            results[target_col] = {
                'model': model,
                'X_holdout': X_holdout,
                'y_holdout': y_holdout,
                'task_type': self.task_type,  # From your auto-detection
                'metrics': self.final_metrics  # From your evaluation
            }
            
            self.logger.info(f"Completed {target_col} - Task: {self.task_type}, "
                           f"Holdout Score: {self._get_holdout_score()}")
            
        except Exception as e:
            self.logger.error(f"Failed to train model for {target_col}: {str(e)}")
            results[target_col] = {'error': str(e)}
    
    return results

def _get_holdout_score(self):
    """Helper to get the appropriate holdout score based on task type"""
    if not hasattr(self, 'final_metrics') or self.final_metrics is None:
        return "N/A"
    
    if self.task_type == 'classification':
        return f"Accuracy: {self.final_metrics.get('holdout_accuracy', 'N/A'):.3f}"
    else:
        return f"RMSE: {self.final_metrics.get('holdout_rmse', 'N/A'):.3f}, " \
               f"R²: {self.final_metrics.get('holdout_r2', 'N/A'):.3f}"
    
def predict_multiple(self, new_data, trained_models):
    """
    Make predictions for multiple targets using trained models
    
    Parameters:
    -----------
    new_data : DataFrame
        New data to make predictions on
    trained_models : dict
        Dictionary from train_multiple_targets()
    
    Returns:
    --------
    dict: Predictions for all targets with appropriate output format
    """
    predictions = {}
    
    for target_name, model_info in trained_models.items():
        if 'model' in model_info and model_info['model'] is not None:
            try:
                model = model_info['model']
                task_type = model_info.get('task_type', 'unknown')
                
                if task_type == 'classification':
                    # Use your classification evaluation format
                    if hasattr(model, 'predict_proba'):
                        pred_proba = model.predict_proba(new_data)
                        pred_class = model.predict(new_data)
                        predictions[target_name] = {
                            'prediction': pred_class,
                            'probabilities': pred_proba,
                            'type': 'classification'
                        }
                    else:
                        predictions[target_name] = {
                            'prediction': model.predict(new_data),
                            'probabilities': None,
                            'type': 'classification'
                        }
                        
                elif task_type == 'regression':
                    # Use your regression evaluation format
                    pred_value = model.predict(new_data)
                    predictions[target_name] = {
                        'prediction': pred_value,
                        'type': 'regression'
                    }
                    
                else:
                    # Fallback for unknown task types
                    predictions[target_name] = {
                        'prediction': model.predict(new_data),
                        'type': 'unknown'
                    }
                    
            except Exception as e:
                self.logger.error(f"Prediction failed for {target_name}: {str(e)}")
                predictions[target_name] = {'error': str(e), 'type': 'error'}
    
    return predictions



def quick_train_all(self, df, regression_targets=None):
    """
    Quick method to train outcome + common regression targets
    """
    if regression_targets is None:
        regression_targets = ['total_goals', 'total_cards', 'total_corners', 'total_shots']
    
    all_targets = ['outcome'] + regression_targets
    
    return self.train_multiple_targets(
        df=df,
        target_columns=all_targets,
        common_config={
            'model_type': 'random_forest',
            'feature_selection_method': 'importance',
            'top_n_features': 25,
            'holdout_ratio': 0.2
        }
    )