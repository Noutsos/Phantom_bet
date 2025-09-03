
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import type_of_target
from ydata_profiling import ProfileReport

def auto_detect_categorical(df, threshold=0.05, max_unique=20, exclude_columns=None):
    """Automatically detect categorical columns with smart heuristics"""
    if exclude_columns is None:
        exclude_columns = []
    
    categoricals = {'binary': [], 'ordinal': [], 'nominal': []}
    
    for col in df.columns:
        if col in exclude_columns:
            continue
            
        # Skip if all values are NA
        if df[col].isna().all():
            continue
            
        n_unique = df[col].nunique(dropna=True)
        unique_ratio = n_unique / len(df)
        
        # Binary detection
        if n_unique == 2:
            categoricals['binary'].append(col)
            continue
            
        # Numeric categories (ordinal)
        if pd.api.types.is_numeric_dtype(df[col]):
            # Check if we can safely convert to int (ignoring NAs)
            try:
                if n_unique <= max_unique and np.allclose(df[col].dropna(), df[col].dropna().astype(int)):
                    categoricals['ordinal'].append(col)
            except (ValueError, TypeError):
                pass
            continue
            
        # String/object types (nominal)
        if (pd.api.types.is_string_dtype(df[col]) or 
            pd.api.types.is_object_dtype(df[col]) or 
            (unique_ratio < threshold) or 
            (n_unique <= max_unique)):
            categoricals['nominal'].append(col)
            
    return categoricals

def validate_outcome_target_ml(df, target_column, date_column=None, manual_categoricals=None):
    """
    Enhanced validation for outcome target classification (0=draw, 1=home win, 2=away win)
    
    Parameters:
        df: Input DataFrame
        target_column: Name of the target column
        date_column: Name of datetime column
        manual_categoricals: User-specified categorical columns
    """
    print("="*80)
    print("OUTCOME TARGET ML VALIDATION REPORT")
    print("="*80)
    
    # --- Phase 1: Target Validation ---
    print("\n🎯 TARGET VALIDATION")
    
    # Check for missing values in target
    if df[target_column].isnull().any():
        null_count = df[target_column].isnull().sum()
        print(f"⚠️ WARNING: Target column contains {null_count} missing values")
        print("Recommendation: Handle missing values before proceeding (impute or remove rows)")
        
        # Show rows with missing targets
        print("\nRows with missing targets:")
        print(df[df[target_column].isnull()].head())
        
        # Convert to float temporarily to avoid conversion error
        target_series = df[target_column].astype(float)
    else:
        target_series = df[target_column]
    
    # Check target values are valid (0, 1, or 2) among non-missing values
    valid_values = {0, 1, 2}
    unique_values = set(target_series.dropna().unique())
    
    if not unique_values.issubset(valid_values):
        invalid_values = unique_values - valid_values
        raise ValueError(f"Target column contains invalid values: {invalid_values}. "
                       "Should only contain 0 (draw), 1 (home win), or 2 (away win).")
    
    # Class distribution (excluding missing values)
    value_names = {0: 'Draw', 1: 'Home Win', 2: 'Away Win'}
    target_counts = target_series.value_counts(dropna=False).rename(index=value_names)
    
    print("\nCLASS DISTRIBUTION (including missing values):")
    print(target_counts)
    
    # Plot distribution
    plt.figure(figsize=(10, 4))
    sns.barplot(x=target_counts.index.astype(str),  # Convert to string to handle NaN
                y=target_counts.values)
    plt.title("Outcome Class Distribution")
    plt.ylabel("Count")
    plt.show()
    
    # Check target type
    target_type = type_of_target(df[target_column])
    print(f"\nTARGET TYPE: {target_type}")
    
    # --- Phase 2: Feature Analysis ---
    exclude = [target_column] + ([date_column] if date_column else [])
    auto_cats = auto_detect_categorical(df, exclude_columns=exclude)
    
    if manual_categoricals:
        for cat_type in auto_cats:
            auto_cats[cat_type] += [c for c in manual_categoricals 
                                  if c in df.columns and c not in exclude]
    
    print("\n🔍 AUTO-DETECTED FEATURES:")
    for cat_type, cols in auto_cats.items():
        print(f"{cat_type.upper()}: {cols}")
    
    # --- Phase 3: Data Quality Checks ---
    print("\n" + "="*80)
    print("DATA QUALITY CHECKS")
    print("="*80)
    
    print(f"\n📐 SHAPE: {df.shape}")
    nulls = df.isnull().sum()
    if nulls.any():
        print("\n⚠️ NULL VALUES:")
        print(nulls[nulls > 0])
    else:
        print("\n✅ NO NULL VALUES")
    
    # --- Phase 4: Recommendations ---
    print("\n" + "="*80)
    print("RECOMMENDED ACTIONS")
    print("="*80)
    
    recommendations = []
    
    # Handle class imbalance
    imbalance_ratio = target_counts.max() / target_counts.min()
    if imbalance_ratio > 3:
        recommendations.append(
            f"Significant class imbalance (ratio: {imbalance_ratio:.1f}x). "
            "Consider using class weights or oversampling."
        )
    
    # Feature engineering
    if date_column:
        recommendations.append(
            f"Create time features from '{date_column}' (day_of_week, month, etc.)"
        )
    
    # Categorical encoding
    for cat_type, cols in auto_cats.items():
        if cols:
            if cat_type == 'binary':
                recommendations.append(f"Label encode binary: {cols}")
            elif cat_type == 'ordinal':
                recommendations.append(f"Use OrdinalEncoder for ordinal: {cols}")
            else:
                recommendations.append(f"One-hot encode nominal: {cols}")
    
    # Numeric features
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != target_column]
    if numeric_cols:
        recommendations.append(f"Scale numeric features: {numeric_cols}")
    
    # Model suggestions
    recommendations.append(
        "\nMODELING STRATEGY:\n"
        "1. Use multi-class classification approach (3 classes)\n"
        "2. For neural networks: sparse_categorical_crossentropy loss\n"
        "3. For sklearn: RandomForestClassifier or LogisticRegression(multi_class='multinomial')\n"
        "4. Consider using class_weight='balanced' for imbalanced datasets"
    )
    
    print("\n".join(recommendations))
    
    # --- Phase 5: Generate Report ---
    print("\n" + "="*80)
    print("GENERATING COMPREHENSIVE REPORT...")
    profile = ProfileReport(df, title="Outcome Target Validation Report")
    profile.to_file("outcome_target_validation_report.html")
    print("✅ Saved report as 'outcome_target_validation_report.html'")

# Example Usage
if __name__ == "__main__":
    # Load your data
    df = pd.read_csv('data/merged/Premier League/2024/merged_data.csv')
    
 
    
    # Run validation
    validate_outcome_target_ml(
        df,
        target_column='outcome',
        date_column='date'
    )