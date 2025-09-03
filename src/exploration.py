import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy import stats
from typing import Dict, List, Optional, Union
import warnings
warnings.filterwarnings('ignore')

def explore_and_analyze(df: pd.DataFrame,
                       target_column: Optional[str] = None,
                       date_column: Optional[str] = None,
                       team_column: Optional[str] = None,
                       save_dir: str = 'analysis_results',
                       show_plots: bool = True) -> Dict[str, Dict]:
    """
    Comprehensive football data exploration and analysis covering:
    - Basic data quality checks
    - Null value analysis
    - Feature distributions
    - Target analysis (if provided)
    - Temporal patterns (if date column provided)
    - Team performance analysis (if team column provided)
    - Outlier detection
    - Cardinality analysis
    
    Parameters:
        df: Input DataFrame
        target_column: Name of target variable column (optional)
        date_column: Name of datetime column (optional)
        team_column: Name of team identifier column (optional)
        save_dir: Directory to save visualizations
        show_plots: Whether to display plots interactively
        
    Returns:
        Dictionary containing all analysis results and visualization paths
    """
    
    # Setup results dictionary and output directory
    os.makedirs(save_dir, exist_ok=True)
    results = {
        'metadata': {
            'shape': df.shape,
            'columns': list(df.columns),
            'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        },
        'data_quality': {},
        'distributions': {},
        'visualizations': {}
    }
    
    # Set visual style
    plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'ggplot')
    sns.set_palette("colorblind")
    plt.rcParams['figure.dpi'] = 120
    plt.rcParams['savefig.dpi'] = 300
    
    # ==================================================
    # 1. BASIC DATA QUALITY CHECKS
    # ==================================================
    print("="*80)
    print("🔍 BASIC DATA QUALITY CHECKS")
    print("="*80)
    
    # Null value analysis
    null_counts = df.isnull().sum()
    null_pct = (null_counts / len(df)) * 100
    null_df = pd.DataFrame({'null_count': null_counts, 'null_pct': null_pct})
    null_df = null_df[null_df['null_count'] > 0].sort_values('null_pct', ascending=False)
    
    results['data_quality']['null_values'] = null_df.to_dict('records')
    
    if len(null_df) > 0:
        print("\n⚠️ NULL VALUES FOUND:")
        print(null_df.to_string())
        
        # Visualize null values
        plt.figure(figsize=(12, 6))
        sns.barplot(data=null_df.reset_index(), x='index', y='null_pct', palette='viridis')
        plt.title('Percentage of Null Values by Column', fontsize=14)
        plt.xlabel('Columns')
        plt.ylabel('Null Values (%)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        null_plot_path = os.path.join(save_dir, 'null_values.png')
        plt.savefig(null_plot_path)
        if show_plots: plt.show()
        else: plt.close()
        results['visualizations']['null_values'] = null_plot_path
    else:
        print("\n✅ No null values found in any columns")
    
    # Duplicate rows check
    duplicates = df.duplicated().sum()
    results['data_quality']['duplicates'] = duplicates
    print(f"\nDuplicate rows: {duplicates}")
    
    # ==================================================
    # 2. DATA TYPE AND CARDINALITY ANALYSIS
    # ==================================================
    print("\n\n" + "="*80)
    print("📊 DATA TYPE AND CARDINALITY ANALYSIS")
    print("="*80)
    
    dtype_info = []
    for col in df.columns:
        dtype_info.append({
            'column': col,
            'dtype': str(df[col].dtype),
            'unique_count': df[col].nunique(),
            'sample_values': list(df[col].dropna().unique()[:3])
        })
    
    dtype_df = pd.DataFrame(dtype_info)
    results['data_quality']['dtype_info'] = dtype_df.to_dict('records')
    
    print("\nData Types and Unique Values:")
    print(dtype_df[['column', 'dtype', 'unique_count']].to_string())
    
    # Identify high cardinality categorical columns
    high_card_cols = dtype_df[
        (dtype_df['unique_count'] > 50) & 
        (dtype_df['dtype'].isin(['object', 'category']))
    ]
    
    if len(high_card_cols) > 0:
        print("\n⚠️ High Cardinality Categorical Columns:")
        print(high_card_cols[['column', 'unique_count']].to_string())
        results['data_quality']['high_cardinality'] = high_card_cols.to_dict('records')
    
    # ==================================================
    # 3. FEATURE DISTRIBUTIONS
    # ==================================================
    print("\n\n" + "="*80)
    print("📈 FEATURE DISTRIBUTIONS")
    print("="*80)
    
    for col in df.columns:
        col_results = {}
        plt.figure(figsize=(12, 5))
        
        # Numeric features
        if pd.api.types.is_numeric_dtype(df[col]):
            # Histogram with KDE
            plt.subplot(1, 2, 1)
            sns.histplot(df[col], kde=True, bins=30)
            plt.title(f'Distribution of {col}')
            
            # Boxplot
            plt.subplot(1, 2, 2)
            sns.boxplot(x=df[col])
            plt.title(f'Spread of {col}')
            
            # Statistics
            col_results['stats'] = {
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'skew': df[col].skew(),
                'kurtosis': df[col].kurtosis(),
                'zeros': (df[col] == 0).sum(),
                'negatives': (df[col] < 0).sum()
            }
            
        # Categorical features
        else:
            value_counts = df[col].value_counts(dropna=False).head(20)
            sns.barplot(x=value_counts.index.astype(str), y=value_counts.values)
            plt.title(f'Top Values in {col}')
            plt.xticks(rotation=45)
            
            col_results['stats'] = {
                'value_counts': value_counts.to_dict(),
                'unique_count': len(value_counts)
            }
        
        plt.tight_layout()
        plot_path = os.path.join(save_dir, f'distribution_{col}.png')
        plt.savefig(plot_path)
        if show_plots: plt.show()
        else: plt.close()
        
        col_results['plot_path'] = plot_path
        results['distributions'][col] = col_results
    
    # ==================================================
    # 4. TARGET ANALYSIS (if provided)
    # ==================================================
    if target_column and target_column in df.columns:
        print("\n\n" + "="*80)
        print(f"🎯 TARGET VARIABLE ANALYSIS: {target_column}")
        print("="*80)
        
        target_counts = df[target_column].value_counts(dropna=False)
        target_pct = df[target_column].value_counts(normalize=True, dropna=False) * 100
        target_stats = pd.DataFrame({'count': target_counts, 'percentage': target_pct})
        
        results['target_analysis'] = {
            'distribution': target_stats.to_dict(),
            'null_count': df[target_column].isnull().sum()
        }
        
        print("\nTarget Distribution:")
        print(target_stats.to_string())
        
        # Visualize target distribution
        plt.figure(figsize=(10, 6))
        ax = sns.countplot(data=df, x=target_column)
        
        # Add annotations
        total = len(df)
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x() + p.get_width()/2, height,
                   f'{height}\n({height/total:.1%})',
                   ha='center', va='center', fontsize=10)
        
        plt.title(f'Distribution of Target: {target_column}', fontsize=14)
        plt.xlabel(target_column)
        plt.ylabel('Count')
        
        target_plot_path = os.path.join(save_dir, 'target_distribution.png')
        plt.savefig(target_plot_path)
        if show_plots: plt.show()
        else: plt.close()
        results['visualizations']['target_distribution'] = target_plot_path
        
        # Check for class imbalance
        if len(target_counts) > 1:
            imbalance_ratio = target_counts.max() / target_counts.min()
            results['target_analysis']['imbalance_ratio'] = imbalance_ratio
            if imbalance_ratio > 5:
                print(f"\n⚠️ Significant class imbalance (ratio: {imbalance_ratio:.1f}x)")
    
    # ==================================================
    # 5. TEMPORAL ANALYSIS (if date column provided)
    # ==================================================
    if date_column and date_column in df.columns:
        print("\n\n" + "="*80)
        print(f"⏰ TEMPORAL ANALYSIS: {date_column}")
        print("="*80)
        
        try:
            df[date_column] = pd.to_datetime(df[date_column])
            results['temporal_analysis'] = {
                'date_range': {
                    'start': df[date_column].min().strftime('%Y-%m-%d'),
                    'end': df[date_column].max().strftime('%Y-%m-%d')
                }
            }
            
            print(f"\nDate Range: {results['temporal_analysis']['date_range']['start']} to {results['temporal_analysis']['date_range']['end']}")
            
            # Monthly trends
            monthly = df.set_index(date_column).resample('M').size()
            
            plt.figure(figsize=(14, 6))
            monthly.plot(kind='bar', width=0.8)
            plt.title('Matches per Month', fontsize=14)
            plt.ylabel('Number of Matches')
            plt.xlabel('Month')
            
            # Format x-axis
            labels = [x.strftime('%b %Y') for x in monthly.index]
            plt.xticks(ticks=range(len(labels)), labels=labels, rotation=45)
            
            monthly_path = os.path.join(save_dir, 'monthly_matches.png')
            plt.savefig(monthly_path, bbox_inches='tight')
            if show_plots: plt.show()
            else: plt.close()
            results['visualizations']['monthly_trends'] = monthly_path
            
            # Target over time (if target provided)
            if target_column:
                monthly_outcomes = df.set_index(date_column).groupby(
                    [pd.Grouper(freq='M'), target_column]
                ).size().unstack()
                
                plt.figure(figsize=(14, 7))
                monthly_outcomes.plot(kind='bar', stacked=True, width=0.8)
                plt.title('Outcomes by Month', fontsize=14)
                plt.ylabel('Number of Matches')
                plt.xlabel('Month')
                
                # Format x-axis
                labels = [x.strftime('%b %Y') for x in monthly_outcomes.index]
                plt.xticks(ticks=range(len(labels)), labels=labels, rotation=45)
                plt.legend(title='Outcome')
                
                outcome_path = os.path.join(save_dir, 'monthly_outcomes.png')
                plt.savefig(outcome_path, bbox_inches='tight')
                if show_plots: plt.show()
                else: plt.close()
                results['visualizations']['monthly_outcomes'] = outcome_path
        
        except Exception as e:
            print(f"\n⚠️ Temporal analysis failed: {str(e)}")
            results['temporal_analysis']['error'] = str(e)
    
    # ==================================================
    # 6. TEAM ANALYSIS (if team column provided)
    # ==================================================
    if team_column and team_column in df.columns:
        print("\n\n" + "="*80)
        print(f"⚽ TEAM PERFORMANCE ANALYSIS: {team_column}")
        print("="*80)
        
        team_stats = {}
        
        # Basic team performance
        if target_column:
            team_outcomes = df.groupby(team_column)[target_column].value_counts().unstack()
            team_outcomes['total_matches'] = team_outcomes.sum(axis=1)
            for outcome in [0, 1, 2]:
                if outcome in team_outcomes:
                    team_outcomes[f'pct_{outcome}'] = (team_outcomes[outcome] / team_outcomes['total_matches']) * 100
            
            results['team_analysis'] = team_outcomes.to_dict()
            print("\nTeam Performance Summary:")
            print(team_outcomes.sort_values('total_matches', ascending=False).head(10).to_string())
            
            # Visualize top teams
            top_teams = team_outcomes.nlargest(10, 'total_matches').index
            
            plt.figure(figsize=(14, 7))
            team_outcomes.loc[top_teams].drop('total_matches', axis=1).plot(kind='bar', stacked=True)
            plt.title('Outcome Distribution for Top Teams', fontsize=14)
            plt.ylabel('Number of Matches')
            plt.xlabel('Team')
            plt.xticks(rotation=45)
            plt.legend(title='Outcome')
            
            team_plot_path = os.path.join(save_dir, 'team_performance.png')
            plt.savefig(team_plot_path, bbox_inches='tight')
            if show_plots: plt.show()
            else: plt.close()
            results['visualizations']['team_performance'] = team_plot_path
    
    # ==================================================
    # 7. OUTLIER DETECTION
    # ==================================================
    print("\n\n" + "="*80)
    print("📊 OUTLIER DETECTION")
    print("="*80)
    
    outliers = {}
    numeric_cols = df.select_dtypes(include=np.number).columns
    
    for col in numeric_cols:
        # Using IQR method
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        outlier_count = outlier_mask.sum()
        
        if outlier_count > 0:
            outliers[col] = {
                'outlier_count': outlier_count,
                'outlier_pct': (outlier_count / len(df)) * 100,
                'min': df[col].min(),
                'max': df[col].max(),
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
    
    results['outlier_analysis'] = outliers
    
    if len(outliers) > 0:
        print("\nOutliers Detected (IQR method):")
        outlier_df = pd.DataFrame(outliers).T.sort_values('outlier_pct', ascending=False)
        print(outlier_df[['outlier_count', 'outlier_pct']].to_string())
    else:
        print("\n✅ No significant outliers detected in numeric columns")
    
    # ==================================================
    # FINAL SUMMARY
    # ==================================================
    print("\n\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nAll visualizations saved to: {save_dir}")
    
    return results

# Example usage
if __name__ == "__main__":
    # Load your data
    data_path = 'data/merged/Premier League/2024/merged_data.csv'
    df = pd.read_csv(data_path)
    
    # Run comprehensive analysis
    analysis_results = explore_and_analyze(
        df=df,
        target_column='outcome',  # 0=draw, 1=home win, 2=away win
        date_column='match_date',
        team_column='team_name',
        save_dir='football_analysis',
        show_plots=False
    )