import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_residuals_table():
    """
    Detailed analysis of the residuals.parquet table.
    """
    
    print("DETAILED ANALYSIS OF RESIDUALS TABLE")
    print("="*80)
    
    # Read the residuals table
    residuals_df = pd.read_parquet("mock_data/residuals.parquet")
    
    print(f"Table Shape: {residuals_df.shape}")
    print(f"Memory Usage: {residuals_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print("\n" + "="*80)
    
    # Basic information
    print("COLUMN INFORMATION")
    print("-" * 50)
    for col in residuals_df.columns:
        dtype = residuals_df[col].dtype
        null_count = residuals_df[col].isnull().sum()
        null_pct = (null_count / len(residuals_df)) * 100
        unique_count = residuals_df[col].nunique()
        
        print(f"{col}:")
        print(f"  Type: {dtype}")
        print(f"  Null values: {null_count} ({null_pct:.2f}%)")
        print(f"  Unique values: {unique_count}")
        
        if dtype in ['float64', 'int64']:
            print(f"  Min: {residuals_df[col].min():.4f}")
            print(f"  Max: {residuals_df[col].max():.4f}")
            print(f"  Mean: {residuals_df[col].mean():.4f}")
            print(f"  Std: {residuals_df[col].std():.4f}")
        
        print()
    
    print("="*80)
    
    # Sample data
    print("SAMPLE DATA (First 10 rows)")
    print("-" * 50)
    print(residuals_df.head(10).to_string())
    
    print("\n" + "="*80)
    
    # Time analysis
    print("TIME DIMENSION ANALYSIS")
    print("-" * 50)
    
    # Convert Marker and Cycle to datetime if they're not already
    if residuals_df['Marker'].dtype == 'object':
        residuals_df['Marker_dt'] = pd.to_datetime(residuals_df['Marker'])
    else:
        residuals_df['Marker_dt'] = residuals_df['Marker']
    
    if residuals_df['Cycle'].dtype == 'object':
        residuals_df['Cycle_dt'] = pd.to_datetime(residuals_df['Cycle'])
    else:
        residuals_df['Cycle_dt'] = residuals_df['Cycle']
    
    print(f"Date range for Marker: {residuals_df['Marker_dt'].min()} to {residuals_df['Marker_dt'].max()}")
    print(f"Date range for Cycle: {residuals_df['Cycle_dt'].min()} to {residuals_df['Cycle_dt'].max()}")
    print(f"Number of unique Marker dates: {residuals_df['Marker_dt'].nunique()}")
    print(f"Number of unique Cycle dates: {residuals_df['Cycle_dt'].nunique()}")
    
    # Horizon analysis
    print(f"\nHorizon values: {sorted(residuals_df['Horizon'].unique())}")
    print(f"Number of unique horizons: {residuals_df['Horizon'].nunique()}")
    
    print("\n" + "="*80)
    
    # Entity analysis
    print("ENTITY ANALYSIS")
    print("-" * 50)
    print(f"Number of unique entities: {residuals_df['Entity'].nunique()}")
    print(f"Entity range: {residuals_df['Entity'].min()} to {residuals_df['Entity'].max()}")
    
    # Count records per entity
    entity_counts = residuals_df['Entity'].value_counts().sort_index()
    print(f"\nRecords per entity (first 10):")
    print(entity_counts.head(10))
    
    print(f"\nRecords per entity (last 10):")
    print(entity_counts.tail(10))
    
    print("\n" + "="*80)
    
    # Error analysis
    print("ERROR ANALYSIS")
    print("-" * 50)
    
    # Basic error statistics
    print("Error Statistics:")
    print(f"  Mean error: {residuals_df['error'].mean():.4f}")
    print(f"  Median error: {residuals_df['error'].median():.4f}")
    print(f"  Std error: {residuals_df['error'].std():.4f}")
    print(f"  Min error: {residuals_df['error'].min():.4f}")
    print(f"  Max error: {residuals_df['error'].max():.4f}")
    
    print("\nAbsolute Error Statistics:")
    print(f"  Mean absolute error: {residuals_df['absolute_error'].mean():.4f}")
    print(f"  Median absolute error: {residuals_df['absolute_error'].median():.4f}")
    print(f"  Std absolute error: {residuals_df['absolute_error'].std():.4f}")
    print(f"  Min absolute error: {residuals_df['absolute_error'].min():.4f}")
    print(f"  Max absolute error: {residuals_df['absolute_error'].max():.4f}")
    
    # Error distribution
    print(f"\nError distribution:")
    print(f"  Positive errors: {(residuals_df['error'] > 0).sum()} ({(residuals_df['error'] > 0).mean()*100:.1f}%)")
    print(f"  Negative errors: {(residuals_df['error'] < 0).sum()} ({(residuals_df['error'] < 0).mean()*100:.1f}%)")
    print(f"  Zero errors: {(residuals_df['error'] == 0).sum()} ({(residuals_df['error'] == 0).mean()*100:.1f}%)")
    
    print("\n" + "="*80)
    
    # Forecast vs Observed analysis
    print("FORECAST vs OBSERVED ANALYSIS")
    print("-" * 50)
    
    print("Observed values:")
    print(f"  Mean: {residuals_df['observed'].mean():.4f}")
    print(f"  Median: {residuals_df['observed'].median():.4f}")
    print(f"  Std: {residuals_df['observed'].std():.4f}")
    print(f"  Min: {residuals_df['observed'].min():.4f}")
    print(f"  Max: {residuals_df['observed'].max():.4f}")
    
    print("\nForecasted values:")
    print(f"  Mean: {residuals_df['forecasted'].mean():.4f}")
    print(f"  Median: {residuals_df['forecasted'].median():.4f}")
    print(f"  Std: {residuals_df['forecasted'].std():.4f}")
    print(f"  Min: {residuals_df['forecasted'].min():.4f}")
    print(f"  Max: {residuals_df['forecasted'].max():.4f}")
    
    # Correlation between observed and forecasted
    correlation = residuals_df['observed'].corr(residuals_df['forecasted'])
    print(f"\nCorrelation between observed and forecasted: {correlation:.4f}")
    
    print("\n" + "="*80)
    
    # Horizon-specific analysis
    print("HORIZON-SPECIFIC ANALYSIS")
    print("-" * 50)
    
    horizon_stats = residuals_df.groupby('Horizon').agg({
        'error': ['mean', 'std', 'min', 'max'],
        'absolute_error': ['mean', 'std', 'min', 'max'],
        'observed': ['mean', 'std'],
        'forecasted': ['mean', 'std']
    }).round(4)
    
    print("Statistics by Horizon:")
    print(horizon_stats)
    
    print("\n" + "="*80)
    
    # Entity-specific analysis (top 10 entities by record count)
    print("ENTITY-SPECIFIC ANALYSIS (Top 10 entities by record count)")
    print("-" * 50)
    
    top_entities = entity_counts.head(10).index
    entity_stats = residuals_df[residuals_df['Entity'].isin(top_entities)].groupby('Entity').agg({
        'error': ['mean', 'std', 'count'],
        'absolute_error': ['mean', 'std'],
        'observed': ['mean', 'std'],
        'forecasted': ['mean', 'std']
    }).round(4)
    
    print("Statistics by Entity (top 10):")
    print(entity_stats)
    
    print("\n" + "="*80)
    
    # Data quality checks
    print("DATA QUALITY CHECKS")
    print("-" * 50)
    
    # Check for missing values
    missing_data = residuals_df.isnull().sum()
    print("Missing values per column:")
    for col, count in missing_data.items():
        if count > 0:
            print(f"  {col}: {count} ({(count/len(residuals_df))*100:.2f}%)")
        else:
            print(f"  {col}: No missing values")
    
    # Check for duplicate rows
    duplicates = residuals_df.duplicated().sum()
    print(f"\nDuplicate rows: {duplicates}")
    
    # Check for impossible values
    print(f"\nImpossible values:")
    print(f"  Negative observed values: {(residuals_df['observed'] < 0).sum()}")
    print(f"  Negative forecasted values: {(residuals_df['forecasted'] < 0).sum()}")
    print(f"  Negative absolute errors: {(residuals_df['absolute_error'] < 0).sum()}")
    
    # Check if absolute_error matches |error|
    error_consistency = (residuals_df['absolute_error'] == residuals_df['error'].abs()).all()
    print(f"  Absolute error consistency: {'✓' if error_consistency else '✗'}")
    
    print("\n" + "="*80)
    
    # Summary insights
    print("SUMMARY INSIGHTS")
    print("-" * 50)
    
    print("1. Data Structure:")
    print(f"   - {residuals_df.shape[0]:,} total records")
    print(f"   - {residuals_df['Entity'].nunique()} unique entities")
    print(f"   - {residuals_df['Horizon'].nunique()} forecast horizons")
    print(f"   - {residuals_df['Marker_dt'].nunique()} unique marker dates")
    print(f"   - {residuals_df['Cycle_dt'].nunique()} unique cycle dates")
    
    print("\n2. Forecast Performance:")
    print(f"   - Mean absolute error: {residuals_df['absolute_error'].mean():.2f}")
    print(f"   - Correlation (observed vs forecasted): {correlation:.3f}")
    print(f"   - Forecast bias (mean error): {residuals_df['error'].mean():.2f}")
    
    print("\n3. Data Quality:")
    print(f"   - Missing values: {missing_data.sum()} total")
    print(f"   - Duplicate rows: {duplicates}")
    print(f"   - Data consistency: {'Good' if error_consistency else 'Issues found'}")
    
    return residuals_df

if __name__ == "__main__":
    residuals_df = analyze_residuals_table()
