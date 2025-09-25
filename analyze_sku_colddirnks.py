import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_sku_colddirnks_table():
    """
    Detailed analysis of the sku-colddirnks.parquet table.
    """
    
    print("DETAILED ANALYSIS OF SKU-COLDDRINKS TABLE")
    print("="*80)
    
    # Read the sku-colddirnks table
    sku_df = pd.read_parquet("mock_data/sku-colddirnks.parquet")
    
    print(f"Table Shape: {sku_df.shape}")
    print(f"Memory Usage: {sku_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print("\n" + "="*80)
    
    # Basic information
    print("COLUMN INFORMATION")
    print("-" * 50)
    for col in sku_df.columns:
        dtype = sku_df[col].dtype
        null_count = sku_df[col].isnull().sum()
        null_pct = (null_count / len(sku_df)) * 100
        unique_count = sku_df[col].nunique()
        
        print(f"{col}:")
        print(f"  Type: {dtype}")
        print(f"  Null values: {null_count} ({null_pct:.2f}%)")
        print(f"  Unique values: {unique_count}")
        
        if dtype in ['float64', 'int64']:
            print(f"  Min: {sku_df[col].min():.4f}")
            print(f"  Max: {sku_df[col].max():.4f}")
            print(f"  Mean: {sku_df[col].mean():.4f}")
            print(f"  Std: {sku_df[col].std():.4f}")
        elif dtype == 'object':
            print(f"  Sample values: {list(sku_df[col].unique()[:10])}")
        
        print()
    
    print("="*80)
    
    # Sample data
    print("SAMPLE DATA (First 10 rows)")
    print("-" * 50)
    print(sku_df.head(10).to_string())
    
    print("\n" + "="*80)
    
    # SKU_ID analysis
    print("SKU_ID ANALYSIS")
    print("-" * 50)
    
    print(f"Number of unique SKU_IDs: {sku_df['SKU_ID'].nunique()}")
    print(f"SKU_ID range: {sku_df['SKU_ID'].min()} to {sku_df['SKU_ID'].max()}")
    print(f"Sample SKU_IDs: {list(sku_df['SKU_ID'].head(10))}")
    
    print("\n" + "="*80)
    
    # Product information analysis
    print("PRODUCT INFORMATION ANALYSIS")
    print("-" * 50)
    
    # Product Name analysis
    print("Product Name:")
    print(f"  Unique names: {sku_df['Product_Name'].nunique()}")
    print(f"  Sample names: {list(sku_df['Product_Name'].head(10))}")
    
    # Brand analysis
    print(f"\nBrand:")
    print(f"  Unique brands: {sku_df['Brand'].nunique()}")
    brand_counts = sku_df['Brand'].value_counts()
    print(f"  Brand distribution:")
    for brand, count in brand_counts.items():
        print(f"    {brand}: {count} products ({count/len(sku_df)*100:.1f}%)")
    
    # Category analysis
    print(f"\nCategory:")
    print(f"  Unique categories: {sku_df['Category'].nunique()}")
    category_counts = sku_df['Category'].value_counts()
    print(f"  Category distribution:")
    for category, count in category_counts.items():
        print(f"    {category}: {count} products ({count/len(sku_df)*100:.1f}%)")
    
    # Flavor analysis
    print(f"\nFlavor:")
    print(f"  Unique flavors: {sku_df['Flavor'].nunique()}")
    flavor_counts = sku_df['Flavor'].value_counts()
    print(f"  Top 10 flavors:")
    for flavor, count in flavor_counts.head(10).items():
        print(f"    {flavor}: {count} products ({count/len(sku_df)*100:.1f}%)")
    
    # Package Size analysis
    print(f"\nPackage Size:")
    print(f"  Unique sizes: {sku_df['Package_Size'].nunique()}")
    size_counts = sku_df['Package_Size'].value_counts()
    print(f"  Size distribution:")
    for size, count in size_counts.items():
        print(f"    {size}: {count} products ({count/len(sku_df)*100:.1f}%)")
    
    print("\n" + "="*80)
    
    # Pricing analysis
    print("PRICING ANALYSIS")
    print("-" * 50)
    
    print("Price Statistics:")
    print(f"  Mean price: ${sku_df['Price'].mean():.2f}")
    print(f"  Median price: ${sku_df['Price'].median():.2f}")
    print(f"  Min price: ${sku_df['Price'].min():.2f}")
    print(f"  Max price: ${sku_df['Price'].max():.2f}")
    print(f"  Std price: ${sku_df['Price'].std():.2f}")
    
    # Price by category
    print(f"\nPrice by Category:")
    price_by_category = sku_df.groupby('Category')['Price'].agg(['mean', 'median', 'std', 'min', 'max']).round(2)
    print(price_by_category)
    
    # Price by brand
    print(f"\nPrice by Brand:")
    price_by_brand = sku_df.groupby('Brand')['Price'].agg(['mean', 'median', 'std', 'min', 'max']).round(2)
    print(price_by_brand)
    
    print("\n" + "="*80)
    
    # Nutritional analysis
    print("NUTRITIONAL ANALYSIS")
    print("-" * 50)
    
    # Sugar content analysis
    print("Sugar Content (g per 100ml):")
    print(f"  Mean: {sku_df['Sugar_Content_g_per_100ml'].mean():.2f}g")
    print(f"  Median: {sku_df['Sugar_Content_g_per_100ml'].median():.2f}g")
    print(f"  Min: {sku_df['Sugar_Content_g_per_100ml'].min():.2f}g")
    print(f"  Max: {sku_df['Sugar_Content_g_per_100ml'].max():.2f}g")
    print(f"  Std: {sku_df['Sugar_Content_g_per_100ml'].std():.2f}g")
    
    # Caffeine content analysis
    print(f"\nCaffeine Content (mg per serving):")
    print(f"  Mean: {sku_df['Caffeine_Content_mg_per_serving'].mean():.2f}mg")
    print(f"  Median: {sku_df['Caffeine_Content_mg_per_serving'].median():.2f}mg")
    print(f"  Min: {sku_df['Caffeine_Content_mg_per_serving'].min():.2f}mg")
    print(f"  Max: {sku_df['Caffeine_Content_mg_per_serving'].max():.2f}mg")
    print(f"  Std: {sku_df['Caffeine_Content_mg_per_serving'].std():.2f}mg")
    
    # Carbonated analysis
    print(f"\nCarbonated:")
    carbonated_counts = sku_df['Carbonated'].value_counts()
    for carbonated, count in carbonated_counts.items():
        print(f"  {carbonated}: {count} products ({count/len(sku_df)*100:.1f}%)")
    
    print("\n" + "="*80)
    
    # Elasticity analysis
    print("ELASTICITY ANALYSIS")
    print("-" * 50)
    
    elasticity_cols = ['elasticity_price', 'elasticity_season', 'elasticity_holiday', 'elasticity_trend']
    
    for col in elasticity_cols:
        print(f"{col}:")
        print(f"  Mean: {sku_df[col].mean():.4f}")
        print(f"  Median: {sku_df[col].median():.4f}")
        print(f"  Min: {sku_df[col].min():.4f}")
        print(f"  Max: {sku_df[col].max():.4f}")
        print(f"  Std: {sku_df[col].std():.4f}")
        
        # Elasticity interpretation
        if col == 'elasticity_price':
            print(f"  Price elasticity interpretation:")
            print(f"    Elastic (|elasticity| > 1): {(abs(sku_df[col]) > 1).sum()} products ({(abs(sku_df[col]) > 1).mean()*100:.1f}%)")
            print(f"    Inelastic (|elasticity| < 1): {(abs(sku_df[col]) < 1).sum()} products ({(abs(sku_df[col]) < 1).mean()*100:.1f}%)")
        
        print()
    
    print("="*80)
    
    # Base ships analysis
    print("BASE SHIPS ANALYSIS")
    print("-" * 50)
    
    print("Base Ships Statistics:")
    print(f"  Mean: {sku_df['base_ships'].mean():.2f}")
    print(f"  Median: {sku_df['base_ships'].median():.2f}")
    print(f"  Min: {sku_df['base_ships'].min():.2f}")
    print(f"  Max: {sku_df['base_ships'].max():.2f}")
    print(f"  Std: {sku_df['base_ships'].std():.2f}")
    
    # Base ships by category
    print(f"\nBase Ships by Category:")
    ships_by_category = sku_df.groupby('Category')['base_ships'].agg(['mean', 'median', 'std', 'min', 'max']).round(2)
    print(ships_by_category)
    
    # Base ships by brand
    print(f"\nBase Ships by Brand:")
    ships_by_brand = sku_df.groupby('Brand')['base_ships'].agg(['mean', 'median', 'std', 'min', 'max']).round(2)
    print(ships_by_brand)
    
    print("\n" + "="*80)
    
    # Data quality checks
    print("DATA QUALITY CHECKS")
    print("-" * 50)
    
    # Check for missing values
    missing_data = sku_df.isnull().sum()
    print("Missing values per column:")
    for col, count in missing_data.items():
        if count > 0:
            print(f"  {col}: {count} ({(count/len(sku_df))*100:.2f}%)")
        else:
            print(f"  {col}: No missing values")
    
    # Check for duplicate rows
    duplicates = sku_df.duplicated().sum()
    print(f"\nDuplicate rows: {duplicates}")
    
    # Check for duplicate SKU_IDs
    sku_duplicates = sku_df['SKU_ID'].duplicated().sum()
    print(f"Duplicate SKU_IDs: {sku_duplicates}")
    
    # Check for impossible values
    print(f"\nImpossible values:")
    print(f"  Negative prices: {(sku_df['Price'] < 0).sum()}")
    print(f"  Negative sugar content: {(sku_df['Sugar_Content_g_per_100ml'] < 0).sum()}")
    print(f"  Negative caffeine content: {(sku_df['Caffeine_Content_mg_per_serving'] < 0).sum()}")
    print(f"  Negative base ships: {(sku_df['base_ships'] < 0).sum()}")
    
    print("\n" + "="*80)
    
    # Correlation analysis
    print("CORRELATION ANALYSIS")
    print("-" * 50)
    
    # Select numeric columns for correlation
    numeric_cols = sku_df.select_dtypes(include=[np.number]).columns
    correlation_matrix = sku_df[numeric_cols].corr()
    
    print("Correlation matrix (numeric columns only):")
    print(correlation_matrix.round(3))
    
    # Find strongest correlations
    print(f"\nStrongest correlations (|r| > 0.5):")
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_val = correlation_matrix.iloc[i, j]
            if abs(corr_val) > 0.5:
                print(f"  {correlation_matrix.columns[i]} vs {correlation_matrix.columns[j]}: {corr_val:.3f}")
    
    print("\n" + "="*80)
    
    # Summary insights
    print("SUMMARY INSIGHTS")
    print("-" * 50)
    
    print("1. Data Structure:")
    print(f"   - {sku_df.shape[0]:,} total products")
    print(f"   - {sku_df['Brand'].nunique()} brands")
    print(f"   - {sku_df['Category'].nunique()} categories")
    print(f"   - {sku_df['Flavor'].nunique()} flavors")
    print(f"   - {sku_df['Package_Size'].nunique()} package sizes")
    
    print("\n2. Product Portfolio:")
    print(f"   - Price range: ${sku_df['Price'].min():.2f} - ${sku_df['Price'].max():.2f}")
    print(f"   - Sugar content: {sku_df['Sugar_Content_g_per_100ml'].min():.1f} - {sku_df['Sugar_Content_g_per_100ml'].max():.1f}g per 100ml")
    print(f"   - Caffeine content: {sku_df['Caffeine_Content_mg_per_serving'].min():.1f} - {sku_df['Caffeine_Content_mg_per_serving'].max():.1f}mg per serving")
    print(f"   - Base ships: {sku_df['base_ships'].min():.0f} - {sku_df['base_ships'].max():.0f} units")
    
    print("\n3. Data Quality:")
    print(f"   - Missing values: {missing_data.sum()} total")
    print(f"   - Duplicate rows: {duplicates}")
    print(f"   - Data consistency: {'Good' if missing_data.sum() == 0 and duplicates == 0 else 'Issues found'}")
    
    print("\n4. Business Context:")
    print("   - This is a product master data table")
    print("   - Contains product attributes, pricing, and elasticity information")
    print("   - Links to entity table via SKU_ID")
    print("   - Supports demand forecasting and pricing analysis")
    
    return sku_df

if __name__ == "__main__":
    sku_df = analyze_sku_colddirnks_table()
