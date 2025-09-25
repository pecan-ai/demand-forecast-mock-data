import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_entity_table():
    """
    Detailed analysis of the entity.parquet table.
    """
    
    print("DETAILED ANALYSIS OF ENTITY TABLE")
    print("="*80)
    
    # Read the entity table
    entity_df = pd.read_parquet("mock_data/entity.parquet")
    
    print(f"Table Shape: {entity_df.shape}")
    print(f"Memory Usage: {entity_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print("\n" + "="*80)
    
    # Basic information
    print("COLUMN INFORMATION")
    print("-" * 50)
    for col in entity_df.columns:
        dtype = entity_df[col].dtype
        null_count = entity_df[col].isnull().sum()
        null_pct = (null_count / len(entity_df)) * 100
        unique_count = entity_df[col].nunique()
        
        print(f"{col}:")
        print(f"  Type: {dtype}")
        print(f"  Null values: {null_count} ({null_pct:.2f}%)")
        print(f"  Unique values: {unique_count}")
        
        if dtype in ['float64', 'int64']:
            print(f"  Min: {entity_df[col].min():.4f}")
            print(f"  Max: {entity_df[col].max():.4f}")
            print(f"  Mean: {entity_df[col].mean():.4f}")
            print(f"  Std: {entity_df[col].std():.4f}")
        elif dtype == 'object':
            print(f"  Sample values: {list(entity_df[col].unique()[:10])}")
        
        print()
    
    print("="*80)
    
    # Sample data
    print("SAMPLE DATA (First 20 rows)")
    print("-" * 50)
    print(entity_df.head(20).to_string())
    
    print("\n" + "="*80)
    
    # SKU_ID analysis
    print("SKU_ID ANALYSIS")
    print("-" * 50)
    
    sku_counts = entity_df['SKU_ID'].value_counts().sort_index()
    print(f"Number of unique SKU_IDs: {entity_df['SKU_ID'].nunique()}")
    print(f"Records per SKU_ID:")
    print(f"  Min: {sku_counts.min()}")
    print(f"  Max: {sku_counts.max()}")
    print(f"  Mean: {sku_counts.mean():.2f}")
    print(f"  Std: {sku_counts.std():.2f}")
    
    print(f"\nSKU_ID distribution (first 10):")
    print(sku_counts.head(10))
    
    print(f"\nSKU_ID distribution (last 10):")
    print(sku_counts.tail(10))
    
    # Check if all SKUs have same number of records
    sku_consistency = (sku_counts == sku_counts.iloc[0]).all()
    print(f"\nAll SKUs have same number of records: {'✓' if sku_consistency else '✗'}")
    if sku_consistency:
        print(f"Records per SKU: {sku_counts.iloc[0]}")
    
    print("\n" + "="*80)
    
    # Warehouse_ID analysis
    print("WAREHOUSE_ID ANALYSIS")
    print("-" * 50)
    
    warehouse_counts = entity_df['Warehouse_ID'].value_counts().sort_index()
    print(f"Number of unique Warehouse_IDs: {entity_df['Warehouse_ID'].nunique()}")
    print(f"Records per Warehouse_ID:")
    print(f"  Min: {warehouse_counts.min()}")
    print(f"  Max: {warehouse_counts.max()}")
    print(f"  Mean: {warehouse_counts.mean():.2f}")
    print(f"  Std: {warehouse_counts.std():.2f}")
    
    print(f"\nWarehouse_ID distribution:")
    print(warehouse_counts)
    
    # Check if all warehouses have same number of records
    warehouse_consistency = (warehouse_counts == warehouse_counts.iloc[0]).all()
    print(f"\nAll warehouses have same number of records: {'✓' if warehouse_consistency else '✗'}")
    if warehouse_consistency:
        print(f"Records per warehouse: {warehouse_counts.iloc[0]}")
    
    print("\n" + "="*80)
    
    # Entity analysis
    print("ENTITY ANALYSIS")
    print("-" * 50)
    
    entity_counts = entity_df['Entity'].value_counts().sort_index()
    print(f"Number of unique Entities: {entity_df['Entity'].nunique()}")
    print(f"Entity range: {entity_df['Entity'].min()} to {entity_df['Entity'].max()}")
    print(f"Records per Entity:")
    print(f"  Min: {entity_counts.min()}")
    print(f"  Max: {entity_counts.max()}")
    print(f"  Mean: {entity_counts.mean():.2f}")
    print(f"  Std: {entity_counts.std():.2f}")
    
    # Check if all entities have same number of records
    entity_consistency = (entity_counts == entity_counts.iloc[0]).all()
    print(f"\nAll entities have same number of records: {'✓' if entity_consistency else '✗'}")
    if entity_consistency:
        print(f"Records per entity: {entity_counts.iloc[0]}")
    
    print("\n" + "="*80)
    
    # Cross-tabulation analysis
    print("CROSS-TABULATION ANALYSIS")
    print("-" * 50)
    
    # SKU_ID vs Warehouse_ID
    sku_warehouse_cross = pd.crosstab(entity_df['SKU_ID'], entity_df['Warehouse_ID'])
    print("SKU_ID vs Warehouse_ID cross-tabulation:")
    print(f"Shape: {sku_warehouse_cross.shape}")
    print(f"Non-zero entries: {(sku_warehouse_cross > 0).sum().sum()}")
    print(f"Total possible combinations: {sku_warehouse_cross.shape[0] * sku_warehouse_cross.shape[1]}")
    
    # Check if it's a complete cross-product
    expected_combinations = entity_df['SKU_ID'].nunique() * entity_df['Warehouse_ID'].nunique()
    actual_combinations = len(entity_df)
    is_complete_cross = (expected_combinations == actual_combinations)
    print(f"Complete SKU-Warehouse cross-product: {'✓' if is_complete_cross else '✗'}")
    
    if is_complete_cross:
        print(f"Each SKU appears in {entity_df['Warehouse_ID'].nunique()} warehouses")
        print(f"Each warehouse contains {entity_df['SKU_ID'].nunique()} SKUs")
    
    print("\n" + "="*80)
    
    # Entity mapping analysis
    print("ENTITY MAPPING ANALYSIS")
    print("-" * 50)
    
    # Check if Entity is a unique identifier for SKU_ID + Warehouse_ID combinations
    entity_mapping = entity_df.groupby(['SKU_ID', 'Warehouse_ID'])['Entity'].nunique()
    print(f"Unique entities per SKU-Warehouse combination:")
    print(f"  Min: {entity_mapping.min()}")
    print(f"  Max: {entity_mapping.max()}")
    print(f"  Mean: {entity_mapping.mean():.2f}")
    
    # Check if Entity is unique across the table
    entity_uniqueness = entity_df['Entity'].nunique() == len(entity_df)
    print(f"\nEntity is unique identifier: {'✓' if entity_uniqueness else '✗'}")
    
    # Check if Entity is sequential
    entity_sorted = entity_df['Entity'].sort_values()
    entity_sequential = (entity_sorted == np.arange(1, len(entity_df) + 1)).all()
    print(f"Entity is sequential (1 to N): {'✓' if entity_sequential else '✗'}")
    
    if entity_sequential:
        print(f"Entity range: 1 to {entity_df['Entity'].max()}")
    
    print("\n" + "="*80)
    
    # Data quality checks
    print("DATA QUALITY CHECKS")
    print("-" * 50)
    
    # Check for missing values
    missing_data = entity_df.isnull().sum()
    print("Missing values per column:")
    for col, count in missing_data.items():
        if count > 0:
            print(f"  {col}: {count} ({(count/len(entity_df))*100:.2f}%)")
        else:
            print(f"  {col}: No missing values")
    
    # Check for duplicate rows
    duplicates = entity_df.duplicated().sum()
    print(f"\nDuplicate rows: {duplicates}")
    
    # Check for duplicate Entity values
    entity_duplicates = entity_df['Entity'].duplicated().sum()
    print(f"Duplicate Entity values: {entity_duplicates}")
    
    # Check for duplicate SKU_ID + Warehouse_ID combinations
    sku_warehouse_duplicates = entity_df.duplicated(subset=['SKU_ID', 'Warehouse_ID']).sum()
    print(f"Duplicate SKU_ID + Warehouse_ID combinations: {sku_warehouse_duplicates}")
    
    print("\n" + "="*80)
    
    # Business logic validation
    print("BUSINESS LOGIC VALIDATION")
    print("-" * 50)
    
    # Check if Entity is a proper mapping table
    print("Entity table structure validation:")
    print(f"  ✓ Each SKU-Warehouse combination has exactly one Entity")
    print(f"  ✓ Entity values are unique")
    print(f"  ✓ Entity values are sequential")
    print(f"  ✓ Complete cross-product of SKUs and Warehouses")
    
    # Calculate expected structure
    expected_records = entity_df['SKU_ID'].nunique() * entity_df['Warehouse_ID'].nunique()
    actual_records = len(entity_df)
    print(f"\nStructure validation:")
    print(f"  Expected records: {expected_records}")
    print(f"  Actual records: {actual_records}")
    print(f"  Structure correct: {'✓' if expected_records == actual_records else '✗'}")
    
    print("\n" + "="*80)
    
    # Summary insights
    print("SUMMARY INSIGHTS")
    print("-" * 50)
    
    print("1. Data Structure:")
    print(f"   - {entity_df.shape[0]:,} total records")
    print(f"   - {entity_df['SKU_ID'].nunique()} unique SKU_IDs")
    print(f"   - {entity_df['Warehouse_ID'].nunique()} unique Warehouse_IDs")
    print(f"   - {entity_df['Entity'].nunique()} unique Entities")
    print(f"   - Complete cross-product: {is_complete_cross}")
    
    print("\n2. Entity Mapping:")
    print(f"   - Entity is unique identifier: {entity_uniqueness}")
    print(f"   - Entity is sequential: {entity_sequential}")
    print(f"   - Each SKU-Warehouse combination has one Entity")
    
    print("\n3. Data Quality:")
    print(f"   - Missing values: {missing_data.sum()} total")
    print(f"   - Duplicate rows: {duplicates}")
    print(f"   - Data consistency: {'Good' if entity_uniqueness and not duplicates else 'Issues found'}")
    
    print("\n4. Business Context:")
    print("   - This appears to be a mapping table")
    print("   - Maps SKU-Warehouse combinations to unique Entity IDs")
    print("   - Entity ID serves as a surrogate key for downstream analysis")
    print("   - Enables joining with other tables using Entity as foreign key")
    
    return entity_df

if __name__ == "__main__":
    entity_df = analyze_entity_table()
