import pandas as pd
import os
from pathlib import Path
import numpy as np

def analyze_parquet_files():
    """
    Analyze all parquet files in the mock_data directory to understand table relationships.
    """
    
    # Define the mock_data directory
    mock_data_dir = Path("mock_data")
    
    if not mock_data_dir.exists():
        print("Error: mock_data directory not found!")
        return
    
    # Get all parquet files
    parquet_files = list(mock_data_dir.glob("*.parquet"))
    
    if not parquet_files:
        print("No parquet files found in mock_data directory!")
        return
    
    print(f"Found {len(parquet_files)} parquet files:")
    for file in parquet_files:
        print(f"  - {file.name}")
    print("\n" + "="*80 + "\n")
    
    # Dictionary to store all dataframes and their metadata
    dataframes = {}
    table_info = {}
    
    # Read each parquet file and extract sample data
    for file_path in parquet_files:
        try:
            print(f"Analyzing: {file_path.name}")
            print("-" * 50)
            
            # Read the parquet file
            df = pd.read_parquet(file_path)
            dataframes[file_path.stem] = df
            
            # Store basic information
            table_info[file_path.stem] = {
                'file_name': file_path.name,
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'sample_data': df.head(3).to_dict('records'),
                'null_counts': df.isnull().sum().to_dict(),
                'unique_counts': df.nunique().to_dict()
            }
            
            # Display basic info
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print(f"Data types:")
            for col, dtype in df.dtypes.items():
                print(f"  {col}: {dtype}")
            
            print(f"\nSample data (first 3 rows):")
            print(df.head(3).to_string())
            
            print(f"\nNull value counts:")
            null_counts = df.isnull().sum()
            for col, count in null_counts.items():
                if count > 0:
                    print(f"  {col}: {count}")
            
            print(f"\nUnique value counts:")
            unique_counts = df.nunique()
            for col, count in unique_counts.items():
                print(f"  {col}: {count}")
            
            print("\n" + "="*80 + "\n")
            
        except Exception as e:
            print(f"Error reading {file_path.name}: {str(e)}")
            print("\n" + "="*80 + "\n")
    
    # Analyze relationships between tables
    print("RELATIONSHIP ANALYSIS")
    print("="*80)
    
    # Look for common columns that might indicate relationships
    all_columns = {}
    for table_name, info in table_info.items():
        for col in info['columns']:
            if col not in all_columns:
                all_columns[col] = []
            all_columns[col].append(table_name)
    
    # Find columns that appear in multiple tables (potential foreign keys)
    common_columns = {col: tables for col, tables in all_columns.items() if len(tables) > 1}
    
    print("Columns that appear in multiple tables (potential foreign keys):")
    for col, tables in common_columns.items():
        print(f"  {col}: {', '.join(tables)}")
    
    print("\n" + "="*80)
    
    # Analyze specific relationships
    print("DETAILED RELATIONSHIP ANALYSIS")
    print("="*80)
    
    # Look for ID columns and their relationships
    id_columns = {}
    for table_name, info in table_info.items():
        for col in info['columns']:
            if 'id' in col.lower() or col.lower().endswith('_id'):
                if col not in id_columns:
                    id_columns[col] = []
                id_columns[col].append(table_name)
    
    print("ID columns and their relationships:")
    for col, tables in id_columns.items():
        print(f"  {col}: {', '.join(tables)}")
        if len(tables) > 1:
            # Check if values in this column are consistent across tables
            for table in tables:
                if table in dataframes:
                    unique_values = dataframes[table][col].nunique()
                    print(f"    {table}: {unique_values} unique values")
    
    print("\n" + "="*80)
    
    # Generate a summary report
    print("SUMMARY REPORT")
    print("="*80)
    
    print("Table Summary:")
    for table_name, info in table_info.items():
        print(f"\n{table_name}:")
        print(f"  File: {info['file_name']}")
        print(f"  Shape: {info['shape']}")
        print(f"  Columns: {info['columns']}")
    
    print(f"\nTotal tables: {len(table_info)}")
    print(f"Total columns across all tables: {len(all_columns)}")
    print(f"Common columns (potential relationships): {len(common_columns)}")
    
    return dataframes, table_info, common_columns

if __name__ == "__main__":
    dataframes, table_info, common_columns = analyze_parquet_files()
