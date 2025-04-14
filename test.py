import pandas as pd

if __name__ == "__main__":
    # Read the CSV file
    csv_path = r"C:\Users\Guy\Downloads\test_essays (1).csv"
    df = pd.read_csv(csv_path)
    
    # Print column names to see what's available
    print("Available columns in CSV:")
    print(df.columns.tolist())
    
    # Print first few rows to understand the data structure
    print("\nFirst few rows of the data:")
    print(df.head())