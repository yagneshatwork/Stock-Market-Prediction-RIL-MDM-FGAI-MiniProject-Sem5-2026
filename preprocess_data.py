
import pandas as pd

# Step 1: Load the Kaggle dataset
df = pd.read_csv("stock_data.csv")

# Step 2: Clean column names
df.columns = df.columns.astype(str).str.strip()

print("=" * 60)
print("RELIANCE STOCK DATASET")
print("=" * 60)

# Step 3: Display dataset shape
print("\n1. Dataset Shape (Rows, Columns):")
print(df.shape)

# Step 4: Display column names
print("\n2. Column Names:")
print(df.columns.tolist())

# Step 5: Display first five rows
print("\n3. First 5 Rows:")
print(df.head())

# Step 6: Check data types
print("\n4. Data Types:")
print(df.dtypes)

# Step 7: Check missing values
print("\n5. Missing Values:")
print(df.isnull().sum())

# Step 8: Check duplicate rows
print("\n6. Duplicate Rows:")
print(df.duplicated().sum())

# Step 9: Display statistical summary
print("\n7. Statistical Summary:")
print(df.describe(include="all"))

# Step 10: Check date range
print("\n8. Date Range:")

if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

    print("Start Date:", df["Date"].min())
    print("End Date:", df["Date"].max())

else:
    print("Date column not found.")
    print("Check the column names printed above.")

print("\nDataset inspection completed!")
