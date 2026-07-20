
# Importing the necessary libraries

import pandas as pd
import numpy as np
import sklearn.impute as skimpu
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Uncomment the import and lines 131 and 132 to view the correlation data structure
# import seaborn as sns
# import matplotlib.pyplot as plt

# Step 1: Data Fidelity - Missing fields and Outliers

# Make a tkinter object and hide the main root window
# root = tk.Tk()
# root.withdraw()

# Get the path to the dataset using the tkinter dialog box
# PATH_TO_DATASET = filedialog.askopenfilename(title="Choose your dataset")


# The path to the dataset
PATH_TO_DATASET = "dataset.csv"

# Read the data with pandas and convert it to a pandas Dataframe
df = pd.read_csv(PATH_TO_DATASET)

# List of numeric data types
NUMERIC_DTYPES = ["int64", "float64"]
target_col_for_model_guess = "TotalPrice" # You can also use the "OrderStatus" column



# Get the mean of the missing values in each column
null_mean = df.isnull().mean()

# Loop through the columns and fix each column according to the percentage of missing values in it 
for col, value in null_mean.items():
    # Calculate the percentage of missing values in the column
    M = value * 100

    # If the percentag of missing items is less than 5% of the data in the column, then delete the empty rows
    if M < 5 and M > 0:
        # Get the bool list of the non-null values
        # bool_list = [not val for i, val in df[col].isnull().items()] does not work as intended
        # Remove the null values from the dataframe
        # df = df[bool_list]
        df = df.dropna(subset=[col])
    # Check if the value of M is between 20% and 5%
    elif M <= 20 and M >= 5:
        # If it is numeric, fill the null values with the median of the column
        if df[col].dtype in NUMERIC_DTYPES:
            df[col] = df[col].fillna(df[col].median())
        
        # If it is not numeric, fill it with the group modes
        else:
            # First we group by a particular column of our choice
            print(df.head(), "\n")
            group_modes = df.groupby("Product")[col].transform(lambda x: x.mode()[0] if not x.mode().empty else np.nan)
            df[col] = df[col].fillna(group_modes)
            # if there are still null values, fill it with the mode value
            if df[col].isnull().mean() > 0:
                df[col] = df[col].fillna(df[col].mode()[0])
            # max_val_list = df[col].value_counts().values.tolist()
            # max_val = df[col].value_counts().max()
            # index_of_max_val = max_val_list.index(max_val)
            # max_item = df[col].value_counts().index[index_of_max_val]
            # # OR max_item = df[col].value_counts().values.tolist().index(df[col].value_counts().max)
            # df.fillna(max_item, inplace=True)

    # if M is greater than 20% use KNN Imputer to fill the empty rows
    elif M > 20:
        
        # get a list of all the values in the column
        column_vals_list = df[col].value_counts().index.tolist()
        if df[col].dtype not in NUMERIC_DTYPES:
            # Replace the non-numeric values with numeric ones for easy calculations with KNN
            df[col] = df[col].map({val:idx for idx,val in enumerate(column_vals_list)})
        numeric_column_list = [column for column in df.columns.to_list() if df[column].dtype in NUMERIC_DTYPES]
        numeric_df = df[numeric_column_list]
        imputer = skimpu.KNNImputer(n_neighbors=10)
        numeric_array = imputer.fit_transform(numeric_df)
        rounded_array = np.round(numeric_array)
        # crate a new dataframe from the transformed data and assign the column back to the dataframe
        numeric_df = pd.DataFrame(rounded_array, columns=numeric_column_list, index=numeric_df.index)
        df[col] = numeric_df[col]
        # df[col] = df[col].map({"SAVE10":0, "FREESHIP": 1, "WINTER15":2})
        df[col] = df[col].map({float(idx):val for idx,val in enumerate(column_vals_list)})
        
        



# A DataFrame of all the numeric columns
numeric_df = df.select_dtypes(include=NUMERIC_DTYPES)

# Loop through the numeric columns and eliminate any outliers (values that are too large or too small compared to other column values)
for column, value in numeric_df.items():
    # Calculate the lower quartile of the column
    q1 = df[column].quantile(0.25)

    # Calculate the upper quartile of the column
    q3 = df[column].quantile(0.75)

    # Calculate the Interquartile range(IQR)
    iqr = q3 - q1
    # Use the IQR to calculate the lower and upper bounds of the column
    column_lower_bound = q1 - (1.5 * iqr)
    column_upper_bound = q3 + (1.5 * iqr)
    
    # Clip/cap the values in the colum using the lower and upper bounds
    df[column] = df[column].clip(lower=column_lower_bound, upper=column_upper_bound)








# Step 2: Eradicating Collinearity, Categorical Encoding, and Feature Engineering

# Part A: Eradicating Collinearity

# Create a correlation matrix to see the correllation values between numerical columns
correlation_matrix = numeric_df.corr().abs()

# Uncomment the code below to plot the correlation values on a graph 
# sns.heatmap(correlation_matrix)
# plt.show()

# Remove the duplicate correlation values by converting the correlation matrix to an array of ones, then to a bool matrix which removes the duplicate correlations
upper_triangle_bool =  np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
upper_triangle_matrix = correlation_matrix.where(upper_triangle_bool)
print(upper_triangle_matrix)

# Make a list of colums that are highly correlated and drop them
columns_to_drop = [column for column in upper_triangle_matrix.columns if (upper_triangle_matrix[column] > 0.8).any()]

# Drops any highly correlated column if present
if columns_to_drop:
    df = df.drop(columns=columns_to_drop)







# Part B: Categorical Encoding

# Get a list of the columns that have less than 10 unique items
low_unique_vals_list = [column for column in df.columns if df[column].dtype not in NUMERIC_DTYPES and df[column].nunique() <= 10]
# Alternatively, use "len(df[column].value_counts().to_list())" (slower method, but it still the same thing as df.nunique())

# The column the model has to guess is removed from the list if it is included in the list
if target_col_for_model_guess in low_unique_vals_list:
    low_unique_vals_list.remove(target_col_for_model_guess)


# Explode the low unique value columns to represent boolean values and drop the original colum to avoid collinear columns
df = pd.get_dummies(data=df, columns=low_unique_vals_list, drop_first=True, dtype=int)





# Part C: Feature Engineering
# Create three new features from the data in the Dataframe 

# 1.Creates a new column called IsWeekend by checking if the date provided is on a weekend and returning a bool value
if "Date" in df.columns.to_list():
    # change the date column to a Datetime object to avaoid repeting code
    df["Date"] = pd.to_datetime(df["Date"])
    df["IsWeekend"] =  df["Date"].dt.day_of_week.isin([5, 6]).astype(int)

    # 2. Calculate the number of years that has passed since the date(transaction age) in the DataFrame
    df["Order_Age"] = 2026 - df["Date"].dt.year


# 3. Divide one numeric column by another to get the ratio of the two
df["Price_Item_Ratio"] = np.round(df["TotalPrice"] / df["ItemsInCart"], decimals=3)







# Step 3: The Predictor Architecture & Splitting Strategy


# create a variable "y", which is the column that the model has to guess
y = df[target_col_for_model_guess]

# Drop colums with highly unique values and split the data into the numeric value columns(X) and target column(y)
cols_to_drop = [column for column in df.columns if df[column].dtype not in NUMERIC_DTYPES and df[column].nunique() >= 50]
# Ensure you also drop the target column so that it is left out of the training data
cols_to_drop.append(target_col_for_model_guess)

X = df.drop(columns=cols_to_drop)






# 1: Splitting the data

# set aside 20 percent of the data for testing and asign the X and y testing and training variables, then set a random seed so th values are reproducable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2: Scaling

# Initialize the scaler
scaler = StandardScaler()

# Calculate the mean and standard deviation of the X training data using "fit_transorm()"
X_train_scaled = scaler.fit_transform(X_train)

# Scale the X test data using "transform()" this time, which forces it to use the mean and variace data calculated from the X train dat
X_test_scaled =  scaler.transform(X_test)

# Turn the scaled arrays back into a pandas DataFrame
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)


