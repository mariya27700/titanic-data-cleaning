import pandas as pd

df=pd.read_csv("gender_submission.csv")
#data cleaning

print(df.shape)
print(df.head())
df.info()
print(df.describe(include="all"))
print(df.duplicated().sum())
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)
print(df.columns.tolist())

print(df.isnull().sum())
df["age"]=df["age"].fillna(df["age"].median())
df["cabin"] = df["cabin"].fillna("Unknown")
df["embarked"]=df["embarked"].fillna(df["embarked"].mode()[0])

print(df["sex"].unique())

print(df["embarked"].unique())
print("Invalid passenger classes:")
print(df[~df["pclass"].isin([1, 2, 3])])

print("Invalid survival values:")
print(df[~df["survived"].isin([0, 1])])

print("Negative ages:")
print(df[df["age"] < 0])
print(df[df["fare"] < 0])

#checking for outliers

column = "fare"
Q1 = df[column].quantile(0.25)
Q3 = df[column].quantile(0.75)
IQR= Q3-Q1
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR
print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower limit:", lower_limit)
print("Upper limit:", upper_limit)

outliers = df[
    (df[column] < lower_limit) |
    (df[column] > upper_limit)
]

print("Number of potential outliers:", len(outliers))

print(outliers[["pclass", "fare", "survived"]].head(20))

columns = "age"
Q11 = df[columns].quantile(0.25)
Q31 = df[columns].quantile(0.75)
IQR1= Q31-Q11
lower_limit1 = Q11 - 1.5 * IQR1
upper_limit1 = Q31 + 1.5 * IQR1
print("Q1:", Q11)
print("Q3:", Q31)
print("IQR:", IQR1)
print("Lower limit:", lower_limit1)
print("Upper limit:", upper_limit1)

outliers1 = df[
    (df[columns] < lower_limit1) |
    (df[columns] > upper_limit1)
]
print("Number of potential outliers:", len(outliers1))
print(outliers1["age"].head(10))


print("Final dataset shape:", df.shape)

print("Missing values:")
print(df.isnull().sum())

print("Duplicate rows:", df.duplicated().sum())

print("Data types:")
print(df.dtypes)

df.to_csv("cleaned_titanic.csv", index=False)

print("Cleaned dataset saved successfully!")