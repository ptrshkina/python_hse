# Libraries
import os
import math
import pandas as pd
import numpy as np
import scipy as sc
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Clear console on application start
os.system("cls" if os.name == "nt" else "clear")

# Helpers
class bСolors:
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    END = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def checkValidity():  # Check if there are any null values in dataset
    isValid = not dataFrame.isnull().values.any()
    print(
        "Data is",
        f"{bСolors.GREEN}valid{bСolors.END}" if isValid else f"{bСolors.RED}invalid{bСolors.END}",
    )
    return isValid


# 5. Load dataframe from file
dataFrame = pd.read_csv("./marketing.csv")

# 6. Build plots
print(f"{bСolors.BOLD}Building plots...{bСolors.END}")
dataFrame.groupby("AgeOfStore").median(numeric_only=True).plot.bar(
    stacked=True, y="Promotion"
).get_figure().savefig("bar_age_promotion", dpi=300)
dataFrame.groupby("Promotion").mean(numeric_only=True).plot.bar(
    stacked=True, y="SalesInThousands"
).get_figure().savefig("bar_promotion_sales", dpi=300)
print("Bar charts generated")
dataFrame.groupby("MarketSize").sum().plot.pie(
    y="SalesInThousands",
    figsize=(5, 5),
).get_figure().savefig("pie_market_volume", dpi=300)
print("Pie chart generated")

# 7. Process data with complicated filter
print(f"\n\n{bСolors.BOLD}Filtering data...{bСolors.END}")
targetPromotion = 1
salesMedian = dataFrame.loc[dataFrame["Promotion"] == targetPromotion][
    ["SalesInThousands"]
].median()[0]
print(f"Mean price of sales for promotion {targetPromotion}:", salesMedian)
filtered = dataFrame.loc[
    (dataFrame["Promotion"] == targetPromotion)
    & (dataFrame["SalesInThousands"] > salesMedian)
    & (dataFrame["MarketSize"] == "Small")
]
print(filtered)

# 8. Fill empty values and remove invalid rows
print(f"\n\n{bСolors.BOLD}Validate data...{bСolors.END}")
if not checkValidity():
    rowsToDrop = []
    for i, row in enumerate(dataFrame.values):
        # If marketSize is empty - set from any non empty entry with current marketId
        if row[1] is np.nan:
            print(f"Empty {bСolors.YELLOW}MARKET_SIZE{bСolors.END} found:", row)
            notEmpty = dataFrame.loc[
                (dataFrame["MarketID"] == row[0]) & (dataFrame["MarketSize"] != np.nan)
            ].values[0]
            # Set current MarketSize to notEmpty's market size
            dataFrame.iat[i, 1] = notEmpty[1]
            print(f'    Filled with "{dataFrame.values[i][1]}"')
        # If AgeOfStore is empty
        if math.isnan(row[3]):
            print(f"Empty {bСolors.BLUE}AGE_OF_STORE{bСolors.END} found:", row)
            notEmpty = dataFrame.loc[
                (dataFrame["LocationID"] == row[2]) & (dataFrame["AgeOfStore"].notnull())
            ].values[0]
            dataFrame.iat[i, 3] = notEmpty[3]
            print(f'    Filled with "{dataFrame.values[i][3]}"')
        # If SalesInThousands is empty
        if math.isnan(row[6]):
            print(f"Empty {bСolors.RED}SALES_IN_THOUSANDS{bСolors.END} found", row)
            rowsToDrop.append(i)
            print(f'    Row "{i}" marked as deleteable')
    # Drop rows, marked as deleteable
    dataFrame = dataFrame.drop(rowsToDrop)
# Check validity again to confirm
checkValidity()

# 9: Remove ejections of small market by left trust interval
print(f"\n\n{bСolors.BOLD}Removing ejections...{bСolors.END}")
smallMarketDataFrame = dataFrame.loc[dataFrame["MarketSize"] == "Small"]
# Make distribution plot
smallMarketDataFrame[["SalesInThousands"]].plot.hist(bins=20, alpha=0.5).get_figure().savefig(
    "distribution", dpi=300
)
# Filter values lower than trust interval
trustInterval = 45
filteredDataFrame = smallMarketDataFrame.loc[dataFrame["SalesInThousands"] > trustInterval]
print(filteredDataFrame)
print(
    "Ejections removed:\n",
    smallMarketDataFrame.loc[dataFrame["SalesInThousands"] < trustInterval].values,
)

# 10: Check if data of small market is normal distributed
print(f"\n\n{bСolors.BOLD}Checking data for normality...{bСolors.END}")
alpha = 0.05
# Visual method
smallMarketDataFrame[["SalesInThousands"]].plot.density().get_figure().savefig(
    "distribution_normal", dpi=300
)
# Shapiro-Wilk criteria
stat, p = sc.stats.shapiro(smallMarketDataFrame["SalesInThousands"])
print(f"[Shapiro-Wilk] Alpha={alpha} Statistics={stat}, p-value={p}")
print(
    "    Distribution is",
    f"{bСolors.GREEN}normal{bСolors.END}" if p > alpha else f"{bСolors.RED}abnormal{bСolors.END}",
)
# Pearson criteria
stat, p = sc.stats.normaltest(smallMarketDataFrame["SalesInThousands"])
print(f"[Pearson] Alpha={alpha} Statistics={stat}, p-value={p}")
print(
    "    Distribution is",
    f"{bСolors.GREEN}normal{bСolors.END}" if p > alpha else f"{bСolors.RED}abnormal{bСolors.END}",
)

# 11: Normalize data and fill with unnormalizeable Data
print(f"\n\n{bСolors.BOLD}Normalize data...{bСolors.END}")
dataColumns = ["MarketID", "LocationID", "AgeOfStore", "Promotion", "Week", "SalesInThousands"]
normalize = preprocessing.normalize(filteredDataFrame[dataColumns], axis=0)
normalizedDataFrame = pd.DataFrame(normalize, columns=dataColumns)
normalizedDataFrame["MarketID"] = filteredDataFrame.iloc[:, [0]].values
normalizedDataFrame["LocationID"] = filteredDataFrame.iloc[:, [2]].values
normalizedDataFrame["Week"] = filteredDataFrame.iloc[:, [5]].values
print(normalizedDataFrame)

# 12: Buld correlation matrix
print(f"\n\n{bСolors.BOLD}Build correlation matrix...{bСolors.END}")
fig, ax = plt.subplots()
sns.heatmap(
    normalizedDataFrame.corr(method="pearson", numeric_only=True),
    annot=True,
    fmt=".4f",
    cmap=plt.get_cmap("coolwarm"),
    cbar=False,
    ax=ax,
)
ax.set_yticklabels(ax.get_yticklabels(), rotation="horizontal")
plt.savefig("correlation_matrix.png", bbox_inches="tight", pad_inches=0.0)
print(f"Matrix built")

# 13: Linear regression model
print(f"\n\n{bСolors.BOLD}Training linear regression model...{bСolors.END}")
independant = normalizedDataFrame.loc[:, ("MarketID", "LocationID")]
dependant = normalizedDataFrame.loc[:, "SalesInThousands"]
# Split for training and test models and check if correct
independantTrain, independantTest, dependantTrain, dependantTest = train_test_split(
    independant, dependant, test_size=0.9
)
# Build model
model = LinearRegression().fit(independantTrain, dependantTrain)
dependantPrediction = model.predict(independantTest)
print(dependantPrediction)
# Check workability of metrics
dependantTest = np.exp(dependantTest)
dependantPrediction = np.exp(dependantPrediction)
mse = mean_squared_error(dependantTest, dependantPrediction)
mae = mean_absolute_error(dependantTest, dependantPrediction)
print(
    f"Mean squared error: {bСolors.UNDERLINE}{mse}{bСolors.END}",
    f"Mean absolute error: {bСolors.UNDERLINE}{mae}{bСolors.END}",
)
