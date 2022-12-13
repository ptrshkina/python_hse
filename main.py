# Libraries
import pandas as pd
import numpy as np
import scipy as sc
import math


def checkValidity():  # Check if there are any null values in dataset
    isValid = not dataFrame.isnull().values.any()
    print(f'Data is {"valid" if isValid else "invalid"}')
    return isValid

# Helpers


class bcolors:
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


null = np.nan

# 5. Load dataframe
dataFrame = pd.read_csv('./marketing.csv')

# 6. Build plots
dataFrame.groupby('AgeOfStore').median(numeric_only=True).plot.bar(
    stacked=True, y="Promotion").get_figure().savefig('bar_age_promotion', dpi=300)
dataFrame.groupby('Promotion').mean(numeric_only=True).plot.bar(
    stacked=True, y="SalesInThousands").get_figure().savefig('bar_promotion_sales', dpi=300)

# 7. Process data with complicated filter
targetPromotion = 1
salesMedian = dataFrame.loc[dataFrame['Promotion'] ==
                            targetPromotion][['SalesInThousands']].median()[0]
print(f'Mean price of sales for promotion {targetPromotion}:', salesMedian)
filtered = dataFrame.loc[(dataFrame['Promotion'] == targetPromotion)
                         & (dataFrame['SalesInThousands'] > salesMedian)
                         & (dataFrame['MarketSize'] == 'Small')]
print(filtered)

# 8. Fill empty values
if (not checkValidity()):
    rowsToDrop = []
    for i, row in enumerate(dataFrame.values):
        # If marketSize is empty - set from any non empty entry with current marketId
        if (row[1] is null):
            print(f'Empty {bcolors.YELLOW}MARKET_SIZE{bcolors.END} found:', row)
            notEmpty = dataFrame.loc[(dataFrame['MarketID'] == row[0])
                                     & (dataFrame['MarketSize'] != null)].values[0]
            # Set current MarketSize to notEmpty's market size
            dataFrame.iat[i, 1] = notEmpty[1]
            print(f'    Filled with "{dataFrame.values[i][1]}"')
        # If AgeOfStore is empty
        if (math.isnan(row[3])):
            print(
                f'Empty {bcolors.BLUE}AGE_OF_STORE{bcolors.END} found:', row)
            notEmpty = dataFrame.loc[(
                dataFrame['LocationID'] == row[2]) & (dataFrame['AgeOfStore'].notnull())].values[0]
            print(notEmpty)
            dataFrame.iat[i, 3] = notEmpty[3]
            print(f'    Filled with "{dataFrame.values[i][3]}"')
        # If SalesInThousands is empty
        if (math.isnan(row[6])):
            print(
                f'Empty {bcolors.RED}SALES_IN_THOUSANDS{bcolors.END} found', row)
            rowsToDrop.append(i)
            print(f'    Row "{i}" marked as deleteable')
    # Drop rows, marked as deleteable
    dataFrame = dataFrame.drop(rowsToDrop)
# Check validity again to confirm
checkValidity()

# 9: Remove ejections of small market by left trust interval
smallMarketDataFrame = dataFrame.loc[dataFrame['MarketSize'] == 'Small']
# Make distribution plot
smallMarketDataFrame[['SalesInThousands']].plot.hist(
    bins=12, alpha=0.5, density=True, edgecolor='w', linewidth=0.5).get_figure().savefig('distribution', dpi=300)
filteredDataFrame = smallMarketDataFrame.loc[dataFrame['SalesInThousands'] > 40]
print(filteredDataFrame)
print(f'Ejections removed:',
      smallMarketDataFrame.loc[dataFrame['SalesInThousands'] < 40].values)

# 10: Check if normal distribute
# Visual method
smallMarketDataFrame[['SalesInThousands']].plot.density(
).get_figure().savefig('distribution_normal', dpi=300)
alpha = 0.05
# Shapiro-Wilk criteria
stat, p = sc.stats.shapiro(smallMarketDataFrame['SalesInThousands'])
print(f'[Shapiro-Wilk] Alpha={alpha} Statistics={stat}, p-value={p}')
print(f'    Distribution is',
      f"{bcolors.GREEN}normal{bcolors.END}" if p > alpha else f"{bcolors.RED}abnormal{bcolors.END}")
# Pearson criteria
stat, p = sc.stats.normaltest(smallMarketDataFrame['SalesInThousands'])
print(f'[Pearson] Alpha={alpha} Statistics={stat}, p-value={p}')
print(f'    Distribution is',
      f"{bcolors.GREEN}normal{bcolors.END}" if p > alpha else f"{bcolors.RED}abnormal{bcolors.END}")

# 11:

# 12:

# 13:
