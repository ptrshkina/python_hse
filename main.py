# Libraries
import pandas as pd
import numpy as np
import math


def checkValidity():  # Check if there are any null values in dataset
    isValid = not dataFrame.isnull().values.any()
    print(f'Data is {"valid" if isValid else "invalid"}')
    return isValid


# 5. Load dataframe
dataFrame = pd.read_csv('./marketing.csv')
null = np.nan

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
            print('Empty \033[94mMARKET_SIZE\033[0m found:', row)
            notEmpty = dataFrame.loc[(dataFrame['MarketID'] == row[0])
                                     & (dataFrame['MarketSize'] != null)].values[0]
            # Set current MarketSize to notEmpty's market size
            dataFrame.iat[i, 1] = notEmpty[1]
            print(f'    Filled with "{dataFrame.values[i][1]}"')
        # If AgeOfStore is empty
        if (math.isnan(row[3])):
            print('Empty \033[92mAGE_OF_STORE\033[0m found:', row)
            notEmpty = dataFrame.loc[(
                dataFrame['LocationID'] == row[2]) & (dataFrame['AgeOfStore'].notnull())].values[0]
            print(notEmpty)
            dataFrame.iat[i, 3] = notEmpty[3]
            print(f'    Filled with "{dataFrame.values[i][3]}"')
        # If SalesInThousands is empty
        if (math.isnan(row[6])):
            print(f'Empty \033[91mSALES_IN_THOUSANDS\033[0m found', row)
            rowsToDrop.append(i)
            print(f'    Row "{i}" marked as deleteable')
    # Drop rows, marked as deleteable
    dataFrame = dataFrame.drop(rowsToDrop)
# Check validity again to confirm
checkValidity()

# 9: 
