# Summary

The program allows to evaluate selected dataset for the presence of a correlation and build up a model that allows you to predict the number of sales in future periods under certain conditions (linear regression model)
Formulation of the problem can be found in [task.pdf](task.pdf)

**Author**: Elizaveta Petrukhina, Anastasiia Kireecheva

## Target:

To determine, based on the selected dataframe, the relationship between the quantity of sales of a particular store on factors such as the size and location of the market, as well as the location of a particular store, its age and level of promotion

## Hypothesis

**H0:**
Quantity of sales depends on the level of promotion and the age of store

**Result**
hypothesis was not confirmed since there was a weak positive correlation between the number of sales and the expected factors, therefore quantity of sales depends on such variables as market id (region of a particular market) and location id (location of a particular store in the market)

## Tasks

- To sanitize the dataframe from gaps and check them for normal distribution
- To remove ejections from the data
- To conduct correlation analysis based on normalized data
- Based on normalized data conduct correlation analysis and build multiple linear regression model

# Dataset structure

| MarketID |      MarketSize      | LocationID | AgeOfStore |       Promotion       |    Week     | SalesInThousands |
| :------: | :------------------: | :--------: | :--------: | :-------------------: | :---------: | :--------------: |
| integer  |        string        |  integer   |  integer   |        integer        |   integer   |      float       |
|   city   |   city market size   |   store    | store age  | store promotion level | week number |   weekly sales   |
|  index   | Small, Medium, Large |   index    |  positive  |          1-3          |     1-4     |      0-100       |

# Useful links

[Download python](https://www.python.org/downloads/)

[Pandas docs](https://pandas.pydata.org/docs/user_guide/index.html)

[Index in groupping issue](https://stackoverflow.com/questions/19202093/how-to-select-columns-from-groupby-object-in-pandas)

[Guide to linear regression](https://python-school.ru/blog/linear-regression-basis/)

# Install libraries

```sh
# Windows
pip install matplotlib pandas numpy scipy scikit-learn seaborn

# Mac OS
pip3 install matplotlib pandas numpy scipy scikit-learn seaborn
```

# Run application

```sh
# Windows
python main.py

# Mac OS
python3 main.py
```

# Code formatting

1. [Install BlackFormatter extention for VS Code](https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter)
2. Insstall 'black' library for python

```sh
# Windows
pip install --upgrade black

# Mac OS
pip3 install --upgrade black
```

3. Set up vscode global `settings.json`

```json
{
  "[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter"
  }
}
```

# Dependencies

|   Package    |             Description             |
| :----------: | :---------------------------------: |
|  matplotlib  | Visualisation dependency of pandas  |
|    pandas    |   Data analisys and manipulation    |
|    numpy     | Matrix and fundamental math support |
|    scipy     |  Statistics and data optimization   |
| scikit-learn |   Machine learning data analisys    |
|   seaborn    |   Statistical data visualisation    |
