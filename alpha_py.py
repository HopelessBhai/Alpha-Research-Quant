import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


import warnings
warnings.filterwarnings('ignore')

data=pd.read_csv('Merged Df.csv') #combined OHLC data
data.drop(axis=1, columns=['Unnamed: 0'], inplace=True)

class AlphaCheck:
    """
    This class contains methods for creating and testing alphas (quantitative signals) based on stock data.
    """

    def __init__(self):
        """
        Initializes the AlphaCheck class by loading the dataset and preparing new columns such as returns.
        """
        # Load dataset and clean unnecessary columns
        self.data = pd.read_csv('Merged Df.csv')
        self.data.drop(columns=['Unnamed: 0'], inplace=True)
        self.data.rename(columns={'Close': 'close', 'Open': 'open', 'Low': 'low', 'High': 'high'}, inplace=True)

        # Prepare new columns for each symbol in the dataset
        syms = self.data['Symbol'].unique()  # Unique stock symbols
        data = pd.DataFrame()  # Empty DataFrame for processed data
        
        # Process each symbol's data
        for sym in syms:
            df = self.data[self.data['Symbol'] == sym].copy()

            # Create shifted columns for previous and future close prices
            df['close_1'] = df.close.shift(1)  # Previous day's close
            df['close1'] = df.close.shift(-1)  # Next day's close

            # Handle missing values by filling with the current close
            df['close1'].fillna(df['close'], inplace=True)
            df['close_1'].fillna(df['close'], inplace=True)

            # Calculate various return metrics
            df['ret1'] = df['close1'] / df['close'] - 1  # Future returns
            df['ret_1'] = df['close'] / df['close_1'] - 1  # Current returns
            df['oret'] = df['open'] / df['close_1'] - 1  # Overnight returns
            df['tret'] = df['close'] / df['open'] - 1  # Intraday returns

            # Adjust volume by multiplying with close price
            df['Volume'] = df['Volume'] * df['close']

            # Convert the date column to datetime format
            df['Date'] = pd.to_datetime(df['Date'])

            # Append processed data for the current symbol
            data = pd.concat([df, data], ignore_index=True)

        # Store the processed data in the class instance
        self.data = data.copy()

    def neutralize(self, alpha_col, neut_col='Date', func='mean'):
        """
        Neutralizes the given alpha column by removing the mean or median per group (grouped by neut_col).

        Parameters:
        - alpha_col: Column name of the alpha to be neutralized.
        - neut_col: Column to group by (default is 'Date').
        - func: Neutralization method ('mean' or 'median', default is 'mean').
        """
        df = self.data
        if func == 'median':
            df[alpha_col] = df.groupby(neut_col)[alpha_col].transform(lambda x: x - x.median())
        else:
            df[alpha_col] = df.groupby(neut_col)[alpha_col].transform(lambda x: x - x.mean())

    def bucketed_prop(self, alpha_col, bckt_col='Volume', func='mean', buckets=10, clip=False, clip_perc=0.975):
        """
        Returns the bucketed properties (mean/median/std) of the alpha in different volume or date buckets.

        Parameters:
        - alpha_col: The alpha column to analyze.
        - bckt_col: The column to use for bucketing (default is 'Volume').
        - func: The aggregation function to use ('mean', 'median', 'std', default is 'mean').
        - buckets: Number of buckets to divide the data into (default is 10).
        - clip: Whether to clip extreme values (default is False).
        - clip_perc: Clipping percentile (default is 0.975).
        """
        df = self.data.copy()

        # Divide the data into quantile-based buckets
        df[f'Bckt_{bckt_col}'] = pd.qcut(df[bckt_col], buckets)

        if func == 'median':
            return df.groupby(f'Bckt_{bckt_col}')[alpha_col].median()
        if func == 'std':
            return df.groupby(f'Bckt_{bckt_col}')[alpha_col].std()
        return df.groupby(f'Bckt_{bckt_col}')[alpha_col].mean()

    def check_corr(self, alpha_col, return_col, bckt_col='Volume', buckets=10):
        """
        Returns the bucketed correlations of the given alpha with future returns.

        Parameters:
        - alpha_col: The alpha column(s) to check correlation for.
        - return_col: The column representing future returns.
        - bckt_col: The column to use for bucketing (default is 'Volume').
        - buckets: Number of buckets (default is 10).
        """
        df = self.data.copy()

        # Divide the data into quantile-based buckets
        df[f'Bucket_{bckt_col}'] = pd.qcut(df[bckt_col], buckets)

        # Dictionary to store correlations for each alpha
        correlations = {}

        # Function to calculate correlation for each group
        def calculate_corr(group, alpha_col):
            return group[alpha_col].corr(group[return_col])

        # Apply the correlation calculation for each alpha column
        for col in alpha_col:
            correlations[col] = df.groupby(f'Bucket_{bckt_col}').apply(calculate_corr, col)

        return correlations

    def alphaZone(self):
        """
        Creates new alphas (alpha_1, alpha_2, alpha_3, alpha_4) based on specific formulas and stores them in the DataFrame.
        """
        df = self.data

        # Create alphas based on the provided formulas
        df['alpha_1'] = -1 * (df['Volume'] - df['Volume'].shift(1)) * (df['close'] - df['close_1']) / df['close']
        df['alpha_2'] = -1 * (df['close'] / (df['close_1'] + df['close_1'].rolling(window=10).std()))
        df['alpha_3'] = -1 * (df['ret_1']) * df['close'].rolling(window=10).std() * (df['Volume'] - df['Volume'].shift(1))
        df['alpha_3'] = (df['alpha_3'] - df['alpha_3'].mean()) / (df['alpha_3'].max() - df['alpha_3'].min())
        df['alpha_4'] = -1 * (df['low'] - df['close']) * df['open'] ** 5 / ((df['low'] - df['high']) * df['close'] ** 5)

        # Handle missing values by filling with zeros
        df['alpha_1'].fillna(0, inplace=True)
        df['alpha_2'].fillna(0, inplace=True)
        df['alpha_3'].fillna(0, inplace=True)
        df['alpha_4'].fillna(0, inplace=True)

    def ret_df(self):
        """
        Returns the final DataFrame with the newly created alphas and other calculated columns.
        """
        return self.data

# Instantiate the AlphaCheck class
alpha = AlphaCheck()

alpha.alphaZone()
data=alpha.ret_df()

import pandas as pd
from sklearn.cluster import KMeans

class SectorClass:
    """
    This class performs sector clustering, momentum calculation, and correlation checks based on the alphas generated from stock data.
    """

    def __init__(self):
        """
        Initializes the SectorClass by accessing and preparing the alpha data from the AlphaCheck class.
        """
        # Create an instance of AlphaCheck and process the alphas
        alpha = AlphaCheck()
        alpha.alphaZone()
        alpha.neutralize('ret_1')
        self.data = alpha.ret_df()

    def alpha_matrix(self, alpha_col):
        """
        Creates a matrix where rows represent stock symbols, columns represent dates, and values are the specified alpha values.

        Parameters:
        - alpha_col: The column name of the alpha values to use for the matrix.
        
        Returns:
        - A DataFrame with symbols as rows and dates as columns, containing the alpha values.
        """
        dict1 = {}
        
        # Create a dictionary with symbols as keys and alpha values as time series
        for sym in self.data.Symbol.unique():
            df1 = self.data[self.data['Symbol'] == sym]
            dict1[sym] = dict(zip(df1.Date, df1[alpha_col]))
        
        # Create a DataFrame from the dictionary
        matrix = pd.DataFrame(dict1).T
        
        # Drop columns with too many missing values and fill the remaining NaNs with 0
        for cols in matrix.columns.sort_values():
            if matrix[cols].isna().sum() > 60:
                matrix.drop(columns=[cols], inplace=True)
            matrix = matrix.fillna(0)
        
        return matrix

    def cluster_matrix(self, alpha_col='ret_1', look_back=100, n_clusters=13):
        """
        Performs K-Means clustering on the alpha matrix and returns a matrix with cluster labels.

        Parameters:
        - alpha_col: The alpha column to use for clustering (default is 'ret_1').
        - look_back: Number of previous days to use for clustering (default is 100).
        - n_clusters: Number of clusters (default is 13).
        
        Returns:
        - A DataFrame with cluster labels for each symbol over time.
        """
        # Initialize K-Means model
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
        
        # Create the alpha matrix
        df1 = self.alpha_matrix(alpha_col)
        columns = df1.columns
        
        # Perform clustering using the look-back window
        for col_idx in range(len(columns) - 1, look_back, -1):
            X = df1.iloc[:, col_idx - look_back:col_idx]
            y_clusters = kmeans.fit_predict(X)
            df1[columns[col_idx]] = y_clusters
        
        # Store the cluster labels
        self.cluster = df1.iloc[:, look_back + 1:]
        return df1.iloc[:, look_back + 1:]

    def get_clusters(self, df, date):
        """
        Returns the clusters and their corresponding symbols for a given date.
        
        Parameters:
        - df: The DataFrame containing the cluster labels.
        - date: The specific date to retrieve clusters for.
        
        Returns:
        - A dictionary where keys are cluster labels and values are lists of symbols in each cluster.
        """
        df1 = self.cluster_matrix()
        clusters = df1[date].unique()
        cluster_dict = {}
        
        # Group symbols by their cluster labels for the given date
        for i in clusters:
            cluster_dict[i] = df1[df1[date] == i].index.to_list()
        
        return cluster_dict

    def new_matrix(self):
        """
        Combines the cluster labels into the original DataFrame.
        
        Returns:
        - A DataFrame containing the stock data along with the corresponding cluster labels.
        """
        def find_uncommon_elements(list1, list2):
            """
            Helper function to find uncommon elements between two lists.
            """
            set1 = set(list1)
            set2 = set(list2)
            uncommon_elements = set1.symmetric_difference(set2)
            return list(uncommon_elements)
        
        date_cols = self.cluster.columns
        new_df = pd.DataFrame()
        
        # Combine cluster labels with the original DataFrame for each date
        for date in date_cols:
            temp_df = self.data[self.data['Date'] == date]
            syms1 = list(self.cluster[date].index)
            syms2 = list(self.data[self.data['Date'] == date]['Symbol'])
            del_syms = find_uncommon_elements(syms1, syms2)
            temp_df['Cluster'] = self.cluster.drop(del_syms)[date].values
            new_df = pd.concat([temp_df, new_df], ignore_index=True)
        
        self.new_df = new_df
        return new_df

    def mom_matrix(self):
        """
        Calculates sector momentum and residuals for each stock based on its cluster.
        
        Returns:
        - A DataFrame with sector momentum and residual columns.
        """
        matrix1 = self.new_df
        matrix1['Sec_Mom'] = 0
        
        # Calculate sector momentum for each date and cluster
        for date in matrix1['Date'].unique():
            temp = matrix1[matrix1['Date'] == date]
            grp = temp.groupby('Cluster')['ret_1'].mean()
            sec_mom = [(index_val, mom_val) for index_val, mom_val in grp.items()]
            
            # Assign sector momentum to each stock
            for sec, mom in sec_mom:
                matrix1.loc[(matrix1['Cluster'] == sec) & (matrix1['Date'] == date), 'Sec_Mom'] = mom
        
        # Calculate the residual (actual return minus sector momentum)
        matrix1['Residual'] = matrix1['ret_1'] - matrix1['Sec_Mom']
        self.matrix = matrix1
        return matrix1

    def check_corr(self, alpha_col, return_col, bckt_col='Volume', buckets=10):
        """
        Returns the correlation of the sector momentum with future returns, bucketed by a specific column.

        Parameters:
        - alpha_col: The alpha column(s) to check correlation for.
        - return_col: The column representing future returns.
        - bckt_col: The column to use for bucketing (default is 'Volume').
        - buckets: Number of buckets (default is 10).
        
        Returns:
        - A dictionary of correlations between the sector momentum and future returns.
        """
        df = self.matrix
        df[f'Bucket_{bckt_col}'] = pd.qcut(df[bckt_col], buckets)
        correlations = {}

        # Function to calculate correlation for each group
        def calculate_corr(group, alpha_col):
            return group[alpha_col].corr(group[return_col])

        # Apply the correlation calculation for each alpha column
        for col in alpha_col:
            correlations[col] = df.groupby(f'Bucket_{bckt_col}').apply(calculate_corr, col)

        return correlations

# Instantiate the SectorClass
sc = SectorClass()

clstr=sc.cluster_matrix()
temp_df=sc.new_matrix()
mom_df=sc.mom_matrix()

sc.check_corr(['Sec_Mom','Residual','ret_1'],'ret1',bckt_col='Volume',buckets=5)