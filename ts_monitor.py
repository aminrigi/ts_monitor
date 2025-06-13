from statsmodels.tsa.seasonal import STL, MSTL
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from skchange.change_detectors import PELT
from skchange.costs import L2Cost
from skchange.costs.gaussian_cost import GaussianCost
import ruptures as rpt



class TSMonitor:
    """
    A class to monitor a ts
    """

    ## constants
    WEEKLY_PERIOD = 7
    YEARLY_PERIOD = 365

    def __init__(self, ts_id:str, ts_df:pd.DataFrame):
        # Input validation
        required_columns = {'ds', 'y'}
        if not required_columns.issubset(ts_df.columns):
            raise ValueError(f"ts_df must contain the following columns: {required_columns}. Found: {ts_df.columns.tolist()}")
        self.ts_id = ts_id

        # Ensure only 'ds' and 'y' columns are kept
        ts_df = ts_df[['ds', 'y']]

        # Clean the DataFrame
        self._clean_ts_df(ts_df)

        self._trend = None
        self._seasonal = None
        self._residual = None

############ ### GETTERS AND SETTERS #######################
    def get_id(self):
        return self.ts_id

    def get_ts(self):
        return self.ts_df
    

    def get_full_analysis(self):
        """
        Get a DataFrame with the full analysis, including ds, y, season, trend, residuals, and outlier.

        Returns:
            pd.DataFrame: A DataFrame with the specified columns.
        """
        if self._trend is None or self._seasonal is None or self._residual is None:
            raise ValueError("STL decomposition has not been performed. Call `stl_decompose` first.")
        if self.outliers_df is None:
            raise ValueError("Outliers have not been detected. Call `detect_outliers_quantile` first.")

        # Combine all components into a single DataFrame
        full_analysis = self.ts_df.copy()
        full_analysis['trend'] = self._trend
        full_analysis['season'] = self._seasonal
        full_analysis['residuals'] = self._residual
        full_analysis['outlier'] = self.outliers_df['outlier']
        full_analysis['change_points'] = self.change_points['change_points']

        return full_analysis
        

    #############  OUTLIER DETECTION #######################


        

    def decompose_stl(self, period=WEEKLY_PERIOD):
        """
        Perform STL decomposition on the time series.

        Parameters:
            seasonal (int): The seasonal window size for STL decomposition.

        Returns:
            None: The method modifies the instance variables directly.            
        """
        if 'y' not in self.ts_df.columns:
            raise ValueError("The time series data must have a 'y' column for values.")

        # Perform STL decomposition
        stl = STL(self.ts_df['y'], period=period)
        result = stl.fit()

        # Return the decomposed components
        
        self._trend = result.trend
        self._seasonal = result.seasonal
        self._residual = result.resid



    def decompose_mstl(self, periods=(WEEKLY_PERIOD, YEARLY_PERIOD)):
        """
        Perform MSTL decomposition on the time series to handle multiple seasonalities.

        Parameters:
            periods (list): A list of seasonal periods to decompose (e.g., [7, 365] for weekly and yearly seasonality).

        Returns:
            None: The method modifies the instance variables directly.
        """
        if 'y' not in self.ts_df.columns:
            raise ValueError("The time series data must have a 'y' column for values.")

        # Perform MSTL decomposition
        mstl = MSTL(self.ts_df['y'], periods=periods)
        result = mstl.fit()

        # Extract components
        self._trend = result.trend
        self._seasonal = result.seasonal.sum(axis=1)  # Combine all seasonal components
        self._residual = result.resid
    
    def detect_outliers(self, lower_quantile=0.02, upper_quantile=0.98):
        """
        Detect outliers in the residuals using quantile thresholds.

        Parameters:
            lower_quantile (float): The lower quantile threshold for detecting outliers.
            upper_quantile (float): The upper quantile threshold for detecting outliers.

        Returns:
            None: The method modifies the instance variable `outliers` directly.
        """
        if self._residual is None:
            raise ValueError("STL decomposition has not been performed. Call `stl_decompose` first.")

        # Drop NaN values from residuals
        residuals = self._residual.dropna()

        # Calculate quantile thresholds
        lower_bound = residuals.quantile(lower_quantile)
        upper_bound = residuals.quantile(upper_quantile)

        # Identify outliers
        outliers = (residuals < lower_bound) | (residuals > upper_bound)

        # Combine residuals, outliers, and the 'ds' column
        self.outliers = pd.DataFrame({
            'ds': residuals.index,  # Add the 'ds' column
            'residuals': residuals,
            'outlier': outliers
        }).reset_index(drop=True)
    
    #############################################################
    ################  CHANGE POINT DETECTION #############

    # why pelt? https://arxiv.org/pdf/2003.06222
    # applying pelt on trend+resid: https://pmc.ncbi.nlm.nih.gov/articles/PMC5546760/
    # choose min value of penalty: https://stats.stackexchange.com/questions/616623/what-is-a-reasonable-range-of-penalty-values-to-try-in-pelt-changepoint-analysis

    def detect_change_points_pelt(self, method="deseasoned", pelt_penalty=None): #deaseasoned= trend+residual
        """
        Detect change points in the deseasonalized time series (trend + residual) using skchange's PELT detector.

        Parameters:
            method (str): The method to use for change point detection. Options: "deseasoned" (trend + residual) or "full" (y).
            penalty (float): The penalty value for the change point detection algorithm.

        Returns:
            list: A list of detected change points.
        """
        if self._trend is None or self._residual is None:
            raise ValueError("STL decomposition has not been performed. Call `stl_decompose` first.")


        # Calculate the deseasonalized time series (trend + residual)
        if method == "deseasoned":
            ts = self._trend + self._residual
        elif method == "full":
            ts = self.ts_df['y']  # Use the original time series if method is not "deseasoned"
        else:
            raise ValueError("Invalid method. Use 'deseasoned' or 'full'.")

        # Convert the deseasonalized time series to a 2D array (required by skchange)
        signal = ts.values.reshape(-1, 1)

        if pelt_penalty is None:
            # set default penalty based on BIC (Bayesian Information Criterion)
            pelt_penalty = np.log(len(signal)) * 1.8 # replaced 2 with 1.8 to be more sensitive to change points

        # Initialize the PELT detector with the selected cost function
        detector = PELT(GaussianCost(), penalty=pelt_penalty)

        # Fit the detector and predict change points
        detected_change_points = detector.fit_predict(signal)

        change_point_indices = detected_change_points['ilocs'].tolist()

        # Map the change point indices to the corresponding datetime values
        change_point_timestamps = self.ts_df.iloc[change_point_indices].index

        # Create a DataFrame with 'ds' and 'change_points'
        change_points_df = self.ts_df[['ds']].copy()
        change_points_df['change_points'] = False  # Initialize all as False
        change_points_df.loc[change_point_timestamps, 'change_points'] = True  # Mark change points as True

        change_points_df = change_points_df.reset_index(drop=True)

        self.change_points = change_points_df
        return detected_change_points
    


    ############### PLOT FUNCTIONS #######################

    def plot_outliers(self):
        """
        Plot the residuals and highlight the outliers.

        Returns:
            None: Displays the plot.
        """
        if self._residual is None:
            raise ValueError("STL decomposition has not been performed. Call `stl_decompose` first.")
        if self.outliers is None:
            raise ValueError("Outliers have not been detected. Call `detect_outliers_quantile` or equivalent first.")

        # Ensure the outliers DataFrame contains the required columns
        if 'ds' not in self.outliers.columns or 'residuals' not in self.outliers.columns or 'outlier' not in self.outliers.columns:
            raise ValueError("The outliers DataFrame must contain 'ds', 'residuals', and 'outlier' columns.")

        # Plot the residuals and highlight the outliers
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=self.outliers,
            x='ds',
            y='residuals',
            hue='outlier',
            palette={True: 'red', False: 'blue'}
        )
        plt.title(f'Outliers Detection in Residuals for {self.ts_id}')
        plt.xlabel('Date')
        plt.ylabel('Residuals')
        plt.grid(True)
        plt.show()


    def plot_stl(self):
        """
        Plot the STL decomposition components (trend, seasonal, residual).

        Returns:
            None: Displays the plot.
        """
        if self._trend is None or self._seasonal is None or self._residual is None:
            raise ValueError("STL decomposition has not been performed. Call `stl_decompose` first.")

        # Create the plot
        fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
        
        # Original time series
        axes[0].plot(self.ts_df.index, self.ts_df['y'], label='Original', color='blue')
        axes[0].set_title('Original Time Series')
        axes[0].legend()

        # Trend component
        axes[1].plot(self.ts_df.index, self._trend, label='Trend', color='green')
        axes[1].set_title('Trend Component')
        axes[1].legend()

        # Seasonal component
        axes[2].plot(self.ts_df.index, self._seasonal, label='Seasonal', color='orange')
        axes[2].set_title('Seasonal Component')
        axes[2].legend()

        # Residual component
        axes[3].plot(self.ts_df.index, self._residual, label='Residual', color='red')
        axes[3].set_title('Residual Component')
        axes[3].legend()

        # Add labels
        plt.xlabel('Date')
        plt.tight_layout()
        plt.show()


    def plot_deseasoned(self, moving_avg_window=1, show_change_points=False):
        """
        Plot the deseasonalized time series (residual + trend) with an optional moving average.

        Parameters:
            moving_avg_window (int): The size of the moving average window. Default is 1 (no smoothing).
            show_change_points (bool): Whether to highlight change points on the plot. Default is False.

        Returns:
            None: Displays the plot.
        """
        if self._trend is None or self._residual is None:
            raise ValueError("STL decomposition has not been performed. Call `stl_decompose` first.")

        # Calculate the deseasonalized time series
        deseasoned = self._trend + self._residual

        # Apply moving average if the window size is greater than 1
        if moving_avg_window > 1:
            deseasoned = deseasoned.rolling(window=moving_avg_window).mean()

        # Plot the deseasonalized time series
        plt.figure(figsize=(10, 6))
        plt.plot(self.ts_df['ds'], deseasoned, label=f'Deseasonalized (Trend + Residual, MA={moving_avg_window})', color='blue')

        # Highlight change points if show_change_points is True
        if show_change_points:
            if self.change_points is None:
                raise ValueError("Change points have not been detected. Call `detect_change_points_pelt` first.")
            
            change_points = self.change_points[self.change_points['change_points']]
            plt.scatter(change_points['ds'], deseasoned.loc[change_points['ds']], color='red', label='Change Points', zorder=5)

        plt.title(f'Deseasonalized Time Series for {self.ts_id}')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid()
        plt.show()

    def plot_original(self, moving_avg_window=1, show_change_points=False):
        """
        Plot the original time series (y) with an optional moving average and change points.

        Parameters:
            moving_avg_window (int): The size of the moving average window. Default is 1 (no smoothing).
            show_change_points (bool): Whether to highlight change points on the plot. Default is False.

        Returns:
            None: Displays the plot.
        """
        if 'y' not in self.ts_df.columns:
            raise ValueError("The time series data must have a 'y' column for values.")

        # Extract the original time series
        original_ts = self.ts_df['y']

        # Apply moving average if the window size is greater than 1
        if moving_avg_window > 1:
            smoothed_ts = original_ts.rolling(window=moving_avg_window).mean()
        else:
            smoothed_ts = original_ts

        # Plot the original time series
        plt.figure(figsize=(10, 6))
        plt.plot(self.ts_df['ds'], smoothed_ts, label=f'Original Time Series (MA={moving_avg_window})', color='blue')

        # Highlight change points if show_change_points is True
        if show_change_points:
            if self.change_points is None:
                raise ValueError("Change points have not been detected. Call `detect_change_points_pelt` or `detect_change_points_ruptures` first.")
            
            change_points = self.change_points[self.change_points['change_points']]
            plt.scatter(change_points['ds'], smoothed_ts.loc[change_points['ds']], color='red', label='Change Points', zorder=5)

        plt.title(f'Original Time Series for {self.ts_id}')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid()
        plt.show()

############################################################################
################ Private Methods ############################

    def _clean_ts_df(self, ts_df):
        """
        Clean the time series DataFrame by sorting, handling missing dates, and setting 'ds' as the index.
        Logic:
        1. Ensure the 'ds' column is in datetime format.
        2. Sort the DataFrame by the 'ds' column to ensure chronological order.
        3. Create a complete date range from the minimum to the maximum date in the 'ds' column.
        4. Reindex the DataFrame to include all dates in the range, filling missing dates with NaN.
        5. Handle missing values in the 'y' column:
            - For weekends (Saturday and Sunday), replace missing values with 0.
            - For weekdays:
                a. If both the previous and next week's same weekday values are available, replace the missing value with their average.
                b. If only one of the previous or next week's same weekday values is available, use that value.
                c. If neither is available, replace the missing value with the maximum of the closest available previous and next dates.
        6. Set the 'ds' column as the index of the DataFrame.


        Parameters:
            ts_df (pd.DataFrame): The input DataFrame with 'ds' (dates) and 'y' (values).

        Returns:
            pd.DataFrame: The cleaned DataFrame.
        """

        # Create a copy of the DataFrame to avoid modifying the original
        ts_df = ts_df.copy()

        # Ensure 'ds' is a datetime column
        ts_df['ds'] = pd.to_datetime(ts_df['ds'])

        # Sort the DataFrame by dates
        ts_df = ts_df.sort_values(by='ds').reset_index(drop=True)

        # Create a complete date range
        full_date_range = pd.date_range(start=ts_df['ds'].min(), end=ts_df['ds'].max(), freq='D')

        # Reindex the DataFrame to include all dates in the range
        ts_df = ts_df.set_index('ds').reindex(full_date_range).rename_axis('ds').reset_index()


            # Check if the number of missing values exceeds 5% of the time series size
        missing_count = ts_df['y'].isna().sum()
        total_count = len(ts_df)
        if missing_count > 0.05 * total_count:
            raise ValueError(
                f"The number of missing values ({missing_count}) exceeds 5% of the time series size ({total_count}). "
                "Please handle missing values differently before proceeding."
            )

        # Handle missing values in 'y'
        ts_df['y'] = ts_df['y'].fillna(0)  # Initialize missing values with 0

        for idx, row in ts_df[ts_df['y'] == 0].iterrows():
            current_date = row['ds']
            weekday = current_date.weekday()  # 0 = Monday, 6 = Sunday

            if weekday in [5, 6]:  # Weekend (Saturday or Sunday)
                ts_df.loc[idx, 'y'] = 0  # Replace missing weekend values with 0
            else:  # Weekday
                # Get the same weekday from the previous and next week
                prev_week_date = current_date - pd.Timedelta(weeks=1)
                next_week_date = current_date + pd.Timedelta(weeks=1)

                prev_value = ts_df.loc[ts_df['ds'] == prev_week_date, 'y'].values
                next_value = ts_df.loc[ts_df['ds'] == next_week_date, 'y'].values

                # Handle cases where one or both values are missing
                if len(prev_value) > 0 and len(next_value) > 0:
                    ts_df.loc[idx, 'y'] = (prev_value[0] + next_value[0]) / 2
                elif len(prev_value) > 0:
                    ts_df.loc[idx, 'y'] = prev_value[0]
                elif len(next_value) > 0:
                    ts_df.loc[idx, 'y'] = next_value[0]
                else:
                    # Use the maximum of the closest available previous and next dates
                    prev_closest = ts_df.loc[ts_df['ds'] < current_date, 'y'].max()
                    next_closest = ts_df.loc[ts_df['ds'] > current_date, 'y'].max()
                    ts_df.loc[idx, 'y'] = max(prev_closest, next_closest)

        # Set 'ds' as the index
        ts_df = ts_df.set_index('ds', drop=False)
        self.ts_df = ts_df
        

          