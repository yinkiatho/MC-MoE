from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    median_absolute_error,
    explained_variance_score,
    root_mean_squared_error
)
import pandas as pd
import numpy as np
import logging
import pickle
from argparse import Namespace
import matplotlib.pyplot as plt
import seaborn as sns
import json
from IPython.display import display

################################################################################################
################################ Model Calculation Metrics #####################################
################################################################################################


def custom_describe_dataframe(df: pd.DataFrame):
    """
    Custom describe function to include additional statistics.
    """
    desc = df.describe().T
    desc['median'] = df.median()
    desc['skew'] = df.skew()
    desc['kurtosis'] = df.kurtosis()
    desc['missing'] = df.isnull().sum()
    desc['missing_pct'] = df.isnull().mean() * 100
    desc = desc[['count', 'missing', 'missing_pct', 'mean', 'std', 'min', '25%', 'median', '75%', 'max', 'skew', 'kurtosis']]
    return desc.T


def regression_calculation(y_pred: pd.Series, y_actual: pd.Series):
    """
    Calculates regression performance metrics, including bias.
    """
    residuals = y_pred - y_actual
    y_var = np.var(y_actual)
    y_std = np.std(y_actual)
    
    if len(y_actual) > 1:
        y_true_std = np.std(y_actual)
        y_pred_std = np.std(y_pred)
        if y_true_std > 1e-8 and y_pred_std > 1e-8:
            corr = np.corrcoef(y_actual, y_pred)[0, 1]
        else:
            corr = np.nan
    else:
        corr = np.nan

    residuals = y_pred - y_actual
    results = {
        "Mean Squared Error": mean_squared_error(y_actual, y_pred),
        "Root Mean Squared Error": root_mean_squared_error(y_actual, y_pred),
        "Mean Absolute Error": mean_absolute_error(y_actual, y_pred),
        "Median Absolute Error": median_absolute_error(y_actual, y_pred),
        "R² Score": r2_score(y_actual, y_pred),
        "Explained Variance": explained_variance_score(y_actual, y_pred),
        "Bias (Mean Error)": np.mean(residuals),
        "Mean Absolute Percentage Error": np.mean(np.abs(residuals / y_actual)) * 100,
        "Symmetric Mean Absolute Percentage Error": 
            100 * np.mean(2 * np.abs(y_pred - y_actual) / (np.abs(y_actual) + np.abs(y_pred))),
        
        "Correlation": corr,
        
        # Standardized metrics (normalized by variance/std of target)
        "Standardized MSE": mean_squared_error(y_actual, y_pred) / y_var if y_var != 0 else np.nan,
        "Standardized RMSE": root_mean_squared_error(y_actual, y_pred) / y_std if y_std != 0 else np.nan,
        "Standardized MAE": mean_absolute_error(y_actual, y_pred) / y_std if y_std != 0 else np.nan,
        "Standardized Bias": np.mean(residuals) / y_std if y_std != 0 else np.nan,
    }

    return results


################################################################################################
######################################## Plots #################################################
################################################################################################

def plot_predicted_vs_actual(y_pred, y_actual, standardized = False):
    """Scatter plot of predicted vs actual values."""
    
    plt.figure(figsize=(16, 10))
    sns.scatterplot(x=y_actual, y=y_pred, alpha=0.6)
    plt.plot([y_actual.min(), y_actual.max()],
             [y_actual.min(), y_actual.max()],
             color="red", linestyle="--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs Actual")
    plt.show()


def plot_residuals(y_pred, y_actual, standardized = False):
    
    """Residuals vs Actual values."""
    residuals = y_actual - y_pred
    plt.figure(figsize=(16, 10))
    sns.scatterplot(x=y_actual, y=residuals, alpha=0.6)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Actual")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Actual")
    plt.show()


def plot_error_distribution(y_pred, y_actual, standardized = False):
    """Distribution of errors (residuals)."""
    residuals = y_actual - y_pred
    # logging.info(residuals.shape)
    # logging.info(residuals.min(), residuals.max(), residuals.std())

    plt.figure(figsize=(12, 8))
    sns.histplot(residuals, bins=100, kde=True, color="blue", alpha=0.6)
    plt.axvline(0, color="red", linestyle="--")
    plt.xlabel("Residuals")
    plt.title("Error Distribution")
    plt.show()

def plot_ticker_timeseries(ticker, df, target = 'next_open'):
    """Helper to plot actual vs predicted time series for a single ticker."""
    plt.figure(figsize=(16, 10))
    plt.plot(df["Date"], df[target], label="Actual", color="blue")
    plt.plot(df["Date"], df[target + "_predicted"], label="Predicted", color="orange")
    plt.title(f"Actual vs Predicted for {ticker}")
    plt.xlabel("Date")
    plt.ylabel(target)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot errors (residuals)
    residuals = df[target] - df[target + "_predicted"]
    plt.figure(figsize=(16, 10))
    plt.plot(df["Date"], residuals, color="purple", alpha=0.7)
    plt.axhline(0, color="red", linestyle="--")
    plt.title(f"Residuals over Time for {ticker}")
    plt.xlabel("Date")
    plt.ylabel("Residual")
    plt.tight_layout()
    plt.show()
    
    
 # Function to create subplot grid for tickers
def plot_ticker_grid(result_df, target, ticker_mae, tickers, title_prefix, standardized = False):
    fig, axes = plt.subplots(2, len(tickers), figsize=(24, 12), sharex=False)

    # Ensure axes is always 2D
    if len(tickers) == 1:
        axes = axes.reshape(2, 1)

    for idx, ticker in enumerate(tickers):
        df = result_df[result_df["Ticker"] == ticker].sort_values("Date")

        # Top row
        ax1 = axes[0, idx]
        ax1.plot(df["Date"], df[target], label="Actual", color="blue")
        ax1.plot(df["Date"], df[target + "_predicted"], label="Predicted", color="orange")
        ax1.set_title(f"{ticker} (MAE={ticker_mae[ticker]:.4f})")
        ax1.tick_params(axis="x", rotation=45)
        if idx == 0:
            ax1.set_ylabel(target)
        if idx == len(tickers) - 1:
            ax1.legend()

        # Bottom row
        ax2 = axes[1, idx]
        residuals = df[target] - df[target + "_predicted"]
        if standardized:
            residuals = (residuals - residuals.mean()) / residuals.std()
            #logging.info(f"Standardized values for plotting Residuals over Time for {ticker}")
        ax2.plot(df["Date"], residuals, color="purple", alpha=0.7)
        ax2.axhline(0, color="red", linestyle="--")
        ax2.set_xlabel("Date")
        if idx == 0:
            ax2.set_ylabel("Residuals")



################################################################################################
################################ S4 Main Functions #############################################
################################################################################################

def flatten_all_sequences_to_dataframe(args: Namespace, all_sequences_inf: list):
    records = []
    for seq_dict in all_sequences_inf:
        pred_keys = seq_dict['prediction_sequence_keys']  # [batch, pred_len, 2]
        y_true = seq_dict['prediction_sequence']          # [batch, pred_len, num_features]
        y_pred = seq_dict['model_prediction_sequence']    # [batch, pred_len]
        
        y_pred = y_pred.squeeze(1)
        batch_size, pred_len = y_pred.shape
        
        for b in range(batch_size):
            for t in range(pred_len):
                date, ticker = pred_keys[b][t]
                true_val = y_true[b][t][0]
                
                if hasattr(true_val, "item"):
                    true_val = true_val.item()
                    
                pred_val = y_pred[b][t]
                if hasattr(pred_val, "item"):
                    pred_val = pred_val.item()
                    
                records.append({
                    "Date": date,
                    "Ticker": ticker,
                    args.regression.S2.target_label: true_val,
                    args.regression.S2.target_label + "_predicted": pred_val
                })
    
    df_results = pd.DataFrame(records)
    
    #df_results_no_duplicates = df_results.drop_duplicates(subset=['Date', 'Ticker'], keep='first').reset_index(drop=True)
    # --- Aggregate duplicates by mean instead of dropping ---
    df_results_no_duplicates = (
        df_results
        .groupby(['Date', 'Ticker'], as_index=False)
        .mean(numeric_only=True)
    )
    logging.info(f"Created validation DataFrame with {len(df_results)} rows")
    logging.info(f"Created validation DataFrame (no dupe) with {len(df_results_no_duplicates)} rows")
    
    # We then make a standardized version of this dataframe of no dupes but we standardize by ticker
    df_results_no_duplicates_standardized = df_results_no_duplicates.copy()
    df_results_no_duplicates_standardized[args.regression.S2.target_label + "_predicted"] = (
        df_results_no_duplicates_standardized
        .groupby('Ticker')[args.regression.S2.target_label + "_predicted"]
        .transform(lambda x: (x - x.mean()) / x.std() if x.std() != 0 else x - x.mean())
    )
    df_results_no_duplicates_standardized[args.regression.S2.target_label] = (
        df_results_no_duplicates_standardized
        .groupby('Ticker')[args.regression.S2.target_label]
        .transform(lambda x: (x - x.mean()) / x.std() if x.std() != 0 else x - x.mean())
    )
    logging.info(f"Created validation DataFrame (no dupe, standardized) with {len(df_results_no_duplicates_standardized)} rows")    
    return df_results, df_results_no_duplicates, df_results_no_duplicates_standardized


def S4_Model_Validation(args: Namespace, all_sequences_inf: list = None):
    '''
    Main entry function for S4 Model Validation 
    all_sequences_inf is a list of 
    {
        "subsequence": subsequences,
        "prediction_sequence": prediction_sequences,    [batch_size, prediction_length, num_sequences] y_label
        "subsequence_keys": subsequence_keys,
        "prediction_sequence_keys": prediction_sequence_keys  [batch_size, prediction_length, 2] but in list
        "model_prediction_sequence"                     [batch_size, prediction_length] y_pred
    }
    
    '''
    logging.info(f"Initializing S4 Model Validation Results..................")
    if all_sequences_inf is None:
        logging.info(f'Loading saved sequences from {args.regression.S3.output_seq_path}')
        with open(args.regression.S3.output_seq_path, "rb") as f:
            all_sequences_inf = pickle.load(f)
        logging.info(f'Loaded saved sequences from {args.regression.S3.output_seq_path}')
        if len(all_sequences_inf) == 0:
            logging.warning(f"No rows after loading!")
            return None
        
    elif len(all_sequences_inf) == 0:
        logging.warning(f"No rows!")
        return None
    
    # Preprocess to DataFrame essentially just putting everything back together, use the prediction_sequence_keys as columns [Date, Ticker]
    result_df, result_df_no_dupe, results_df_standardized_no_dupe = flatten_all_sequences_to_dataframe(args, all_sequences_inf)
    
    #regression_res_df = S4_regression_calculation_df_level(args, result_df_no_dupe)
    regression_res_df = S4_regression_calculation_df_level(args, results_df_standardized_no_dupe)
    regression_res_full_df_seq, regression_res_full_tickers_df_seq = S4_regression_calculation_seq_level(args, all_sequences_inf)
        
    return regression_res_df, regression_res_full_df_seq, regression_res_full_tickers_df_seq, result_df, result_df_no_dupe, results_df_standardized_no_dupe


def S4_regression_calculation_df_level(args: Namespace, result_df: pd.DataFrame):
    '''
    Regression Calculation at the DataFrame Level
    '''
    logging.info(f"Calculating Regression Metrics at the aggregate dataframe Level..................")
    # Overall Regression Metrics
    regression_res = {}
    regression_res['Overall'] = regression_calculation(result_df[args.regression.S2.target_label + "_predicted"], result_df[args.regression.S2.target_label])
    for classes in args.regression.S4.subset_classes:
        regression_res[classes] = {}
        for class_item in result_df[classes].unique():
            unique_class_df = result_df[result_df[classes] == class_item]
            regression_res[classes][class_item] =  regression_calculation(unique_class_df[args.regression.S2.target_label + "_predicted"], 
                                                                            unique_class_df[args.regression.S2.target_label])
            
    # Specific Calculation by Year
    if "Date" in result_df.columns:
        result_df["Year"] = pd.to_datetime(result_df["Date"]).dt.year
        regression_res["Year"] = {}
        for year in sorted(result_df["Year"].unique()):
            year_df = result_df[result_df["Year"] == year]
            regression_res["Year"][year] = regression_calculation(
                year_df[args.regression.S2.target_label + "_predicted"],
                year_df[args.regression.S2.target_label]
            )
            
    # Plotting Functions
    if args.regression.S4.plot_graphs:
        S4_plot_graphs(args, result_df, regression_res)
    
    return regression_res


def S4_regression_calculation_seq_level(args: Namespace, all_sequences_inf: list, show_all: bool = False, 
                                        selected_metrics_boxplot: list = ['Mean Absolute Percentage Error', 'Mean Absolute Error']):
    '''
    Regression Calculation at the Sequence Level, Regression Metrics by each sequence
    '''
    logging.info(f"Calculating Regression Metrics at the sequence Level..................")
    # Overall Regression Metrics
    regression_res_full = []
    regression_res_full_tickers = {}
    for seq_dict in all_sequences_inf:
        pred_keys = seq_dict['prediction_sequence_keys'] # [batch, pred_len, 2]
        y_true = seq_dict['prediction_sequence']          # [batch, pred_len, num_features]
        y_pred = seq_dict['model_prediction_sequence']
        
        # Calculate by overall first
        y_pred = y_pred.squeeze(1)
        batch_size, pred_len = y_pred.shape
        for b in range(batch_size):
            y_pred_indiv = pd.Series(y_pred[b].detach().cpu().numpy())
            y_true_indiv = pd.Series(y_true[b][:, 0].detach().cpu().numpy())
            #y_pred_indiv, y_true_indiv = pd.Series(y_pred[b]), pd.Series(y_true[b][:, 0])
            res = regression_calculation(y_pred_indiv, y_true_indiv)
            regression_res_full.append(res)
            
            # Track which ticker isit and then add to the ticker metrics
            ticker = pred_keys[b][0][1]
            if ticker not in regression_res_full_tickers: 
                regression_res_full_tickers[ticker] = []
            regression_res_full_tickers[ticker].append(res)
    
    # We then Aggregate across all the individual sequence level results and then get a distribution
    regression_res_full_df = pd.DataFrame(regression_res_full)
    regression_res_full_tickers_df = {ticker: pd.DataFrame(df) for ticker, df in regression_res_full_tickers.items()}
    
    logging.info(f"Displaying Sequence Level Metrics for All")    
    display(custom_describe_dataframe(regression_res_full_df))
    
    # We plot Boxplots for selected metrics
    plt.figure(figsize=(len(selected_metrics_boxplot) * 3.5, 6))
    regression_res_full_df[selected_metrics_boxplot].boxplot()
    plt.title("Distribution of Regression Metrics Across All Sequences")
    plt.ylabel("Metric Value")
    plt.xticks(rotation=20)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()
    
    
    
    if show_all:
        for ticker, df in regression_res_full_tickers_df.items():
            logging.info(f"Displaying Sequence Level Metrics for {ticker}")
            #display(df.describe())
            display(custom_describe_dataframe(df))
    
    return regression_res_full_df, regression_res_full_tickers_df
        
        
def S4_plot_graphs(args: Namespace, result_df: pd.DataFrame, regression_res: dict):
    """
    Main entry for S4 plotting functions.
    Includes global plots and per-ticker performance visualization in subplots.
    """
    target = args.regression.S2.target_label
    y_actual = result_df[target]
    y_pred = result_df[target + "_predicted"]
    assert len(y_actual) == len(y_pred)
    logging.info(f"Plotting global residuals and error distribution... for {len(y_actual)} sets of points")

    # === Global plots ===
    plot_error_distribution(y_pred, y_actual, standardized=True)
    plot_residuals(y_pred, y_actual, standardized=True)
    plot_predicted_vs_actual(y_pred, y_actual, standardized=True)

    # === Per-ticker analysis ===
    if "Ticker" not in result_df.columns:
        logging.warning("No Ticker column found, skipping per-ticker plots.")
        return

    # Extract MAEs directly from regression_res (instead of recomputing)
    ticker_mae = {
        ticker: metrics["Mean Absolute Error"]
        for ticker, metrics in regression_res.get("Ticker", {}).items()
    }
    ticker_mae = pd.Series(ticker_mae).sort_values()

    top_5 = ticker_mae.head(5).index
    bottom_5 = ticker_mae.tail(5).index

    logging.info(f"Top 5 tickers (lowest MAE): {list(top_5)}")
    logging.info(f"Bottom 5 tickers (highest MAE): {list(bottom_5)}")

    # Plot top 5 tickers (2x5 grid)
    plot_ticker_grid(result_df=result_df, target=args.regression.S2.target_label, 
                     ticker_mae=ticker_mae, tickers=top_5, title_prefix="Top 5 (Lowest MAE)", standardized=True)

    # Plot bottom 5 tickers (2x5 grid)
    plot_ticker_grid(result_df=result_df, target=args.regression.S2.target_label, 
                     ticker_mae=ticker_mae, tickers=bottom_5, title_prefix="Bottom 5 (Highest MAE)", standardized=True)
    
    
    
################################################################################################################
################################### Evaluating Metrics Distributions Across Tickers ############################
################################################################################################################


def S4_evaluate_metrics_distribution_across_tickers(
    args,
    regression_res_full_tickers_df: dict,
    metrics_to_evaluate: list = ['Mean Absolute Percentage Error', 'Bias (Mean Error)', 'Correlation'],
    labels_to_evaluate: list = ['mean', 'mean', 'mean']
):
    """
    Evaluate and summarize the distribution of regression metrics across tickers.
    Combines all metric summaries into a single table and visualizes each metric.
    """
    logging.info("Evaluating Metrics Distribution Across Tickers...")

    summary_dict = {}

    # --- Compute summary stats for each metric ---
    for metric, label in zip(metrics_to_evaluate, labels_to_evaluate):
        ticker_values = {}

        for ticker, df in regression_res_full_tickers_df.items():
            if metric in df.columns:
                ticker_values[ticker] = (
                    df[metric].loc[label].values if label in df.index else df[metric].values
                )

        tickers = list(ticker_values.keys())
        metric_values = [
            val[0] if len(val) > 0 else np.nan for val in ticker_values.values()
        ]
        metric_series = pd.Series(metric_values, index=tickers)

        summary_dict[metric] = {
            "mean": metric_series.mean(),
            "median": metric_series.median(),
            "std": metric_series.std(),
            "min": metric_series.min(),
            "max": metric_series.max(),
            "25th_percentile": metric_series.quantile(0.25),
            "75th_percentile": metric_series.quantile(0.75),
            "IQR": metric_series.quantile(0.75) - metric_series.quantile(0.25),
        }

    # --- Combine summaries into one table ---
    summary_df = pd.DataFrame(summary_dict)
    logging.info("Combined metrics summary table:")
    display(summary_df)

    # --- Plot distributions for each metric ---
    for metric, label in zip(metrics_to_evaluate, labels_to_evaluate):
        plot_all_regression_metrics_by_ticker(args, regression_res_full_tickers_df, metric, label)

    return summary_df


def plot_distribution_by_metric_tickers(args: Namespace, regression_res_full_tickers_df: dict, metric: str, label: str = 'mean'):
    """
    Plot the distribution of a specific regression metric across different tickers.
    """
    logging.info(f"Plotting Distribution of {metric} Across Tickers..................")
    ticker_values = {}
    for ticker, df in regression_res_full_tickers_df.items():
        if metric in df.columns:
            ticker_values[ticker] = df[metric].loc[label].values if label in df.index else df[metric].values
            
    tickers = list(ticker_values.keys())
    metric_values = [val[0] if len(val) > 0 else np.nan for val in ticker_values.values()]
    
            
    plt.figure(figsize=(16, 10))
    sns.barplot(x=tickers, y=metric_values)
    plt.xticks(rotation=90)
    plt.xlabel("Ticker")
    plt.ylabel(metric)
    plt.title(f"Distribution of {metric} Across Tickers")
    plt.tight_layout()
    plt.show()


def plot_sorted_bar_plot_by_metric_tickers(args: Namespace, regression_res_full_tickers_df: dict, metric: str, label: str = 'mean'):
    """
    Plot a sorted bar plot of a specific regression metric across different tickers.
    """
    logging.info(f"Plotting Sorted Bar Plot of {metric} Across Tickers..................")
    ticker_values = {}
    
    for ticker, df in regression_res_full_tickers_df.items():
        if metric in df.columns:
            ticker_values[ticker] = df[metric].loc[label].values if label in df.index else df[metric].values
            
    sorted_tickers = sorted(ticker_values.items(), key=lambda x: x[1][0] if len(x[1]) > 0 else np.nan)
    tickers = [item[0] for item in sorted_tickers]
    metric_values = [item[1][0] if len(item[1]) >
                        0 else np.nan for item in sorted_tickers]
    
    plt.figure(figsize=(16, 10))
    sns.barplot(x=tickers, y=metric_values)
    plt.xticks(rotation=90)
    plt.xlabel("Ticker")
    plt.ylabel(metric)
    plt.title(f"Sorted Distribution of {metric} Across Tickers")
    plt.tight_layout()
    plt.show()
    

def plot_box_plot_by_metric_tickers(args: Namespace, regression_res_full_tickers_df: dict, metric: str):
    """
    Plot a box plot of a specific regression metric across different tickers.
    """
    logging.info(f"Plotting Box Plot of {metric} Across Tickers..................")
    ticker_values = {}
    
    for ticker, df in regression_res_full_tickers_df.items():
        if metric in df.columns:
            ticker_values[ticker] = df[metric].values
            
    tickers = list(ticker_values.keys())
    metric_values = [val for val in ticker_values.values()]
            
    plt.figure(figsize=(16, 10))
    sns.boxplot(data=metric_values)
    plt.xticks(ticks=range(len(tickers)), labels=tickers, rotation=90)
    plt.xlabel("Ticker")
    plt.ylabel(metric)
    plt.title(f"Box Plot of {metric} Across Tickers")
    plt.tight_layout()
    plt.show()
    
    

def plot_all_regression_metrics_by_ticker(args: Namespace, regression_res_full_tickers_df: dict, metric: str, label: str = 'mean'):
    """
    Plots three subplots (3x1):
        1. Histogram of metric values across tickers
        2. Sorted distribution of metric across tickers (sorted bar)
        3. Box plot of metric distribution across tickers
    Ignores NaN values in all plots.
    """
    logging.info(f"Plotting combined regression metric plots for {metric} across tickers...")

    # --- Collect values for each ticker ---
    ticker_values = {}
    for ticker, df in regression_res_full_tickers_df.items():
        if metric in df.columns:
            if label in df.index:
                vals = df.loc[label, metric]
            else:
                vals = df[metric]

            # Convert to numpy array, drop NaNs
            vals = np.array(vals, dtype=float)
            vals = vals[~np.isnan(vals)]
            if len(vals) > 0:
                ticker_values[ticker] = vals

    if not ticker_values:
        logging.warning(f"No valid (non-NaN) values found for metric '{metric}'. Skipping plot.")
        return

    tickers = list(ticker_values.keys())
    metric_values = np.array([np.mean(val) for val in ticker_values.values()])

    # --- Sort tickers for second subplot ---
    sorted_items = sorted(
        ticker_values.items(),
        key=lambda x: np.mean(x[1]) if len(x[1]) > 0 else np.nan
    )
    sorted_tickers = [item[0] for item in sorted_items]
    sorted_values = [np.mean(item[1]) for item in sorted_items]

    # --- Boxplot data ---
    boxplot_data = [vals for vals in ticker_values.values()]

    # --- Create subplots ---
    fig, axes = plt.subplots(3, 1, figsize=(16, 18))

    # (1) Histogram
    sns.histplot(metric_values[~np.isnan(metric_values)], bins=50, kde=True, ax=axes[0])
    axes[0].set_title(f"Histogram of {metric} Across Tickers", fontsize=14)
    axes[0].set_xlabel(metric)
    axes[0].set_ylabel("Frequency")

    # (2) Sorted Bar Plot
    sns.barplot(x=sorted_tickers, y=sorted_values, ax=axes[1])
    axes[1].set_title(f"Sorted Distribution of {metric} Across Tickers", fontsize=14)
    axes[1].set_xlabel("Ticker")
    axes[1].set_ylabel(metric)
    axes[1].tick_params(axis='x', rotation=90)

    # (3) Box Plot
    sns.boxplot(data=boxplot_data, ax=axes[2])
    axes[2].set_title(f"Box Plot of {metric} Across Tickers", fontsize=14)
    axes[2].set_xlabel("Ticker")
    axes[2].set_ylabel(metric)
    axes[2].set_xticks(range(len(tickers)))
    axes[2].set_xticklabels(tickers, rotation=90)

    plt.tight_layout()
    plt.show()



########################################################################################################################
################################### Evaluating Metrics Distributions Across Various Buckets ############################
########################################################################################################################

def compute_ticker_metric(df: pd.DataFrame, ticker_col: str, value_col: str, metric_func) -> pd.DataFrame:
    """
    Compute a metric (e.g., volatility) per ticker.

    Parameters:
    - df: DataFrame with tickers and values
    - ticker_col: column containing ticker names
    - value_col: column containing the values to compute metric on (e.g., price)
    - metric_func: function to compute the metric on a series (e.g., np.std for volatility)

    Returns:
    - DataFrame with columns ['Ticker', 'metric']
    """
    metric_df = df.groupby(ticker_col)[value_col].apply(metric_func).reset_index()
    metric_df.rename(columns={value_col: 'metric'}, inplace=True)
    return metric_df



def volatility_bucket_function(args, df: pd.DataFrame):
    volatility_df = compute_ticker_metric(df, ticker_col='Ticker', value_col=args.regression.S2.target_label, metric_func=np.std)
    #volatility_df.rename(columns={'metric': 'volatility'}, inplace=True)
    return volatility_df    


def S4_evaluate_metrics_distribution_across_buckets(args: Namespace, 
                                                    labels_df: pd.DataFrame, # DataFrame with at least ['Date', 'Ticker', target_label]
                                                    regression_res_full_tickers_df: dict, 
                                                    bucket_column_func = volatility_bucket_function, 
                                                    bucket_column: str = 'Bucket',
                                                    metrics_to_evaluate: list = ['Mean Absolute Percentage Error', 'Bias (Mean Error)', 'Correlation'], 
                                                    labels_to_evaluate: list = ['mean', 'mean', 'mean'],
                                                    save_buckets: bool = True,
                                                    save_buckets_path: str = None,
                                                    n_buckets: int = 5):
    """
    Evaluate and summarize the distribution of regression metrics across different buckets.
    Combines all metric summaries into a single table and visualizes each metric.
    """
    logging.info(f"Evaluating Metrics Distribution Across Buckets in column '{bucket_column}'...")
    

   # First get the groundtruth dataframe values and then apply the bucket function to get the bucket column
        
    # Apply the bucketting function such that we will bucket the tickers into different buckets by the specificed function metric
    ticker_metric_df = bucket_column_func(args, labels_df)
    ticker_metric_df[bucket_column] = pd.qcut(ticker_metric_df['metric'], q=n_buckets, labels=False)
    ticker_metric_df = ticker_metric_df.drop_duplicates(subset=['Ticker'], keep='first').reset_index(drop=True)
    ticker_to_bucket = dict(zip(ticker_metric_df['Ticker'], ticker_metric_df[bucket_column]))
    
    # Write to a json file
    if save_buckets:
        # Get out the Ticker to Bucket mapping
        logging.info(f"Ticker to {bucket_column}")
        with open(f"{save_buckets_path}", "w") as f:
            json.dump(ticker_to_bucket, f, indent=4)
        logging.info(f"Saved Ticker to {bucket_column} mapping to {save_buckets_path}")
    
    
    summary_dict = {}
    # --- Compute summary stats for each metric ---
    for ticker, ticker_res in regression_res_full_tickers_df.items():
        ticker_res_summary_stats = custom_describe_dataframe(ticker_res)
        
        for metric, label in zip(metrics_to_evaluate, labels_to_evaluate):
            
            if metric not in summary_dict:
                summary_dict[metric] = {}
            
            # Extract out the metric
            metric_value = ticker_res_summary_stats.loc[label, metric]
            
            # Insert into the bucket 
            bucket = ticker_to_bucket.get(ticker)
            if bucket not in summary_dict[metric]:
                summary_dict[metric][bucket] = []
            summary_dict[metric][bucket].append(metric_value)

    # Within each metric and bucket get calculate out the statistical distributions
    for metric, buckets in summary_dict.items():
        for bucket, bucket_vals in buckets.items():
            buckets[bucket] = pd.Series(bucket_vals).describe()
    

    # Convert to hierarchical DataFrame
    summary_df = pd.concat(
        {metric: pd.DataFrame.from_dict(buckets, orient='index') for metric, buckets in summary_dict.items()},
        names=['Metric', 'Bucket']
    )
    flattened_df = flatten_to_display(summary_df)
    return flattened_df


def flatten_to_display(summary_df: pd.DataFrame):
    # 1) Flatten (if it's multi-indexed)
    df_reset = summary_df.reset_index()

    # 2) Melt the stats columns into long form
    melted = df_reset.melt(
        id_vars=["Metric", "Bucket"],
        var_name="Stat",
        value_name="Value"
    )

    # 3) Pivot so index is (Metric, Stat) and columns are Bucket
    final_df = melted.pivot_table(
        index=["Metric", "Stat"],
        columns="Bucket",
        values="Value",
        aggfunc="first"   # there shouldn't be duplicates; first keeps single value
    )

    # 4) (Optional) ensure the statistics appear in a specific order
    desired_stat_order = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
    # keep only stats that actually exist
    existing_stats = [s for s in desired_stat_order if s in final_df.index.get_level_values(1)]
    # reorder the second-level index (Metric, Stat) by Stat order
    final_df = final_df.reindex(
        pd.MultiIndex.from_product(
            [final_df.index.get_level_values(0).unique(), existing_stats],
            names=["Metric", "Stat"]
        )
    )

    # 5) (Optional) order bucket columns numerically if bucket labels are numeric-strings
    try:
        # Convert column labels to ints for sorting if possible, otherwise leave as-is
        sorted_cols = sorted(final_df.columns, key=lambda x: int(x) if str(x).isdigit() else x)
        final_df = final_df[sorted_cols]
    except Exception:
        pass

    # Display
    display(final_df)
    return final_df
    

def plot_metric_across_buckets(args: Namespace, final_df: pd.DataFrame, metric: str, stat: str = "mean"):
    """
    Plot a specific regression metric and statistic (e.g., mean, std, 50%) across buckets.

    Parameters:
        args: Namespace — argument container
        final_df: pd.DataFrame — pivoted DataFrame with MultiIndex (Metric, Stat)
        metric: str — name of the metric to plot (e.g., "Mean Absolute Percentage Error")
        stat: str — which statistic to plot (default "mean")
    """
    # Ensure metric and stat exist
    if metric not in final_df.index.get_level_values('Metric'):
        logging.warning(f"Metric '{metric}' not found in DataFrame.")
        return
    if stat not in final_df.loc[metric].index:
        logging.warning(f"Stat '{stat}' not found for metric '{metric}'. Available: {final_df.loc[metric].index.tolist()}")
        return

    # Extract the row corresponding to (metric, stat)
    bucket_series = final_df.loc[(metric, stat)]

    # Convert to DataFrame for seaborn
    bucket_df = bucket_series.reset_index()
    bucket_df.columns = ['Bucket', 'Value']

    # Ensure buckets are sorted numerically if possible
    try:
        bucket_df['Bucket'] = bucket_df['Bucket'].astype(int)
        bucket_df = bucket_df.sort_values('Bucket')
    except ValueError:
        pass

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Bucket', y='Value', data=bucket_df, palette='viridis')
    plt.xlabel('Bucket', fontsize=12)
    plt.ylabel(f"{metric} ({stat})", fontsize=12)
    plt.title(f"{metric} ({stat}) Across Buckets", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
