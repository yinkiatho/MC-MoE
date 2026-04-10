import pandas as pd
import numpy as np
import json
import datetime
from argparse import Namespace
import logging

def safe_div(n, d):
    return np.where(d == 0, np.nan, n / d)


def S1_preprocessing(args: Namespace) -> pd.DataFrame: 
    '''
    Entry Function to S1 Preprocessing, loads the data and then outputs a pandas dataframe
    '''
    logging.info(f"Initiating S1 Data Preprocessing....................")
    
    # Read in the tickers
    tickers = []
    with open(args.regression.S1.ticker_list, 'r') as file:
        for line in file:
            tickers.append(line.strip())
    
    logging.info(f"Read in total of {len(tickers)} tickers...")
    
    
    # Read in other file sources to concat into multi-variate data
    price_df = pd.read_parquet(args.regression.S1.price_data)
    
    
    # Preprocessing of the alternative data
    full_df = data_preprocessing(args, price_df)
    full_df = full_df[(full_df['Ticker'].isin(tickers)) & ~(full_df['Ticker'].isin(args.regression.S1.tickers_to_exclude)) & 
                        (full_df['Date'] >= args.regression.S1.start_date) &
                        (full_df['Date'] <= args.regression.S1.end_date)].drop_duplicates().reset_index(drop=True)
    if args.regression.S1.save_file:
        full_df.to_parquet(args.regression.S1.processed_data_file_path)
        logging.info(f"Saved file to {args.regression.S1.processed_data_file_path}")
    
    return full_df
    

def data_preprocessing(args: Namespace, price_df: pd.DataFrame) -> pd.DataFrame:
    '''
    Data preprocessing of price_df and other features, returns the preprocessed dataframe
    '''    
    # First read in all the alternative data
    processed_alt_data = {}
    financial_statements = {}
    logging.info(f"Preprocessing all alternative data............")
    for k, path in args.regression.S1.alternative_data.__dict__.items():
        df = pd.read_parquet(path)
        if k == 'congress_trading_disclosures':
            processed_alt_data[k] = preprocess_congress_trading_disclosures(args, df)
        if k == 'earnings_quality_scores':
            processed_alt_data[k] = preprocess_earnings_quality_scores(args, df)
        if k == 'earnings_surprises':
            processed_alt_data[k] = preprocess_earnings_surprises(args, df)
        if k in ['bs_quarterly', 'ic_quarterly', 'cf_quarterly']:
            financial_statements[k] = df
            if len(financial_statements) == 3:
                processed_alt_data['financial_statements'] = preprocess_financial_statements(args, **financial_statements)
        if k == 'esg_scores':
            processed_alt_data[k] = preprocess_esg_scores(args, df)
        if k == 'insider_sentiment': 
            processed_alt_data[k] = preprocess_insider_sentiment(args, df)
        if k == 'insider_transactions':
            processed_alt_data[k] = preprocess_insider_transactions(args, df)
        if k == 'market_cap':
            processed_alt_data[k] = preprocess_market_cap(args, df)
        if k == 'social_sentiment':
            processed_alt_data[k] = preprocess_social_sentiment(args, df)
        if k == 'usa_spending_history':
            processed_alt_data[k] = preprocess_usa_spending_history(args, df)
            
    # Next we do outer merge, by = 'Date' and 'Ticker', for quarterly data, we will merge and ffill, while the rest we fill as 0 first
    # Quarterly data are esg_scores, financial_statements, earnings_quality_scores, earnings_surprises, we will only ffill up to a quarter of a year
    
    # Step 2: Merge all datasets with price_df
    logging.info(f"Merging with OHLC............")

    merged_df = price_df.copy()
    quarterly_keys = {"esg_scores", "financial_statements", "earnings_quality_scores", "earnings_surprises"}
    merged_df["Date"] = merged_df["Date"].astype(str)
    
    for name, alt_df in processed_alt_data.items():
        alt_df['Date'] = alt_df['Date'].astype(str)
        merged_df = merged_df.merge(
            alt_df, on=["Date", "Ticker"], how="left"
        )

        # # Step 3: Fill missing values
        # if name in quarterly_keys:
        #     # Forward-fill, but only within a quarter (~90 days)
        #     merged_df = merged_df.sort_values(["Ticker", "Date"])
        #     merged_df[alt_df.columns.difference(["Date", "Ticker"])] = (
        #         merged_df.groupby("Ticker")[alt_df.columns.difference(["Date", "Ticker"])]
        #         .ffill(limit=90)   # forward fill up to 90 days
        #     )
        # else:
        #     # Daily/transactional data → fillna(0)
        #     merged_df[alt_df.columns.difference(["Date", "Ticker"])] = (
        #         merged_df[alt_df.columns.difference(["Date", "Ticker"])].fillna(0)
        #     )
    logging.info(f'Merged Complete! Total NaN rows: {merged_df.isna().any(axis=1).sum()}, Total Rows: {len(merged_df)}')
    return merged_df.reset_index(drop=True)
            
    
    
    
##########################################################################################################
###################### Alternative Data Preprocessing Functions ##########################################
##########################################################################################################
def preprocess_usa_spending_history(args: Namespace, usa_spending_history: pd.DataFrame):
    
    # Keep relevant columns and filter positive values
    usa_spending_history_agg = usa_spending_history[['actionDate', 'symbol', 'totalValue']]
    usa_spending_history_agg = usa_spending_history_agg[usa_spending_history_agg['totalValue'] > 0]
    
    usa_spending_history_agg = usa_spending_history_agg.rename(columns={'symbol': 'Ticker', 'actionDate': 'Date'})
    usa_spending_history_agg = usa_spending_history_agg.groupby(['Date', 'Ticker'], as_index=False).sum()
    usa_spending_history_agg = usa_spending_history_agg.sort_values(by=['Date', 'Ticker']).reset_index(drop=True)
    
    return usa_spending_history_agg

def preprocess_social_sentiment(args: Namespace, social_sentiment: pd.DataFrame):
    # Convert timestamp to date
    social_sentiment['Date'] = pd.to_datetime(social_sentiment['atTime']).dt.date
    social_sentiment_agg = social_sentiment[['Date', 'symbol', 'score']].rename(
        columns={'symbol': 'Ticker', 'score': 'social_sentiment_score'}
    )
    # Aggregate by Date and Ticker using SUM
    social_sentiment_agg = social_sentiment_agg.groupby(['Date', 'Ticker'], as_index=False).sum()
    social_sentiment_agg = social_sentiment_agg.sort_values(by=['Date', 'Ticker']).reset_index(drop=True)
    
    return social_sentiment_agg


def preprocess_market_cap(args: Namespace, market_cap: pd.DataFrame):
    market_cap_agg = market_cap.rename(columns={'symbol': 'Ticker', 'atDate': 'Date'}).sort_values(by=['Date', 'Ticker']).reset_index(drop=True)
    return market_cap_agg


def preprocess_insider_transactions(args: Namespace, insider_transactions: pd.DataFrame):
    # Keep relevant columns
    insider_transactions_agg = insider_transactions[['filingDate', 'symbol', 'change', 'share', 'transactionPrice']].copy()
    
    # Calculate before_share and change_vol
    insider_transactions_agg['before_share'] = insider_transactions_agg['share'] + insider_transactions_agg['change']
    insider_transactions_agg['change_vol'] = insider_transactions_agg['change'] * insider_transactions_agg['transactionPrice']
    
    # Rename columns
    insider_transactions_agg = insider_transactions_agg.rename(columns={'symbol': 'Ticker', 'filingDate': 'Date'})
    
    # Aggregate by Date and Ticker
    insider_transactions_agg = insider_transactions_agg.groupby(['Date', 'Ticker'], as_index=False).agg({
        'before_share': 'sum',
        'change': 'sum',
        'share': 'sum',
        'change_vol': 'sum',
        'transactionPrice': 'mean'
    }).drop_duplicates(subset=['Date', 'Ticker'])
    
    # Sort and reset index
    insider_transactions_agg = insider_transactions_agg.sort_values(by=['Date', 'Ticker']).reset_index(drop=True)
    assert len(insider_transactions_agg[insider_transactions_agg.duplicated(subset=['Date', 'Ticker'])]) == 0
    return insider_transactions_agg

def preprocess_insider_sentiment(args: Namespace, insider_sentiment: pd.DataFrame):
    
    # Create Date column as first day of month
    insider_sentiment["Date"] = pd.to_datetime(dict(year=insider_sentiment["year"], month=insider_sentiment["month"], day=1))

    # Rename mspr
    insider_sentiment = insider_sentiment.rename(columns={"mspr": "monthly_share_purchase_ratio", 'change': 'net_change_insider_transactions'}).sort_values(by='Date').drop_duplicates()

    # Shift within each symbol group
    insider_sentiment["prev_monthly_share_purchase_ratio"] = (
        insider_sentiment.groupby("symbol")["monthly_share_purchase_ratio"].shift(1)
    )
    insider_sentiment_agg = insider_sentiment[['Date', 'symbol', 'net_change_insider_transactions', 'monthly_share_purchase_ratio', 'prev_monthly_share_purchase_ratio']].rename(columns={'symbol': 'Ticker'}).sort_values(by=['Date', 'Ticker']).reset_index(drop=True)
    assert len(insider_sentiment_agg[insider_sentiment_agg.duplicated(subset=['Date', 'Ticker'])]) == 0
    return insider_sentiment_agg


def preprocess_esg_scores(args: Namespace, historical_esg: pd.DataFrame):
    # We just keep the aggregated esg scores 
    historical_esg_agg = historical_esg[['period', 'symbol', 'environmentScore', 'governanceScore','socialScore', 'totalESGScore']].copy().rename(columns={'period': 'Date', 'symbol': 'Ticker'}).sort_values(by=['Date', 'Ticker']).reset_index(drop=True)
    return historical_esg_agg

def preprocess_congress_trading_disclosures(args: Namespace, trading_disclosure: pd.DataFrame):
    # We flip all the amountFrom to amountTo if there is any violating conditions
    mask_sale = (trading_disclosure['transactionType'] == 'Sale') & \
                (trading_disclosure['amountFrom'] < trading_disclosure['amountTo'])

    mask_purchase = (trading_disclosure['transactionType'] == 'Purchase') & \
                    (trading_disclosure['amountFrom'] > trading_disclosure['amountTo'])

    # Combine both masks
    mask = mask_sale | mask_purchase

    # Swap amountFrom and amountTo for violating rows
    trading_disclosure.loc[mask, ['amountFrom', 'amountTo']] = \
        trading_disclosure.loc[mask, ['amountTo', 'amountFrom']].values
        
        
    # Aggregate function
    trading_disclosure_agg = (trading_disclosure.groupby(['transactionDate', 'symbol']).apply(lambda g: pd.Series({'totalAmountFrom': g['amountFrom'].sum(),'totalAmountTo': g['amountTo'].sum()})).reset_index()).sort_values(by=['transactionDate', 'symbol']).rename(columns={'transactionDate': 'Date', 'symbol': 'Ticker'})
    trading_disclosure_agg
    assert len(trading_disclosure_agg[trading_disclosure_agg.duplicated(subset=['Date', 'Ticker'])]) == 0
    return trading_disclosure_agg



def preprocess_earnings_quality_scores(args: Namespace, earnings_quality_score: pd.DataFrame):
    earnings_quality_score_agg = earnings_quality_score[['period', 'symbol', 'cashGenerationCapitalAllocation', 'growth', 'leverage', 'profitability', 'score']].rename(columns={'period': 'Date', 'symbol': 'Ticker'}).sort_values(by=['Date', 'Ticker']).reset_index(drop=True)
    assert len(earnings_quality_score_agg [earnings_quality_score_agg.duplicated(subset=['Date', 'Ticker'])]) == 0
    return earnings_quality_score_agg


def preprocess_earnings_surprises(args: Namespace, earnings_surprises: pd.DataFrame):
    earnings_surprises_agg = earnings_surprises[['period', 'symbol', 'surprisePercent']].rename(columns={'period': 'Date', 'symbol': 'Ticker', 'surprisePercent':'surprise_percent_quarterly'}).drop_duplicates(subset=['Date', 'Ticker']).sort_values(by=['Date', 'Ticker']).reset_index(drop=True)
    assert len(earnings_surprises_agg[earnings_surprises_agg.duplicated(subset=['Date', 'Ticker'])]) == 0
    return earnings_surprises_agg


def preprocess_employee_count(args: Namespace, employee_count: pd.DataFrame):
    employee_count_agg = employee_count.rename(columns={'atDate': 'Date', 'symbol': 'Ticker'}).sort_values(by=[ 'Ticker', 'Date']).reset_index(drop=True)
    assert len(employee_count_agg[employee_count_agg.duplicated(subset=['Date', 'Ticker'])]) == 0
    return employee_count_agg


def preprocess_financial_statements(args: Namespace, bs_quarterly: pd.DataFrame, ic_quarterly: pd.DataFrame, cf_quarterly: pd.DataFrame):
    # Merge on period + symbol
    df_agg = (
        bs_quarterly
        .merge(ic_quarterly, on=["period", "symbol"], suffixes=("_bs", "_ic"))
        .merge(cf_quarterly, on=["period", "symbol"], suffixes=("", "_cf"))
    )
    
    # Revenue growth (YoY or QoQ depending on your frequency)
    df_agg = df_agg.sort_values(by=["symbol", "period"])
    df_agg["revenue_growth"] = df_agg.groupby("symbol")["revenue"].pct_change()

    # Profitability
    df_agg["gross_profit"] = df_agg["grossIncome"]       # already gross profit
    df_agg["operating_profit"] = df_agg["ebit"]
    df_agg["net_profit"] = df_agg["netIncome"]

    # Margins
    df_agg["gross_margin"] = safe_div(df_agg["grossIncome"], df_agg["revenue"])
    df_agg["operating_margin"] = safe_div(df_agg["ebit"], df_agg["revenue"])
    df_agg["net_margin"] = safe_div(df_agg["netIncome"], df_agg["revenue"])

    # Liquidity
    df_agg["current_ratio"] = safe_div(df_agg["currentAssets"], df_agg["currentLiabilities"])
    df_agg["quick_ratio"] = safe_div(df_agg["currentAssets"] - df_agg["inventory"],
                                    df_agg["currentLiabilities"])

    # Leverage
    df_agg["debt_equity"] = safe_div(df_agg["totalDebt"], df_agg["totalEquity"])

    # Returns
    df_agg["roa"] = safe_div(df_agg["netIncome"], df_agg["totalAssets"])
    df_agg["roe"] = safe_div(df_agg["netIncome"], df_agg["totalEquity"])

    df_agg["dupont_net_margin"] = safe_div(df_agg["netIncome"], df_agg["revenue"])
    df_agg["dupont_asset_turnover"] = safe_div(df_agg["revenue"], df_agg["totalAssets"])
    df_agg["dupont_equity_multiplier"] = safe_div(df_agg["totalAssets"], df_agg["totalEquity"])

    df_agg["dupont_roe"] = (
        df_agg["dupont_net_margin"] *
        df_agg["dupont_asset_turnover"] *
        df_agg["dupont_equity_multiplier"]
    )

    ratios = [
        "revenue_growth", "gross_profit", "operating_profit", "net_profit",
        "gross_margin", "operating_margin", "net_margin",
        "current_ratio", "quick_ratio",
        "roa", "roe", "debt_equity",
        "dupont_net_margin", "dupont_asset_turnover", "dupont_equity_multiplier", "dupont_roe"
    ]

    df_final = df_agg[["period", "symbol"] + ratios]
    df_final = df_final.rename(columns={'period': 'Date', 'symbol': 'Ticker'}).sort_values(by=['Date', 'Ticker']).reset_index(drop=True)
    combined_fs_agg = df_final.copy()
    assert len(combined_fs_agg[combined_fs_agg.duplicated(subset=['Date', 'Ticker'])]) == 0
    return combined_fs_agg
     
        
    
    
    
    
