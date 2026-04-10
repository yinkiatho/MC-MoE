from math import log
import os
import asyncio
import pandas as pd
from datetime import datetime, timedelta
from tqdm.asyncio import tqdm_asyncio
from dotenv import load_dotenv
import time
import finnhub
import sys
import logging 
from concurrent.futures import ThreadPoolExecutor
import tqdm

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
os.chdir(r'C:/Users/yinki/OneDrive/NUS/BT4101/fyp-kiat/ML_Core/src/')
load_dotenv('../config/.env')
API_KEY = os.getenv("FINNHUB_API_KEY")

print(f"Finnhub API Key loaded. {API_KEY}")
BASE_URL = "https://finnhub.io/api/v1"
finnhub_client = finnhub.Client(api_key=os.getenv("FINNHUB_API_KEY"))
executor = ThreadPoolExecutor(max_workers=5)


# Generic async fetch
async def fetch_json(session, url, params):
    try:
        async with session.get(url, params=params) as resp:
            return await resp.json()
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None
    
async def async_finnhub_call(func, *args, **kwargs):
    delay = 1
    max_retries = 5
    for attempt in range(max_retries):
        try:
            # Run blocking SDK call in separate thread
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(executor, lambda: func(*args, **kwargs))
            return result
        except finnhub.FinnhubAPIException as e:
            print(f"Finnhub API error: {e}, retrying in {delay}s...")
            await asyncio.sleep(delay)
            delay *= 2
        except Exception as e:
            print(f"Unexpected error: {e}, retrying in {delay}s...")
            await asyncio.sleep(delay)
            delay *= 2
    print(f"Failed after {max_retries} attempts: {func.__name__}")
    return None

# --------------------------
# Async fetch per ticker with window
# --------------------------

async def fetch_market_cap(symbol, start_global, end_global):
    logging.info(f"Fetching historical market cap for {symbol}")
    dfs = []
    end_date = end_global
    while end_date > start_global:
        start_date = max(end_date - timedelta(days=365), start_global)
        try:
            res = await async_finnhub_call(
                finnhub_client.historical_market_cap,
                symbol,
                _from=start_date.strftime("%Y-%m-%d"),
                to=end_date.strftime("%Y-%m-%d")
            )
            if res and "data" in res and len(res["data"]) > 0:
                df = pd.DataFrame(res["data"])
                df["symbol"] = symbol
                dfs.append(df)
                end_date = start_date
            else:
                end_date = start_date
        except Exception as e:
            logging.error(f"Error fetching market cap for {symbol} ({start_date}-{end_date}): {e}")
        await asyncio.sleep(0.2)  # respect rate limits
    return dfs

async def fetch_employee_count(symbol, start_global, end_global):
    logging.info(f"Fetching historical employee count for {symbol}")
    dfs = []
    end_date = end_global
    while end_date > start_global:
        start_date = max(end_date - timedelta(days=365), start_global)
        try:
            res = await async_finnhub_call(
                finnhub_client.historical_employee_count,
                symbol,
                _from=start_date.strftime("%Y-%m-%d"),
                to=end_date.strftime("%Y-%m-%d")
            )
            if res and "data" in res and len(res["data"]) > 0:
                df = pd.DataFrame(res["data"])
                df["symbol"] = symbol
                dfs.append(df)
                end_date = start_date
            else:
                end_date = start_date
        except Exception as e:
            logging.error(f"Error fetching employee count for {symbol} ({start_date}-{end_date}): {e}")
        
        await asyncio.sleep(0.2)
    return dfs





async def fetch_insider_transactions(symbol, start_global, end_global):
    logging.info(f"Fetching insider transactions for {symbol}")
    dfs = []
    end_date = end_global
    while end_date > start_global:
        start_date = max(end_date - timedelta(days=365), start_global)
        res = await async_finnhub_call(
            finnhub_client.stock_insider_transactions,
            symbol,
            _from=start_date.strftime("%Y-%m-%d"),
            to=end_date.strftime("%Y-%m-%d")
        )
        if res and "data" in res and len(res["data"]) > 0:
            df = pd.DataFrame(res["data"])
            df["symbol"] = symbol
            dfs.append(df)
        end_date = start_date
        await asyncio.sleep(0.2)  # respect rate limits
    logging.info(f"Fetched insider transactions for {symbol}")
    return dfs

async def fetch_insider_sentiment(symbol, start_global, end_global):
    logging.info(f"Fetching insider sentiment for {symbol}")
    dfs = []
    end_date = end_global
    while end_date > start_global:
        start_date = max(end_date - timedelta(days=365), start_global)
        res = await async_finnhub_call(
            finnhub_client.stock_insider_sentiment,
            symbol,
            _from=start_date.strftime("%Y-%m-%d"),
            to=end_date.strftime("%Y-%m-%d")
        )
        if res and "data" in res and len(res["data"]) > 0:
            df = pd.DataFrame(res["data"])
            df["symbol"] = symbol
            dfs.append(df)
        end_date = start_date
        await asyncio.sleep(0.2)
    logging.info(f"Fetched insider sentiment for {symbol}")
    return dfs

async def fetch_earnings(symbol):
    logging.info(f"Fetching earnings surprises for {symbol}")
    res = await async_finnhub_call(finnhub_client.company_earnings, symbol)
    if res and len(res) > 0:
        df = pd.DataFrame(res)
        df['symbol'] = symbol
        if 'period' in df.columns:
            df['period'] = pd.to_datetime(df['period']).dt.strftime('%Y-%m-%d')
        await asyncio.sleep(0.2)
        return [df]
    logging.info(f"Fetched earnings surprises for {symbol}")
    return []

async def fetch_usa_spending(symbol, start_global, end_global):
    logging.info(f"Fetching USA spending for {symbol}")
    dfs = []
    end_date = end_global
    while end_date > start_global:
        start_date = max(end_date - timedelta(days=365), start_global)
        res = await async_finnhub_call(
            finnhub_client.stock_usa_spending,
            symbol,
            _from=start_date.strftime("%Y-%m-%d"),
            to=end_date.strftime("%Y-%m-%d")
        )
        if res and "data" in res and len(res["data"]) > 0:
            df = pd.DataFrame(res["data"])
            for col in ["actionDate", "performanceStartDate", "performanceEndDate"]:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col]).dt.strftime('%Y-%m-%d')
            df["symbol"] = symbol
            dfs.append(df)
        end_date = start_date
        await asyncio.sleep(0.2)
    logging.info(f"Fetched USA spending for {symbol}")
    return dfs

async def fetch_social_sentiment(symbol, start_global, end_global):
    """
    Fetch social sentiment (Reddit + Twitter) for a ticker in yearly chunks.
    """
    logging.info(f"Fetching social sentiment for {symbol}")
    dfs = []
    end_date = end_global
    while end_date > start_global:
        start_date = max(end_date - timedelta(days=365), start_global)
        try:
            res = await async_finnhub_call(
                finnhub_client.stock_social_sentiment,
                symbol,
                _from=start_date.strftime("%Y-%m-%d"),
                to=end_date.strftime("%Y-%m-%d")
            )
            if res and "data" in res and len(res["data"]) > 0:
                df = pd.DataFrame(res["data"])
                df["symbol"] = symbol
                dfs.append(df)
                end_date = start_date
            else:
                end_date = start_date
        except Exception as e:
            logging.error(f"Error fetching social sentiment for {symbol} ({start_date}-{end_date}): {e}")
        
        await asyncio.sleep(0.2)  # respect rate limits
    logging.info(f"Fetched social sentiment for {symbol}")
    return dfs


async def fetch_historical_esg(symbol, start_global, end_global):
    """
    Fetch historical ESG scores for a ticker in yearly chunks.
    """
    logging.info(f"Fetching historical ESG for {symbol}")
    dfs = []
    end_date = end_global
    tries = 0
    while end_date > start_global and tries < 5:
        start_date = max(end_date - timedelta(days=365), start_global)
        try:
            res = await async_finnhub_call(
                finnhub_client.company_historical_esg_score,
                symbol
            )
            if res and "data" in res and len(res["data"]) > 0:
                df = pd.json_normalize(res["data"])  # flatten nested 'data' dict
                df["symbol"] = symbol
                dfs.append(df)
                break
            else:
                break  # no more data, exit loop
        except Exception as e:
            logging.error(f"Error fetching ESG for {symbol} ({start_date}-{end_date}): {e}")
            await asyncio.sleep(0.2) 
            tries += 1            
            
        #end_date = start_date
        await asyncio.sleep(0.2)  # respect rate limits
    logging.info(f"Fetched historical ESG for {symbol}")
    return dfs


async def fetch_earnings_quality(symbol, freq="quarterly"):
    logging.info(f"Fetching earnings quality score for {symbol}")
    dfs = []
    tries = 0
    try:
        while True and tries < 5:
            try:
                res = await async_finnhub_call(finnhub_client.company_earnings_quality_score, symbol, freq)
                if res and "data" in res and len(res["data"]) > 0:
                    df = pd.DataFrame(res["data"])
                    df["symbol"] = res.get("symbol", symbol)
                    df["freq"] = res.get("freq", freq)
                    dfs.append(df)
                    break
                else:
                    break  # no data, exit loop
            except Exception as e:
                logging.warning(f"Error fetching earnings quality for {symbol}: {e}")
                tries += 1
                await asyncio.sleep(0.2)
        await asyncio.sleep(0.2)  # respect rate limits
        logging.info(f"Fetched earnings quality score for {symbol}")
        return dfs
    
    except Exception as e:
        logging.warning(f"Error fetching earnings quality for {symbol}: {e}")
        
    
    


async def fetch_congressional_trading(symbol, start_global, end_global):
    logging.info(f"Fetching Congressional Trading for {symbol}")
    dfs = []
    end_date = end_global
    while end_date > start_global:
        start_date = max(end_date - timedelta(days=365), start_global)
        try:
            res = await async_finnhub_call(
                finnhub_client.congressional_trading,
                symbol,
                _from=start_date.strftime("%Y-%m-%d"),
                to=end_date.strftime("%Y-%m-%d")
            )
            if res and "data" in res and len(res["data"]) > 0:
                df = pd.DataFrame(res["data"])
                df["symbol"] = symbol
                dfs.append(df)
                end_date = start_date
            else:
                end_date = start_date  # no more data, exit loop
                
        except Exception as e:
            logging.warning(f"Error fetching congressional trading for {symbol} ({start_date}-{end_date}): {e}")
            break
        await asyncio.sleep(0.2)  # respect API limits
    logging.info(f"Fetched Congressional Trading for {symbol}")
    return dfs


async def fetch_financial_statements(symbol, stmt, freq="quarterly"):
    
    try:
        res = await async_finnhub_call(
            finnhub_client.financials,
            symbol,
            stmt,
            freq
        )
        if res and "financials" in res and len(res["financials"]) > 0:
            df = pd.DataFrame(res["financials"])
            df['symbol'] = symbol
            df['statement_type'] = stmt
            
            #df = df.add_prefix(f"{stmt}_")  # prefix columns with statement type
            #logging.info(f"Fetched {stmt} for {symbol}: {len(df)} rows")
            await asyncio.sleep(0.5)  # respect API rate limits
            return [df]
        else:
            logging.info(f"No {stmt} data for {symbol}")
            return []
    except Exception as e:
        logging.warning(f"Error fetching {stmt} for {symbol}: {e}")
        return []
    
    
    

# --------------------------
# Run all tickers concurrently
# --------------------------
async def main(symbols):
    logging.info(f"Starting data fetch for {len(symbols)} symbols")

    error_tickers = []
    all_insider_dfs = []
    all_sentiment_dfs = []
    all_earnings_dfs = []
    all_usa_dfs = []
    all_mcap_dfs = []
    all_emp_dfs = []
    all_social_dfs = []
    all_esg_dfs = []
    all_eq_dfs = []
    all_congress_dfs = []   
    all_bs, all_ic, all_cf = [], [], []
    for symbol in tqdm.tqdm(symbols, desc="Processing tickers"):
        logging.info(f"Fetching data for {symbol}")
        try:
            tasks = [
                fetch_insider_transactions(symbol, datetime(2005,9,17), datetime(2025,9,17)),
                fetch_insider_sentiment(symbol, datetime(2005,9,17), datetime(2025,9,17)),
                fetch_earnings(symbol),
                fetch_usa_spending(symbol, datetime(2005,1,1), datetime(2025,9,17)),
                fetch_market_cap(symbol, datetime(2005,9,17), datetime(2025,9,17)),
                
                # fetch_employee_count(symbol, datetime(2005,9,17), datetime(2025,9,17)),
                # fetch_social_sentiment(symbol, datetime(2005,9,17), datetime(2025,9,17)),  # <-- new
                # fetch_historical_esg(symbol, datetime(2005,9,17), datetime(2025,9,17)),
                # fetch_earnings_quality(symbol, freq="quarterly"),  # <-- new
                # fetch_congressional_trading(symbol, datetime(2005,9,17), datetime(2025,9,17)),  # <-- new
                # fetch_financial_statements(symbol, 'bs'),
                # fetch_financial_statements(symbol, 'ic'),
                # fetch_financial_statements(symbol, 'cf'),
            ]


            results = await asyncio.gather(*tasks) 
            (insider_dfs, sentiment_dfs, earnings_dfs, usa_dfs, mcap_dfs) = results
            #(emp_dfs, social_dfs, esg_dfs, eq_dfs, cg_dfs, bs_dfs, ic_dfs, cf_dfs) = results
            if len(insider_dfs)> 0: all_insider_dfs.extend(insider_dfs)
            if len(sentiment_dfs)> 0: all_sentiment_dfs.extend(sentiment_dfs)
            if len(earnings_dfs)> 0: all_earnings_dfs.extend(earnings_dfs)
            if len(usa_dfs)> 0: all_usa_dfs.extend(usa_dfs)
            if len(mcap_dfs)> 0: all_mcap_dfs.extend(mcap_dfs)
            
            # if len(emp_dfs)> 0: all_emp_dfs.extend(emp_dfs)
            # if len(social_dfs)> 0: all_social_dfs.extend(social_dfs)
            # if len(esg_dfs)> 0: all_esg_dfs.extend(esg_dfs)
            # if len(eq_dfs) > 0: all_eq_dfs.extend(eq_dfs)
            # if len(cg_dfs)> 0: all_congress_dfs.extend(cg_dfs)
            # if len(bs_dfs)> 0: all_bs.extend(bs_dfs)
            # if len(ic_dfs)> 0: all_ic.extend(ic_dfs)
            # if len(cf_dfs)> 0: all_cf.extend(cf_dfs)
            
            await asyncio.sleep(2)  # buffer between tickers
        except Exception as e:
            logging.warning(f'Encountered error for ticker: {symbol}: {e}')
            error_tickers.append(symbol)
            continue
            

    if all_insider_dfs:
        pd.concat(all_insider_dfs, ignore_index=True).to_parquet("../data/raw_data/insider_transactions.parquet", index=False)
    if all_sentiment_dfs:
        pd.concat(all_sentiment_dfs, ignore_index=True).to_parquet("../data/raw_data/insider_sentiment.parquet", index=False)
    if all_earnings_dfs:
        pd.concat(all_earnings_dfs, ignore_index=True).to_parquet("../data/raw_data/earnings_surprises.parquet", index=False)
    if all_usa_dfs:
        pd.concat(all_usa_dfs, ignore_index=True).to_parquet("../data/raw_data/usa_spending_history.parquet", index=False)
    if all_mcap_dfs:
        pd.concat(all_mcap_dfs, ignore_index=True).drop_duplicates(subset=["atDate"]).sort_values("atDate").to_parquet("../data/raw_data/market_cap_history.parquet", index=False)
    if all_emp_dfs:
        pd.concat(all_emp_dfs, ignore_index=True).drop_duplicates(subset=["atDate"]).sort_values("atDate").to_parquet("../data/raw_data/employee_count_history.parquet", index=False)
    if all_social_dfs:
        pd.concat(all_social_dfs, ignore_index=True).to_parquet("../data/raw_data/social_sentiment.parquet", index=False)
    if all_esg_dfs:
        pd.concat(all_esg_dfs, ignore_index=True).drop_duplicates(subset=["period"]).sort_values("period").to_parquet("../data/raw_data/historical_esg_scores.parquet", index=False)
    if all_eq_dfs:
        pd.concat(all_eq_dfs, ignore_index=True).to_parquet("../data/raw_data/earnings_quality_score.parquet", index=False)
    if all_congress_dfs:
        pd.concat(all_congress_dfs, ignore_index=True).to_parquet("../data/raw_data/congressional_trading_disclosure.parquet", index=False)
    if all_bs:
        pd.concat(all_bs, ignore_index=True).to_parquet(f"../data/raw_data/financials_bs_quarterly_all_tickers.parquet", index=False)
    if all_ic:
        pd.concat(all_ic, ignore_index=True).to_parquet(f"../data/raw_data/financials_ic_quarterly_all_tickers.parquet", index=False)
    if all_cf:
        pd.concat(all_cf, ignore_index=True).to_parquet(f"../data/raw_data/financials_cf_quarterly_all_tickers.parquet", index=False)

    logging.info("Data fetch complete and saved.")
    logging.info('Tickers with errors: ')
    print(error_tickers)
    
    
# Run async
# Read in the saved tickers
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Loading top tickers from file...........................")
    top_sorted_tickers = []
    ticker_path = '../data/raw_data/tickers_448_v2.txt'
    with open(ticker_path, "r") as f:
        for line in f:
            # split by tab and take the first part (symbol)
            symbol = line.strip().split("\t")[0]
            top_sorted_tickers.append(symbol)
            
    top_sorted_tickers = list(set(top_sorted_tickers))
    logging.info(f"Loaded {len(top_sorted_tickers)} tickers")
    asyncio.run(main(top_sorted_tickers))



