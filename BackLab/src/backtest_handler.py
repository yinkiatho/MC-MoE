import yaml
import os 
import sys
from src.optimization_engine import OptimizationEngine
from src.backtest_engine import BacktestEngine
from src.log import Log
from objs.stock import initialize_stock
import platform
import time
import multiprocessing
import pandas as pd
import tqdm

class BacktestHandler:
    def __init__(self, BacktestLogic, Data, Optimization):
        self.optimization = Optimization()
        self.mixed_params = OptimizationEngine.mix_parameters(self.optimization.run, self.optimization.params)
        ##
        self.BacktestLogic = BacktestLogic
        self.Data = Data

    def run(self):
        if (self.optimization.run):
            # run optimization with backtest here and sends some parameter to backtest_engine
            if (hasattr(self.optimization, "parallel_run") and self.optimization.parallel_run):
                self.grid_search_parallel()
            else:
                # this will be treated as false
                self.grid_search_single()
            
        else:
            self.backtest_logic = self.BacktestLogic(self.mixed_params)
            self.inputs = self.backtest_logic.inputs
            self.log = Log(self.inputs.create_log, self.inputs.filename)
            self.data = self.Data(self.log)

            # run normal single backtest and sends parameter = None
            backtest_engine = BacktestEngine(self.backtest_logic, self.data, self.inputs, self.log)
            backtest_engine.run()

        return 

    def create_or_clear_folder(self, folder_path, clear = True):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        else:
            if (clear):
                for filename in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, filename)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            os.rmdir(file_path)
                    except Exception as e:
                        print(f"Failed to delete {file_path}. Reason: {e}")

    def grid_search_single(self):
        optimization_start_time = time.time()

        if (hasattr(self.optimization, "start_from")):
            start_from = self.optimization.start_from
            clear = False
        else:
            start_from = 0      
            clear = True

        for params_idx in tqdm.tqdm(range(start_from, len(self.mixed_params), 1)):
            params = self.mixed_params[params_idx]
            self.backtest_logic = self.BacktestLogic(params)
            self.inputs = self.backtest_logic.inputs
            if (not hasattr(self, "filename")):
                # only happens once and the first time
                self.filename = self.inputs.filename 
                self.inputs.csv_folder = self.filename

                folder_path = f"csv/{self.inputs.csv_folder}" if (platform.system() == "Linux") else f"csv\\{self.inputs.csv_folder}"
                self.create_or_clear_folder(folder_path, clear)

            self.parametric_filename = OptimizationEngine.parameter_to_filename(self.filename, params)
            self.inputs.filename = self.parametric_filename
            self.inputs.csv_folder = self.filename
            self.log = Log(self.inputs.create_log, self.inputs.filename, optimization_run=True)

            if not hasattr(self, "data"):
                self.data = self.Data(self.log)
            else:
                # we need to clean stock data objects, so that it works
                tickers = self.data.tickers
                stocks = initialize_stock(tickers)
                self.data.data_handler.stocks = stocks

            backtest_engine = BacktestEngine(self.backtest_logic, self.data, self.inputs, self.log)
            backtest_engine.run()
            # current_time_elapsed = time.time() - optimization_start_time
            # estimated_total_time = (current_time_elapsed/(params_idx+1-start_from))*(len(self.mixed_params)-start_from)
            # estimated_leftover_time = estimated_total_time - current_time_elapsed
            # print(f"[{(params_idx+1)}/{len(self.mixed_params)}][Time elapsed: {round(current_time_elapsed,1)}s, Est Time for Completion: {round(estimated_leftover_time,1)}s] Optimization on-going: {params}")
  
    def grid_search_parallel(self):
        self.backtest_logic = self.BacktestLogic(self.mixed_params[0])
        self.inputs = self.backtest_logic.inputs

        if (hasattr(self.optimization, "start_from")):
            start_from = self.optimization.start_from
            clear = False
        else:
            start_from = 0
            clear = True

        self.filename = self.inputs.filename 
        self.inputs.csv_folder = self.filename
        self.log = Log(self.inputs.create_log, self.inputs.filename, optimization_run=True)
        folder_path = f"csv/{self.inputs.csv_folder}" if (platform.system() == "Linux") else f"csv\\{self.inputs.csv_folder}"
        self.create_or_clear_folder(folder_path, clear)
        
        self.data = self.Data(self.log)
        num_cores_in_use = multiprocessing.cpu_count()-1
        print("Optimization running! Cores in-use: " + str(num_cores_in_use))

        with multiprocessing.Pool(processes=num_cores_in_use) as pool:
            results = list(tqdm.tqdm(pool.imap(self.backtest_wrapper, self.mixed_params[start_from:]), total=len(self.mixed_params[start_from:])))

        return 

    def backtest_wrapper(self, params):
        self.backtest_logic = self.BacktestLogic(params)
        self.inputs = self.backtest_logic.inputs
        self.parametric_filename = OptimizationEngine.parameter_to_filename(self.filename, params)
        self.inputs.filename = self.parametric_filename
        self.inputs.csv_folder = self.filename
        self.log = Log(self.inputs.create_log, self.inputs.filename, optimization_run=True)
        tickers = self.data.tickers
        stocks = initialize_stock(tickers)
        self.data.data_handler.stocks = stocks
        backtest_engine = BacktestEngine(self.backtest_logic, self.data, self.inputs, self.log)
        backtest_engine.run()
        
        return 
        
if __name__ == "__main__":
    backtest_handler = BacktestHandler()