import pandas as pd
import platform 
import os 

class FileExport:
    @staticmethod
    def run(inputs, performance_tracker, snapshot):
        filename = inputs.filename
        folder = inputs.csv_folder
        if (inputs.create_performance_file):
            all_data_row = []

            for bar_date_time, close2close_pnl in performance_tracker.portfolio_close2close_pnl_pct.items():
                data_row = [close2close_pnl]
                if (inputs.snapshot):
                    data_row = data_row + [snapshot.open_session_nav[bar_date_time], snapshot.open_prices[bar_date_time],
                                           snapshot.open_units[bar_date_time], snapshot.open_messages[bar_date_time],
                                           snapshot.open_current_proportions[bar_date_time], snapshot.open_set_proportions[bar_date_time],
                                           snapshot.open_borrowing_amt[bar_date_time],
                                           snapshot.open_dollar_pnl[bar_date_time], snapshot.open_pct_pnl[bar_date_time],
                                           snapshot.close_session_nav[bar_date_time], snapshot.close_prices[bar_date_time],
                                           snapshot.close_units[bar_date_time], snapshot.close_messages[bar_date_time],
                                           snapshot.close_current_proportions[bar_date_time], snapshot.close_set_proportions[bar_date_time],
                                           snapshot.close_borrowing_amt[bar_date_time],
                                           snapshot.close_dollar_pnl[bar_date_time], snapshot.close_pct_pnl[bar_date_time]]

                all_data_row.append(data_row)

            export_columns = ["pnl(%)"]

            if (inputs.snapshot):
                export_columns = export_columns + ["OpenSessionNav", "OpenPrices",
                                                   "OpenUnitsHolding", "OpenMessages",
                                                   "OpenCurrentProportions", "OpenSetProportions",
                                                   "OpenBorrowingAmt",
                                                   "OpenDollarPnl", "OpenPctPnl",
                                                   "CloseSessionNav", "ClosePrices",
                                                   "CloseUnitsHolding", "CloseMessages",
                                                   "CloseCurrentProportions", "CloseSetProportions",
                                                   "CloseBorrowingAmt",
                                                   "CloseDollarPnl", "ClosePctPnl"]

            export_df = pd.DataFrame(all_data_row, index = performance_tracker.portfolio_close2close_pnl_pct.keys(), columns=export_columns)
            print(os.getcwd())

            #csv_filepath = f"csv/{folder}/{filename}.csv" if (folder is not None) else f"csv/{filename}.csv"
            csv_filepath = f"BackLab/csv_data/{folder}/{filename}.csv" if (folder is not None) else f"BackLab/csv_data/{filename}.csv"
            print(csv_filepath)
            if (platform.system() != "Linux"):
                csv_filepath = csv_filepath.replace("/", "\\")
   
            export_df.to_csv(csv_filepath)

        return 