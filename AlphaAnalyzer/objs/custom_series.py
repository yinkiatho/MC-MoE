import numpy as np
import pandas as pd

class CustomSeries:
    def __init__(self, specific_csv = [], weights = [], folder = None, percentile_start = 1, percentile_end = 1):
        self.specific_csv = specific_csv
        self.weights = weights
        self.folder = folder
        self.percentile_start = percentile_start
        self.percentile_end = percentile_end

        if (len(specific_csv) != len(weights)) :
            raise Exception("Mismatch in the length of specific csv and weights")
        
        if (len(specific_csv) != 0):
            # load the csvs
            returns_lst = []
            file_names = []
            for csv in specific_csv:
                folder_name_start = csv.rfind("\\")
                file_names.append(csv[folder_name_start+1:].replace(".csv",""))
                returns_lst.append(pd.read_csv(csv, index_col=0))

            combination_df = pd.DataFrame()
            for performance_df, file_name in zip(returns_lst, file_names):
                if (combination_df.shape[0] == 0):
                    z = performance_df[["pnl(%)"]]
                    combination_df = z.rename(columns={"pnl(%)":file_name})

                else:
                    z = performance_df[["pnl(%)"]]
                    combination_df = pd.concat([combination_df, z.rename(columns={"pnl(%)":file_name})], axis=1)
                
            self.combination_df = combination_df.fillna(0)

    def create_series(self, file_name = "temp"):
        custom_series = pd.DataFrame(index = self.combination_df.index, columns = ["pnl(%)"])
        custom_series = custom_series.fillna(0)
        iterator = 0

        for col in self.combination_df.columns:
            custom_series["pnl(%)"] += self.weights[iterator] * self.combination_df[col]
            iterator += 1

        custom_series["pnl(%)"] = custom_series["pnl(%)"]
        self.custom_series = custom_series
        self.custom_series.to_csv(f"csv\\{file_name}.csv")
        return 
