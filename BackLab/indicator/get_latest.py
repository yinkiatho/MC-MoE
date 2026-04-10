import indicator_prerun
import numpy as np
from series import Series
from indicator_validity import indicator_validity
import pandas as pd

class GetLatest:
    def __init__(self):
        self.values = Series()
        self.active_bars = 0

    def __getitem__(self, index):
        return self.values[index]
    
    def __str__(self):
        return str(self.values)
    
    def __repr__(self):
        return repr(self.values)

    def length(self):
        return self.values.length()

    def update(self, df, attribute, target):
        """
        purpose: to get the nearest target but not exceeding the target (target is normally a date)
        variables: [df] is the general purpose SORTED dataframe whereby the index of df is used as the key for searching the target (make sure that the df is sorted!)
                   [attribute] is the column name of the df; it will update self.values to df.loc[nearest_row, attribute]
                   [target] is the value/date
        
        """
        # indicator logic
        if (not hasattr(self, "attribute_col_idx")):
            self.attribute_col_idx = df.columns.get_loc(attribute)
        
        index_found = df.index.get_indexer([target], method="ffill")

        if (index_found[0] != -1):
            target_value = df.iat[index_found[0], self.attribute_col_idx]
            self.values.add(target_value)
        else:
            self.values.add(np.nan)

        return 
    