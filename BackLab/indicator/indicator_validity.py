import numpy as np
from series import Series

def indicator_validity(active_bars, *vars):
    """"
    vars can be Series or other variable type
    """
    if (active_bars > 0):
        for var in vars:
            series_var_active = 1 if (repr(var) == "Series") else -1
            if (series_var_active >= 0 and np.isnan(var[0])):
                raise Exception("Indicator validity has raised an error as the indicator is already active but input values is NaN")
            
        return active_bars+1
    
    else:
        all_active_bar = 1
        for var in vars:
            series_var_active = 1 if (repr(var) == "Series") else -1
            if (series_var_active >= 0 and ((var.length() != 0 and np.isnan(var[0])) or var.length() == 0)):
                all_active_bar = 0
                break

        return all_active_bar
    