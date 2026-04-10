import time 
from objs.stock import Stock
import numpy as np
from datetime import datetime

def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Strategy took {elapsed_time:.2f} seconds to execute.")
        return result
    
    return wrapper


