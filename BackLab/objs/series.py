import numpy as np

class Series:
    '''
    O(1) append to the front of series (technically behind)
    O(1) look up time
    behaves exactly like list but with reversed indexing
    '''
    def __init__(self, values = None):
        if (values is None):
            self.values = []
        else:
            if (not isinstance(values, list)):
                raise Exception(f"Type of values should be list not {type(values)}")
            
            self.values = values

    def __getitem__(self, index):
        if (not isinstance(index, int) and not isinstance(index, slice)):
            raise Exception(f"{self.__class__.__name__} Indexer is not an integer.")

        if (isinstance(index, int)):
            series_length = len(self.values)
            reverse_index = series_length - index - 1

            if (reverse_index < 0 or reverse_index >= series_length):
                raise Exception(f"[{self.__class__.__name__}] Exceed bound of indexing.")
            
            return self.values[reverse_index]
        
        else :
            return self.values[index]
        
    def length(self):
        return len(self.values)
    
    def add(self, val):
        self.values.append(val)
        return 
    
    def pop_oldest(self):
        if not self.values:
            raise Exception(f"[{self.__class__.__name__}] No elements to pop.")
        return self.values.pop(0)
    
    def pop_latest(self):
        if not self.values:
            raise Exception(f"[{self.__class__.__name__}] No elements to pop.")
        return self.values.pop()

    def __str__(self):
        start_print = "Series(["
        end_print = "])"
        mid_print = ""
        series_length = len(self.values)
        for i in range(series_length):
            mid_print += str(self.values[series_length-i-1])
            if (i != series_length - 1):
                mid_print += ","

        return f"{start_print}{mid_print}{end_print}"
    
    def __repr__(self):
        return "Series"

