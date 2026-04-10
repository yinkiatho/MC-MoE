import os
import sys
cur_path = os.getcwd()
parent_directory = os.path.dirname(cur_path)
sys.path.append(parent_directory + "\\" + "preloads")
sys.path.append(parent_directory + "/" + "preloads")
try:
    import preloads.load_all
except:
    import load_all