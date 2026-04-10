import os
import pandas as pd
from itertools import combinations
import tqdm
import time
import numpy as np
from pbo.performance_stats import PerformanceStatistics
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import copy


class CSCV:
    """
    - This is an implementation from https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253
    - The probability of Backtest Overfitting paper describes a framework that estimates the probability of backtest over-fitting (PBO) specifically in context 
    of investment simulations. 
    - CSCV represents "Combinatorially Symmetric Cross-Validation"
    - Utilising "Percentile Method", aiming for top percentile of ranks then get the average logits out of it, this might be a better representation of the optimization framework
    """
    def __init__(self, folder_path):
        # folder is the path where the csvs are held
        self.folder_path = folder_path
        self.folder_name = self.folder_path[self.folder_path.find("\\")+1:]
        self.PS = PerformanceStatistics()

    def load_all_csv(self):
        folder_path_dir = os.listdir(self.folder_path)
        self.csv_dict = {}

        for file_name in folder_path_dir:
            if (file_name[0] != "_"):
                csv_path = os.path.join(self.folder_path, file_name)
                df = pd.read_csv(csv_path, index_col = 0)
                df.index = pd.to_datetime(df.index)
                self.csv_dict[file_name] = df
                
    def load_all_csv_with_keywords(self, keywords):
        folder_path_dir = os.listdir(self.folder_path)
        self.csv_dict = {}

        for file_name in folder_path_dir:
            if (file_name[0] != "_" and all(file_name.__contains__(keyword) for keyword in keywords) and file_name.endswith(".csv")):
                csv_path = os.path.join(self.folder_path, file_name)
                df = pd.read_csv(csv_path, index_col = 0)
                df.index = pd.to_datetime(df.index)
                self.csv_dict[file_name] = df

        print(f"Total CSVs loaded: {len(self.csv_dict)}")
            
    def get_unified_indexes(self):
        unified_indexes = None
        index_start = 0

        for key, df in self.csv_dict.items():
            if (unified_indexes == None):
                unified_indexes = set(list(df.index))
                index_start = len(list(df.index))
            else:
                cur_indexes = set(list(df.index))
                unified_indexes = unified_indexes.intersection(cur_indexes)

        self.unified_indexes = sorted(list(unified_indexes))

        if (index_start != len(unified_indexes)):
            print(f"Discrepancy found in starting index length ({index_start}) and final unified index ({len(unified_indexes)})")
            
        return 
    
    def create_M_matrix(self):
        # Step 1: M matrix is a variable described in the paper; M is of order of (T x N)
        M_matrix = []
       
        for key, df in self.csv_dict.items():
            M_matrix.append(df.loc[self.unified_indexes]["pnl(%)"].values)
        
        # transpose it to a right format
        self.M_matrix = [list(row) for row in zip(*M_matrix)]

        return

    def split_matrix(self, S = 16):
        # Step 2: create S partionas of M matrix, now known as M_s of order (T/S x N)
        if (S % 2 != 0):
            raise Exception("The variable S must be even!")
        
        if (S <= 1):
            raise Exception("The variable S should be more than 1!")

        self.S = S
        self.M_dict = {}
        self.partitions = []
        splitting_row_size = int(len(self.M_matrix)/self.S)
        for i in range(S):
            self.M_dict[i+1] = self.M_matrix[i*splitting_row_size:((i+1)*splitting_row_size)]
            self.partitions.append(i+1)
        return 
    
    def form_combinations_pairs(self):
        # Step 3: form all combinations C_S of M_s
        self.combinations_pairs = list(combinations(self.partitions, int(self.S/2)))
        print(f"Total combinations: {len(self.combinations_pairs)}; total points to evaluate: {len(self.combinations_pairs)*len(self.M_matrix)*len(self.M_matrix[0])}")
        return 
    
    def form_combo_set(self, combo):
        matrix = []
        for s in combo:
            matrix = matrix + self.M_dict[s]
        return matrix

    def add_to_R_bar_sets(self, J_hat_matrix_ps):
        self.R_bar_sets_sets.append(J_hat_matrix_ps)
        return
    
    def get_logit(self, ps_method="sharpe", risk_free_rate = 0.0, percentile_start = 1, percentile_end = 1, one_year_bars = 252):
        # Step 4: compute for logit values through inner steps a to g
        # training_combo and testing_combo will be a list with s specified for training combo and s^{hat} as the not part of s
        if (percentile_start < 0  or percentile_start > 1 or percentile_end < 0  or percentile_end > 1):
            raise Exception("Percentile should be in between 0 and 1 (inclusive)")
        if (percentile_end < percentile_start):
            raise Exception("Percentile end should be larger or equivalent than percentile start.")

        self.logit_lambda_c_collections = []
        self.ps_pairs = []
        self.R_bar_sets_sets = []
        printed = False
        for training_combination in tqdm.tqdm(self.combinations_pairs):
            # creating the list of sorted index combination
            training_combo = sorted(list(training_combination))
            testing_combo = sorted([x for x in self.partitions if x not in training_combo])

            J_matrix = self.form_combo_set(training_combo)
            J_hat_matrix = self.form_combo_set(testing_combo)

            # ps stands for performance statistics
            J_matrix_ps = self.PS.run(J_matrix, method = ps_method, risk_free = risk_free_rate, one_year_bars = one_year_bars)
            J_hat_matrix_ps = self.PS.run(J_hat_matrix, method = ps_method, risk_free = risk_free_rate, one_year_bars=one_year_bars)

            # for stochastic dominance use
            self.add_to_R_bar_sets(J_hat_matrix_ps)

            R_c = self.PS.rank_ps(J_matrix_ps) # rank on the J matrix
            R_hat_c = self.PS.rank_ps(J_hat_matrix_ps) # rank on the J hat matrix

            r_c_n_star = max(R_c) # get the best rank (highest value not lowest!)
            r_c_n_star_col = np.where(R_c == r_c_n_star)[0][0] # get the column location of the R_c
            r_hat_c_n_star = R_hat_c[r_c_n_star_col] # get the corresponding rank of r_c_n_star from the R hat 

            if (percentile_start != 1 and percentile_end != 1):
                highest_rank = r_c_n_star
                additional_ranks_start = max(int(percentile_start * highest_rank)+1,1)
                additional_ranks_end = min(int(percentile_end * highest_rank), highest_rank)
                if (not printed):
                    print(f"Percentile method: In-sample result choosing from rank {additional_ranks_start} to rank {additional_ranks_end}")
                    printed = True

                total_logit_lambda_sampling = 0
                in_sample_ps = []
                oo_sample_ps = []
                for r_c_n_star_reduction_rank in range(additional_ranks_start, additional_ranks_end+1, 1):
                    target_rank = r_c_n_star_reduction_rank
                    good_flow = True
                    while (good_flow):
                        try:
                            r_c_n_star_reduction_search_col = np.where(R_c == target_rank)[0][0]
                            r_hat_c_n_star_reduction_corresponding = R_hat_c[r_c_n_star_reduction_search_col]
                            in_sample_ps.append(J_matrix_ps[r_c_n_star_reduction_search_col])
                            oo_sample_ps.append(J_hat_matrix_ps[r_c_n_star_reduction_search_col])
                            omega_hat = r_hat_c_n_star_reduction_corresponding/(len(J_matrix[0])+1)
                            logit_lambda = math.log(omega_hat/(1-omega_hat))
                            total_logit_lambda_sampling += logit_lambda
                            good_flow = False
                        except:
                            target_rank -= 1 

                average_logit_lambda = total_logit_lambda_sampling/(additional_ranks_end-additional_ranks_start+1)
                
                average_in_sample_ps = sum(in_sample_ps)/len(in_sample_ps)
                average_oos_sample_ps = sum(oo_sample_ps)/len(oo_sample_ps)
                
                self.ps_pairs.append(tuple([average_in_sample_ps, average_oos_sample_ps]))
                self.logit_lambda_c_collections.append(average_logit_lambda)
            else:
                self.ps_pairs.append(tuple([J_matrix_ps[r_c_n_star_col], J_hat_matrix_ps[r_c_n_star_col]]))
                omega_hat_c = r_hat_c_n_star/(len(J_matrix[0])+1)
                logit_lambda_c = math.log(omega_hat_c/(1-omega_hat_c))
                self.logit_lambda_c_collections.append(logit_lambda_c)
        
        return
    
    def construct_distribution(self):
        # step 5: compute distribution of ranks OOS by collecting the logit_lambda_c_collections
        sorted_logit_lambda_c_collections = sorted(self.logit_lambda_c_collections)
        cdf = {}
        increment = 1/len(sorted_logit_lambda_c_collections)
        cdf_tracker = 0.0
        for logit_lambda_c in sorted_logit_lambda_c_collections:
            cdf_tracker = cdf_tracker + increment
            cdf[logit_lambda_c] = round(cdf_tracker, 7)

        pdf = {}
        cluster_bin_size = 0.2
        cluster = {}
        cluster_current_midpt = min(list(cdf.keys()))+cluster_bin_size*0.5

        prev_logit_val = None
        cache_logit_element = None

        for logit_lambda_c, cum_probability in cdf.items():
            while (not (logit_lambda_c <= cluster_current_midpt-cluster_bin_size*0.5-0.00000001 and logit_lambda_c <= cluster_current_midpt+cluster_bin_size*0.5+0.00000001)):
                cluster_current_midpt += cluster_bin_size
            
            if (cluster_current_midpt not in cluster):
                cluster[cluster_current_midpt] = 1
            else:
                cluster[cluster_current_midpt] += 1

            if (cache_logit_element is not None):
                # get pdf using [dF(lambda)/d lambda]
                pdf[cache_logit_element] = (cdf[logit_lambda_c]-cdf[cache_logit_element])/(logit_lambda_c - cache_logit_element)
                cache_logit_element = None

            if (prev_logit_val is None):
                cache_logit_element = logit_lambda_c
            else:
                pdf[logit_lambda_c] = (cdf[logit_lambda_c] - cdf[prev_logit_val])/(logit_lambda_c - prev_logit_val)

            prev_logit_val = logit_lambda_c

        self.lowest_r_bar_ps = 9999999999999999999999999 # need it to filter stochastic dominance plot
        for ps_pair in self.ps_pairs:
            self.lowest_r_bar_ps = min(self.lowest_r_bar_ps, ps_pair[1])

        total_values_clusters = sum(list(cluster.values()))
 
        for cluster_mid_pt, frequency in cluster.copy().items():
            cluster[cluster_mid_pt] = cluster[cluster_mid_pt]/total_values_clusters
        
        sum_prob = sum(cluster.values())
        print(sum_prob)
        plt.plot(cdf.keys(), cdf.values())
        plt.xlabel("Logit lambda")
        plt.ylabel("Cumulative Density Function (CDF)")
        plt.show()
        plt.bar(cluster.keys(), cluster.values())
        plt.xlabel("Logit lambda")
        plt.ylabel("Probability ")
        plt.show()
        self.cdf_logit_c = cdf
        self.pdf_logit_c = pdf
        return
    
    def pbo_calc(self):
        # this function calculates the probability of backtest overfitting
        self.pbo_val = 0.0
        cache_logit_lambda = None
        prev_logit_val = None
        self.pdf_logit_c = dict(sorted(self.pdf_logit_c.items()))
  
        for logit_lambda_c, pdf_val in self.pdf_logit_c.items():
            if (cache_logit_lambda is not None):
                if (cache_logit_lambda < 0):
                    self.pbo_val += (self.pdf_logit_c[cache_logit_lambda])*(logit_lambda_c-cache_logit_lambda)
                cache_logit_lambda = None

            if (prev_logit_val is None):
                cache_logit_lambda = logit_lambda_c
            else:
                if (logit_lambda_c < 0):
                    self.pbo_val += pdf_val*(logit_lambda_c-prev_logit_val)

            prev_logit_val = logit_lambda_c

        pbo_string = "{:.1f}%".format(self.pbo_val*100)
        print(f"Probability of Backtest Overfitting: {pbo_string}")
        return
    

    def performance_degradation_and_prob_of_loss(self):
        R_n_star = []
        R_bar_n_star = []

        for ps_pair in self.ps_pairs:
            R_n_star.append(ps_pair[0])
            R_bar_n_star.append(ps_pair[1])

        model = LinearRegression()
        R_n_star_reshaped = np.array(R_n_star).reshape(-1,1)
        model.fit(R_n_star_reshaped, R_bar_n_star)
        slope = model.coef_[0]
        intercept = model.intercept_
        eqn = f"OOS={round(slope,2)} * IS + {round(intercept,2)}"
        min_x = min(R_n_star)
        max_x = max(R_n_star)
        splits = 1000
        spacing = (max_x-min_x)/splits
        x_values = [min_x + spacing*i for i in range(splits)]
        x_values_reshaped = np.array(x_values).reshape(-1,1)
        y_pred_vals = model.predict(x_values_reshaped)

        count_neg = [1 for val in R_bar_n_star if val < 0]
        probability_of_negative = len(count_neg)/len(R_bar_n_star)
        
        pon_string = "{:.1f}%".format(probability_of_negative*100)
        prob_neg_string  = f"Prob[Out-of-Sample Performance Stats < 0]= {pon_string}"

        plt.plot(x_values, y_pred_vals, color='r', label=eqn+"\n"+prob_neg_string)
        plt.scatter(R_n_star, R_bar_n_star, s=0.4)
        plt.xlabel("In-Sample Performance Statistics")
        plt.ylabel("Out-of-Sample Performance Statistics")
        plt.legend()

        return 

    def first_order_stochastic_dominance(self):
        R_bar_n_stars = []
        R_bar_means = []
        for R_bar_sets in self.R_bar_sets_sets:
            R_bar_n_stars.append(max(R_bar_sets))
            R_bar_means.append(np.mean(R_bar_sets))

        R_bar_n_stars = sorted(R_bar_n_stars)
        R_bar_means = sorted(R_bar_means)

        flattened_R_bar = sorted([R_bar for R_bar_sets in self.R_bar_sets_sets for R_bar in R_bar_sets], reverse = True)
        prob_R_bar_n_star_more_than_x = {}
        prob_R_bar_mean_more_than_x = {}
        prev_R_bar_n_stars_i = len(R_bar_n_stars)
        prev_R_bar_mean_j = len(R_bar_means)
        stochastic_dominance = True
        for R_bar in tqdm.tqdm(flattened_R_bar):
            for i in range(0, prev_R_bar_n_stars_i):
                if (R_bar_n_stars[i] >= R_bar):
                    break
                prev_R_bar_n_stars_i = i

            for j in range(0, prev_R_bar_mean_j):
                if (R_bar_means[j] >= R_bar):
                    break
                prev_R_bar_mean_j = j

            prob_R_bar_n_star_more_than_x[R_bar] = 1-(prev_R_bar_n_stars_i+1)/len(R_bar_n_stars)
            prob_R_bar_mean_more_than_x[R_bar] = 1-(prev_R_bar_mean_j+1)/len(R_bar_means)

            if (prob_R_bar_n_star_more_than_x[R_bar] < prob_R_bar_mean_more_than_x[R_bar]):
                stochastic_dominance = False

        self.prob_R_bar_n_star_more_than_x = dict(sorted(prob_R_bar_n_star_more_than_x.items()))
        self.prob_R_bar_mean_more_than_x = dict(sorted(prob_R_bar_mean_more_than_x.items()))

        if (stochastic_dominance):
            stochastic_dominance_statement = "First order stochastic dominance ESTABLISHED: Prob[R_bar_n_star > x] >= Prob[mean(R_bar) > x] for all x"
        else:
            stochastic_dominance_statement = "First order stochastic dominance FAILED: Prob[R_bar_n_star > x] >= Prob[mean(R_bar) > x] for all x IS NOT TRUE."
        print(stochastic_dominance_statement)

        filtered_prob_R_bar_n_star_more_than_x = copy.deepcopy(prob_R_bar_n_star_more_than_x)
        filtered_prob_R_bar_mean_more_than_x = copy.deepcopy(prob_R_bar_mean_more_than_x)
        for R_bar, prob in prob_R_bar_n_star_more_than_x.items():
            if (R_bar < self.lowest_r_bar_ps or prob > 0.98):
                del filtered_prob_R_bar_n_star_more_than_x[R_bar]

        for R_bar, prob in prob_R_bar_mean_more_than_x.items():
            if (R_bar < self.lowest_r_bar_ps or prob > 0.98):
                del filtered_prob_R_bar_mean_more_than_x[R_bar]

        plt.plot(filtered_prob_R_bar_n_star_more_than_x.keys(), filtered_prob_R_bar_n_star_more_than_x.values(), label = "Prob[R_bar_n_star > x]")
        plt.plot(filtered_prob_R_bar_mean_more_than_x.keys(), filtered_prob_R_bar_mean_more_than_x.values(), label = "Prob[mean(R_bar) > x]")
        plt.xlabel("Out-of-sample Performance Statistics (x)")
        plt.ylabel("Probability")
        plt.legend()

        return
    
    def second_order_stochastic_dominance(self):
        self.prob_R_bar_n_star_less_than_x = {}
        self.prob_R_bar_mean_less_than_x = {}

        for R_bar, prob in self.prob_R_bar_n_star_more_than_x.items():
            self.prob_R_bar_n_star_less_than_x[R_bar] = 1-prob
            self.prob_R_bar_mean_less_than_x[R_bar] = 1-self.prob_R_bar_mean_more_than_x[R_bar]

        # second-order stochastic dominance
        sd2 = {}
        sd2_dominance = True
        cache_R_bar = None
        prev_R_bar = None

        for R_bar, _ in self.prob_R_bar_n_star_less_than_x.items():
            if (cache_R_bar is not None):
                sd2[cache_R_bar] = (self.prob_R_bar_mean_less_than_x[cache_R_bar] - self.prob_R_bar_n_star_less_than_x[cache_R_bar])*(R_bar-cache_R_bar)
                cache_R_bar = None

            if (prev_R_bar is None):
                cache_R_bar = R_bar
            else:
                sd2[R_bar] = sd2[prev_R_bar] + (self.prob_R_bar_mean_less_than_x[R_bar] - self.prob_R_bar_n_star_less_than_x[R_bar])*(R_bar-prev_R_bar)
                
                if (sd2[R_bar] < 0):
                    sd2_dominance = False

            prev_R_bar = R_bar

        if (sd2_dominance):
            stochastic_dominance_statement = "Second order stochastic dominance ESTABLISHED: \Int_{-inf, x} (Prob[mean(R_bar) <= x]-Prob[mean(R_bar) <= x]) dx >= 0 for all x"
        else:
            stochastic_dominance_statement = "Second order stochastic dominance FAILED: \Int_{-inf, x} (Prob[mean(R_bar) <= x]-Prob[mean(R_bar) <= x]) dx >= 0 for all x IS NOT TRUE"

        print(stochastic_dominance_statement)

        filtered_prob_R_bar_n_star_less_than_x = copy.deepcopy(self.prob_R_bar_n_star_less_than_x)
        filtered_prob_R_bar_mean_less_than_x = copy.deepcopy(self.prob_R_bar_mean_less_than_x)
        for R_bar, prob in self.prob_R_bar_n_star_less_than_x.items():
            if (R_bar < self.lowest_r_bar_ps or prob > 0.95):
                del filtered_prob_R_bar_n_star_less_than_x[R_bar]

        for R_bar, prob in self.prob_R_bar_mean_less_than_x.items():
            if (R_bar < self.lowest_r_bar_ps or prob > 0.98):
                del filtered_prob_R_bar_mean_less_than_x[R_bar]

        min_key = min(list(filtered_prob_R_bar_n_star_less_than_x.keys()))
        min_key = min(min_key, min(list(filtered_prob_R_bar_mean_less_than_x.keys())))
        
        filtered_sd2 = copy.deepcopy(sd2)
        for R_bar, sd2_cdf in sd2.items():
            if (R_bar < min_key):
                del filtered_sd2[R_bar]

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx() 

        ax1.plot(filtered_prob_R_bar_n_star_less_than_x.keys(), filtered_prob_R_bar_n_star_less_than_x.values(), label = "Prob[R_bar_n_star <= x]", color = 'b')
        ax1.plot(filtered_prob_R_bar_mean_less_than_x.keys(), filtered_prob_R_bar_mean_less_than_x.values(), label = "Prob[mean(R_bar) <= x]", color = 'g')

        ax2.plot(filtered_sd2.keys(), filtered_sd2.values(), label = "Second-Order Stochastic Dominance")

        ax1.set_xlabel("Out-of-sample Performance Statistics (x)")
        ax1.set_ylabel("Probability")
        ax2.set_ylabel("Second-Order Stochastic Dominance")
        ax1.legend(loc="upper left")
        ax2.legend(loc="center left")
        plt.show()