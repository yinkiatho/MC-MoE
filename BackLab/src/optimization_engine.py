import itertools

class OptimizationEngine:
    @staticmethod
    def mix_parameters(run, params):
        """
        params.keys = parameters name
        params.values = parameters values to iterate through
        return values:  list of parameters in the form of dictionary
        """
        if (not run):
            return None
        
        params_list = []
        params_names = []
        for param_name, params_values in params.items():
            params_list.append(params_values)
            params_names.append(param_name)

        mixed_params = []
        for l in itertools.product(*params_list):
            iterator = 0
            params_dict = {}
            for j in l:
                params_dict[params_names[iterator]] = j
                iterator += 1
            mixed_params.append(params_dict)
        
        return mixed_params
    
    @staticmethod
    def parameter_to_filename(default_filename, params_dict):
        filename = default_filename + "["
        for parameter_name, parameter_values in params_dict.items():
            parameter_label = parameter_name + "=" + str(parameter_values).replace(".", "p")
            filename += parameter_label + ";"

        filename = filename[:len(filename)-1]
        filename += "]"
        return filename

            


if __name__ == "__main__":
    params = {'a': [1,2,3], 'b': [4,5,6], 'c': [7,8,9]}
    
    mixed_params = OptimizationEngine.mix_parameters(True, params)
    print(mixed_params)
    print(len(mixed_params))