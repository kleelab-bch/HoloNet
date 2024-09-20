# Main
# The model code for HoloNet and Transfer Multi-TasK HoloNet in holographic image analysis for profiling
# breast cancer cells
# The model codes are in the model.py, and other functions are in Utilities.py
#
#
# TH Song Sep 2024

from lib.models import *

# Parameter Setting ====================================================================================================
# Can be changed by your own dataset

all_data_path = 'Data/All_Data.mat'

# Main =================================================================================================================
# There are three parameters --- path, Model_Type, and report_sign:
#       'path' is necessary to be provided for data reading
#       'Model_Type' includes two models: 'HoloNet' (default) and 'Trans_HoloNet'
#       'Working_Task' only for HoloNet. It includes Classification (0) and Regression (1). The default is 0.
#       'report_sign' is to save the model and print the results(True) or not(False). The default is False.
#

if __name__ == "__main__":
    main(all_data_path)




