from fancyimpute import IterativeImputer
import sys,os

import configparser

config = configparser.ConfigParser()
current_dir = os.path.dirname(os.path.abspath(__file__))

# Go up one level to the parent directory
parent_dir = os.path.dirname(current_dir)

# Define the path to the config file
config_file_path = os.path.join(parent_dir, 'config.ini')

sys.path.append(parent_dir)

from utils import *

# Read and parse the configuration file
config.read(config_file_path)

original_path = config.get('Path','originalpath')
missing_path = config.get('Path','missingpath')
# Step 1: Capture Original CSV
f = Func()
originaldf = f.get_df(rf'{original_path}')
#print(originaldf)

# Step 2: Capture Missing CSV
missingdf = f.get_df(rf'{missing_path}')

# Step 3: Get the cell list of NaN values
nanlist = f.get_nan_value(missingdf)

# step 4: Impute values using imputation algorithms
mice_imputer = IterativeImputer()
# imputing the missing value with mice imputer
data = mice_imputer.fit_transform(missingdf)
missingdf = pd.DataFrame(data)

# Step 5: Capture actual and pred values
actual_value = f.get_value(nanlist,originaldf)
pred_value = f.get_value(nanlist,missingdf)

# Step 6: Display the error
avg_abs_error = f.get_absolute_error(actual_value,pred_value)
print(f"Average Absolute Error using Mice is {avg_abs_error}")
