import pandas as pd
import numpy as np
import configparser

class Func:

    def format_to_two_decimal_places(self,value):
        return '{:.2f}'.format(value)


    def get_df(self,path):
        data_df = pd.read_csv(rf'{path}')
        selected_columns_df = data_df[['2017', '2018', '2019', '2020', '2021', '2022']]
        return selected_columns_df

    def get_nan_value(self,originaldf):
        nanno = []

        for i in range(originaldf.shape[0]):
            for j in range(originaldf.shape[1]):
                if np.isnan(originaldf.iloc[i, j]):
                    nanno.append([i, j])
        return nanno

    def get_nan_value_field(self,originaldf):
        nanno = []

        for j in range(originaldf.shape[1]):
            for i in range(originaldf.shape[0]):
                if np.isnan(originaldf.iloc[i, j]):
                    nanno.append([i, j])
        print(nanno)
        return nanno

    def get_value(self,nanarray,df):
        arrval = []
        for val in nanarray:
            arrval.append(df.iloc[val[0], val[1]])
        return arrval

    def get_absolute_error(self,actual_values,imputed_values):
        absolute_errors = []
        for i in range(0,len(actual_values)):
            absolute_errors.append(abs(actual_values[i] - imputed_values[i]))


        # Calculate the average absolute error
        average_absolute_error = sum(absolute_errors)//len(absolute_errors)

        return average_absolute_error

    def replace_nan_with_mean(self,row):
        mean = row.mean(skipna=True)  # Calculate row mean excluding NaN values
        return row.fillna(mean)  # Replace NaN with the mean

    def replace_nan_with_median(self,row):
        median = row.median(skipna=True)  # Calculate row mean excluding NaN values
        return row.fillna(median)  # Replace NaN with the mean


    def get_local_gradient(self,df):
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                if np.isnan(df.iloc[i, j]):
                    if j == 0:
                        df.iloc[i, j] = 2 * df.iloc[i, 1] - df.iloc[i, 2]
                    elif j == 5:
                        df.iloc[i, j] = 2 * df.iloc[i, 4] - df.iloc[i, 3]
                    else:

                        preceding_year = j - 1
                        following_year = j + 1

                        df.iloc[i, j] = (df.iloc[i, preceding_year] + df.iloc[i, following_year]) / 2
        return df


    # Define the L1 distance function
    def l1_distance(self,row1, row2):
        common_fields = set(row1.dropna().index) & set(row2.dropna().index)
        distance = 0
        count = 0

        for field in common_fields:
            distance += abs(row1[field] - row2[field])
            count += 1

        return distance / count if count > 0 else float('inf')

        # Define the L2 distance function : Average Squared differences
    def l2_distance(self, row1, row2):
        common_fields = set(row1.dropna().index) & set(row2.dropna().index)
        distance = 0
        count = 0

        for field in common_fields:
            distance += (row1[field] - row2[field])**2
            count += 1

        return distance / count if count > 0 else float('inf')

    def get_nn(self,df,type):
        dfres = df.copy(deep=True)
        for i, row1 in df.iterrows():
            min_distance = float('inf')  # Initialize the minimum distance to infinity
            nearest_neighbor = None  # Initialize the nearest neighbor

            for j, row2 in df.iterrows():
                if i != j:  # Exclude the current faculty member
                    # Check if the current faculty member has missing values in the same field as the neighbor
                    missing_fields_i = set(row1.index[row1.isna()])
                    missing_fields_j = set(row2.index[row2.isna()])
                    if not missing_fields_i & missing_fields_j:
                        if type == 'l1':
                            distance = self.l1_distance(row1, row2)
                        elif type == 'l2':
                            distance = self.l2_distance(row1, row2)
                        if distance < min_distance:
                            #print(type,distance,i,j)
                            min_distance = distance
                            nearest_neighbor = row2

            # Replace missing values in the current faculty member with values from the nearest neighbor
            if nearest_neighbor is not None:
                for field in row1.index[row1.isna()]:
                    dfres.at[i, field] = nearest_neighbor[field]
        return dfres
