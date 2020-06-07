"""
author: Florian Krach
"""

#%% IMPORTS
import numpy as np
import pandas as pd
import scipy.interpolate as interp
import os
import argparse


#%% GLOBAL VARIABLES
min_age = 6.0
child_adult_split = 18.0
max_age = 82.0
male = 0
female = 1
path = 'data/'
age_col = 'age'
gender_col = 'gender'
filename_children = 'children_LMS_{}_gender{}.csv'
filename_adults = 'adults_LMS_{}_gender{}.csv'



#%% FUNCTION DEFINITIONS
def compute_zscore(y_vals, t_vals, L, M, S, eps=0.00001):
    l_vals = L(t_vals)
    m_vals = M(t_vals)
    s_vals = S(t_vals)
    zscores = np.empty_like(y_vals)
    small = abs(l_vals) < eps
    zscores[small] = np.log(y_vals[small]/m_vals[small]) / s_vals[small]
    zscores[~small] = ((y_vals[~small]/m_vals[~small])**l_vals[~small] - 1)/(l_vals[~small]*s_vals[~small])

    # zscores = []
    # for y, l, m, s in zip(y_vals, l_vals, m_vals, s_vals):
    #     if abs(l) < eps:
    #         zscores.append(np.log(y/m) / s)
    #     else:
    #         zscores.append(((y/m)**l - 1)/(l*s))

    return zscores


def get_LMS(param, gender, filename, path=path):
    file_path = os.path.join(path, filename.format(param, gender))
    df_ref = pd.read_csv(file_path)
    L = interp.interp1d(x=df_ref['age'], y=df_ref['lambda'], kind='cubic')
    M = interp.interp1d(x=df_ref['age'], y=df_ref['mu'], kind='cubic')
    S = interp.interp1d(x=df_ref['age'], y=df_ref['sigma'], kind='cubic')
    return L, M, S


def compute_zscores_file(filename):
    df = pd.read_excel(filename, header=0, index_col=None)

    assert age_col in df.columns and gender_col in df.columns, \
        "file does not have the columns: {} and {}".format(age_col, gender_col)

    for col in df.columns:
        if col not in [age_col, gender_col, 'ID']:
            print('computing zscores for: {}'.format(col))
            new_col = 'zscore-{}'.format(col)
            for gender in [male, female]:
                for age_int in [[min_age, child_adult_split],
                                [child_adult_split, max_age]]:
                    df_curr = df.loc[(df[age_col] >= age_int[0]) &
                                     (df[age_col] < age_int[1]) &
                                     (df[gender_col] == gender), :]
                    if len(df_curr) > 0:
                        try:
                            if age_int[0] == min_age:
                                file_name = filename_children
                            else:
                                file_name = filename_adults
                            y_vals = df_curr[col].values
                            t_vals = df_curr[age_col].values
                            L, M, S = get_LMS(
                                param=col, gender=gender, filename=file_name)
                            zscores = compute_zscore(y_vals, t_vals, L, M, S)
                        except Exception:
                            zscores = None
                        df.loc[
                            (df[age_col] >= age_int[0]) &
                            (df[age_col] < age_int[1]) &
                            (df[gender_col] == gender), new_col] = zscores
    df.to_excel(filename, index=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="scientific zscore computations")
    parser.add_argument(
        '--filename', type=str,
        help="filename (with path if not in same directory), "
             "default: 'example_file.xlsx'",
        default="example_file.xlsx")

    args = parser.parse_args()
    filename = args.filename
    compute_zscores_file(filename)

















