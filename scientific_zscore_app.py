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
    zscores[~small] = ((y_vals[~small]/m_vals[~small])**l_vals[
        ~small] - 1)/(l_vals[~small]*s_vals[~small])

    # zscores = []
    # for y, l, m, s in zip(y_vals, l_vals, m_vals, s_vals):
    #     if abs(l) < eps:
    #         zscores.append(np.log(y/m) / s)
    #     else:
    #         zscores.append(((y/m)**l - 1)/(l*s))

    return zscores


def compute_zscore_splines(
        y_vals, t_vals, h_vals, L, M, S, Lspline, Mspline, Sspline, eps=0.00001
):
    l_vals = L(t_vals, h_vals, Lspline(t_vals))
    m_vals = M(t_vals, h_vals, Mspline(t_vals))
    s_vals = S(t_vals, h_vals, Sspline(t_vals))
    if type(s_vals) == float:
        s_vals = np.ones_like(m_vals) * s_vals
    if type(l_vals) == float:
        l_vals = np.ones_like(m_vals) * l_vals
    zscores = np.empty_like(y_vals)
    small = abs(l_vals) < eps
    zscores[small] = np.log(y_vals[small] / m_vals[small]) / s_vals[small]
    zscores[~small] = ((y_vals[~small] / m_vals[~small]) ** l_vals[
        ~small] - 1) / (l_vals[~small] * s_vals[~small])

    return zscores


def get_formula(s):
    s = s.strip().lower().replace("exp", "np.exp").replace("log", "np.log").replace(
        "ln", "np.log").replace("mspline", "spline").replace(
        "sspline", "spline").replace("lspline", "spline")
    f = eval("lambda age, height, spline: {}".format(s))
    return f


def get_LMS(param, gender, filename, path):
    file_path = os.path.join(path, filename.format(param, gender))
    try:
        df_ref = pd.read_csv(file_path, index_col=None)
    except Exception:
        df_ref = pd.read_excel(
            file_path.replace(".csv", ".xlsx"), index_col=None)

    # distinguish whether LMS params are given or need to be computed
    cols = df_ref.columns
    if "lambda" in cols and "mu" in cols and "sigma" in cols:
        L = interp.interp1d(x=df_ref['age'], y=df_ref['lambda'], kind='cubic')
        M = interp.interp1d(x=df_ref['age'], y=df_ref['mu'], kind='cubic')
        S = interp.interp1d(x=df_ref['age'], y=df_ref['sigma'], kind='cubic')
        return L, M, S, None, None, None
    elif "Mspline" in cols and "Sspline" in cols and "Lspline" in cols:
        M_formula = get_formula(str(df_ref["M"].values[0]))
        S_formula = get_formula(str(df_ref["S"].values[0]))
        L_formula = get_formula(str(df_ref["L"].values[0]))
        Lspline = interp.interp1d(
            x=df_ref['age'], y=df_ref['Lspline'], kind='cubic')
        Mspline = interp.interp1d(
            x=df_ref['age'], y=df_ref['Mspline'], kind='cubic')
        Sspline = interp.interp1d(
            x=df_ref['age'], y=df_ref['Sspline'], kind='cubic')
        return L_formula, M_formula, S_formula, Lspline, Mspline, Sspline


def compute_zscores_file(filename, datapath, age_col, gender_col, height_col,
                         exclude_zeros=True):
    df = pd.read_excel(filename, header=0, index_col=None)

    assert age_col in df.columns and gender_col in df.columns, \
        "file does not have the columns: {} and {}".format(age_col, gender_col)
    height_col_exist = True
    if height_col not in df.columns:
        height_col_exist = False

    for col in df.columns:
        if col not in [age_col, gender_col, height_col, 'ID', "PT_Nr"]:
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
                            if height_col_exist:
                                h_vals = df_curr[height_col].values
                            else:
                                h_vals = np.zeros_like(t_vals)
                            L, M, S, Lspline, Mspline, Sspline = get_LMS(
                                param=col, gender=gender, filename=file_name,
                                path=datapath)
                            if Lspline is None and Sspline is None \
                                    and Mspline is None:
                                zscores = compute_zscore(y_vals, t_vals, L, M, S)
                            else:
                                if not height_col_exist:
                                    print("WARNING: height column not found, "
                                          "using 0s instead")
                                zscores = compute_zscore_splines(
                                    y_vals, t_vals, h_vals, L, M, S,
                                    Lspline, Mspline, Sspline)
                            if exclude_zeros:
                                zscores[y_vals <= 0] = None
                        except Exception:
                            # raise
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
    parser.add_argument(
        '--datapath', type=str,
        help="path to data",
        default="data/")
    parser.add_argument(
        '--age', type=str,
        help="name of column with age values",
        default="age")
    parser.add_argument(
        '--gender', type=str,
        help="name of column with gender values",
        default="gender")
    parser.add_argument(
        '--height', type=str,
        help="name of column with height values",
        default="height")
    parser.add_argument(
        '--exclude_negative', type=bool,
        help="if True then does not compute zscores for values <=0",
        default=True)

    args = parser.parse_args()
    filename = args.filename
    datapath = args.datapath
    age_col = args.age
    gender_col = args.gender
    height_col = args.height
    exclude_zeros = args.exclude_negative
    compute_zscores_file(
        filename=filename, datapath=datapath, age_col=age_col,
        gender_col=gender_col, height_col=height_col,
        exclude_zeros=exclude_zeros)

















