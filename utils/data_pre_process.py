"""# Data preprocessig"""

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import pandas as pd


def data_processing():
    telecom = pd.read_csv("input/churn-bigml-80.csv")
    telecom_test = pd.read_csv("input/churn-bigml-20.csv")

    # Removing correlated and unneccessary columns
    col_to_drop = ['State', 'Area code', 'Total day charge', 'Total eve charge',
                   'Total night charge', 'Total intl charge']

    telecom = telecom.drop(columns=col_to_drop, axis=1)
    telecom_test = telecom_test.drop(columns=col_to_drop, axis=1)

    # target column
    target_col = ["Churn"]

    # number of levels in feature to be a categorical feature
    nlevels = 6

    # Separating categorical and numerical columns
    # categorical columns
    cat_cols = list(set(telecom.nunique()[telecom.nunique() < nlevels].keys().tolist()
                        + telecom.select_dtypes(include='object').columns.tolist()))
    cat_cols = [x for x in cat_cols if x not in target_col]

    # numerical columns
    num_cols = [x for x in telecom.columns if x not in cat_cols + target_col]

    # Binary columns with 2 values
    bin_cols = telecom.nunique()[telecom.nunique() == 2].keys().tolist()

    # Columns more than 2 values
    multi_cols = [i for i in cat_cols if i not in bin_cols]

    # Label encoding Binary columns
    le = LabelEncoder()
    for i in bin_cols:
        telecom[i] = le.fit_transform(telecom[i])
        telecom_test[i] = le.transform(telecom_test[i])

    # combining the train and test datasets
    trainsize = telecom.shape[0]
    comb = pd.concat((telecom, telecom_test), sort=False)

    # Duplicating columns for multi value columns
    comb = pd.get_dummies(data=comb, columns=multi_cols)

    # Separating the train and test datasets
    telecom = comb[:trainsize]
    telecom_test = comb[trainsize:]

    # Scaling Numerical columns
    std = StandardScaler()
    scaled = std.fit_transform(telecom[num_cols])
    scaled = pd.DataFrame(scaled, columns=num_cols)

    scaled_test = std.transform(telecom_test[num_cols])
    scaled_test = pd.DataFrame(scaled_test, columns=num_cols)

    # dropping original values and merging scaled values for numerical columns
    df_telecom_og = telecom.copy()
    telecom = telecom.drop(columns=num_cols, axis=1)
    telecom = telecom.merge(scaled, left_index=True, right_index=True, how="left")

    df_telecom_test_og = telecom_test.copy()
    telecom_test = telecom_test.drop(columns=num_cols, axis=1)
    telecom_test = telecom_test.merge(scaled_test, left_index=True, right_index=True, how="left")

    data = {
        'un-scaled': {
            'test': df_telecom_test_og,
            'train': df_telecom_og
        },
        'scaled': {
            'test': telecom_test,
            'train': telecom
        }
    }

    return data
