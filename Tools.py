import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn import preprocessing


def data_clean(data):
    gender = data['gender']
    re_gender = []
    for i in gender:
        if i == "Female":
            re_gender.append(0)
        elif i == "Male":
            re_gender.append(1)
        else:
            re_gender.append(2)
    data['gender'] = re_gender

    # 'smoking_history' -- 'No Info' into np.nan
    sh = data['smoking_history']
    re_sh = []
    for i in sh:
        if i == "never":
            re_sh.append(0)
        elif i == "former":
            re_sh.append(1)
        elif i == "ever":
            re_sh.append(2)
        elif i == "not current":
            re_sh.append(3)
        elif i == "current":
            re_sh.append(4)
        elif i == "No Info":
            re_sh.append(np.nan)
    data['smoking_history'] = re_sh

    # knn 补全
    td = data.drop('diabetes', axis=1)
    imputer = KNNImputer(n_neighbors=11)
    re_td = imputer.fit_transform(td)
    re_td = pd.DataFrame(re_td)

    # print(re_td)
    #
    # # Normalization
    # normalizer = preprocessing.Normalizer().fit(re_td)
    # nre_td = normalizer.transform(re_td)

    # return re_td.astype(float), data['diabetes'].astype(float)
    return re_td, data['diabetes']
