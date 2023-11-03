import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.base import TransformerMixin, BaseEstimator
import lightgbm as lgbm
from scipy import stats
import joblib

class BoxCoxTransform(TransformerMixin, BaseEstimator):
    def __init__(self) -> None:
        pass

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_np = np.array(X)
        if X_np.shape[0] == 1:
            return X_np  # No transformation needed for a single point
        
        transformed_data = X_np.copy()
        for column in range(X_np.shape[1]):
            transformed_data[:, column], _ = stats.boxcox(X_np[:, column])
        return transformed_data

""" with open('boston_housing_model.pkl', 'rb') as f:
    loaded_model = joblib.load(f) """

box_cox_transformer = BoxCoxTransform()
log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)
std_scaler = StandardScaler()

transformations = ColumnTransformer([
    ('box_cox', box_cox_transformer, ['AGE', 'B']),
    ('log', log_transformer, ['CRIM', 'DIS', 'LSTAT'])
], 
remainder='passthrough')
preprocessing = make_pipeline(transformations, std_scaler)


def get_input():

    user_input = {}
    feat1 = float(input())
    feat2 = float(input())
    feat3 = float(input())
    feat4 = float(input())
    feat5 = float(input())
    feat6 = float(input())
    feat7 = float(input())
    feat8 = float(input())
    feat9 = float(input())
    feat10 = float(input())
    feat11 = float(input())
    feat12 = float(input())
    feat13 = float(input())


    user_input['CRIM'] = feat1
    user_input['ZN'] = feat2
    user_input['INDUS'] = feat3
    user_input['CHAS'] = feat4
    user_input['NOX'] = feat5
    user_input['RM'] = feat6
    user_input['AGE'] = feat7
    user_input['DIS'] = feat8
    user_input['RAD'] = feat9
    user_input['TAX'] = feat10
    user_input['PTRATIO'] = feat11
    user_input['B'] = feat12
    user_input['LSTAT'] = feat13

    user_input_df = pd.DataFrame([user_input])
    return user_input_df


if __name__ == '__main__':
    
    with open('boston_housing_model.pkl', 'rb') as f:
        loaded_model = joblib.load(f)
    input = get_input()
    print(f"{loaded_model.predict(input)[0].round(2)}")
