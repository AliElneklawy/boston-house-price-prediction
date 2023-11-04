import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.base import TransformerMixin, BaseEstimator
import lightgbm as lgbm
from scipy import stats
import joblib
import streamlit as st

user_input, user_input_df = {}, {}

class BoxCoxTransform(TransformerMixin, BaseEstimator):
    def __init__(self) -> None:
        pass

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_np = np.array(X)
        if X_np.shape[0] == 1:
            return X_np
        
        transformed_data = X_np.copy()
        for column in range(X_np.shape[1]):
            transformed_data[:, column], _ = stats.boxcox(X_np[:, column])
        return transformed_data

box_cox_transformer = BoxCoxTransform()
log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)
std_scaler = StandardScaler()

transformations = ColumnTransformer([
    ('box_cox', box_cox_transformer, ['AGE', 'B']),
    ('log', log_transformer, ['CRIM', 'DIS', 'LSTAT'])
], 
remainder='passthrough')
preprocessing = make_pipeline(transformations, std_scaler)

with open('boston_housing_model.pkl', 'rb') as f:
                loaded_model = joblib.load(f)

placeholder_value = -1.0
st.markdown("<h1>Boston Housing Price Prediction</h1>", unsafe_allow_html=True)
with st.form('form'):
    col1, col2 = st.columns(2)

    CRIM = col1.text_input('CRIM', help='Per Capita Crime Rate')
    ZN = col1.text_input('ZN', help='Proportion of Residential Land Zoned')
    INDUS = col1.text_input('INDUS', help='Proportion of Non-retail Business Acres (1 if tract bounds river; 0 otherwise)')
    NOX = col1.text_input('NOX [parts/10M]', help='nitric oxides concentration')
    RM = col1.text_input('RM', help='average number of rooms per dwelling')
    AGE = col1.text_input('Age', help='proportion of owner-occupied units built prior to 1940')
    DIS = col2.text_input('DIS', help='weighted distances to five Boston employment centres')
    RAD = col2.text_input('RAD', help='index of accessibility to radial highways')
    PTRATIO = col2.text_input('PTRATIO', help='pupil-teacher ratio by town')
    TAX = col2.text_input('TAX [$/10k]', help='full-value property-tax rate')
    B = col2.text_input('B', help='The result of the equation B=1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town')
    LSTAT = col2.text_input('LSTAT', help='% lower status of the population')
    CHAS = st.text_input('CHAS', help='Charles River dummy variable (1 if tract bounds river; 0 otherwise')

    submitted = st.form_submit_button('Submit')
    
    if submitted:
        empty_fields = []
        
        if not CRIM:
            empty_fields.append('CRIM')
        else:
            CRIM = float(CRIM)
        
        if not ZN:
            empty_fields.append('ZN')
        else:
            ZN = float(ZN)
        
        if not INDUS:
            empty_fields.append('INDUS')
        else:
            INDUS = float(INDUS)
        
        if not NOX:
            empty_fields.append('NOX [parts/10M]')
        else:
            NOX = float(NOX)
        
        if not RM:
            empty_fields.append('RM')
        else:
            RM = float(RM)
        
        if not AGE:
            empty_fields.append('Age')
        else:
            AGE = float(AGE)
        
        if not DIS:
            empty_fields.append('DIS')
        else:
            DIS = float(DIS)
        
        if not RAD:
            empty_fields.append('RAD')
        else:
            RAD = float(RAD)
        
        if not PTRATIO:
            empty_fields.append('PTRATIO')
        else:
            PTRATIO = float(PTRATIO)
        
        if not TAX:
            empty_fields.append('TAX [$/10k]')
        else:
            TAX = float(TAX)
        
        if not B:
            empty_fields.append('B')
        else:
            B = float(B)
        
        if not LSTAT:
            empty_fields.append('LSTAT')
        else:
            LSTAT = float(LSTAT)
        
        if not CHAS:
            empty_fields.append('CHAS')
        else:
            CHAS = float(CHAS)

        if empty_fields:
            st.error(f'The following fields are empty: {", ".join(empty_fields)}')

        else:
            user_input['CRIM'] = CRIM
            user_input['ZN'] = ZN
            user_input['INDUS'] = INDUS
            user_input['CHAS'] = CHAS
            user_input['NOX'] = NOX
            user_input['RM'] = RM
            user_input['AGE'] = AGE
            user_input['DIS'] = DIS
            user_input['RAD'] = RAD
            user_input['TAX'] = TAX
            user_input['PTRATIO'] = PTRATIO
            user_input['B'] = B
            user_input['LSTAT'] = LSTAT
            user_input_df = pd.DataFrame([user_input])

            st.success('Processing...')
            prediction = loaded_model.predict(user_input_df)[0].round(2)
            st.write(f"Median value of owner-occupied homes in \$1000's [k$]: {prediction}")
            
