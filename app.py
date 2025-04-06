import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import joblib
import streamlit as st

#data = pd.read_csv('C:\\Users\\leona\\OneDrive\\Área de Trabalho\\Leo\\Python\\diabetes\\diabetes2.csv')

#X = data.drop('Outcome', axis=1)
#y = data['Outcome']

#X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=50)

#X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=50)

#model = joblib.load('notebooks\\model.pkl')

st.title('Diabetes classification')

with st.sidebar:
    st.header('Data information')
    st.caption('This model was trained using the diabetes dataset found in [here](https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset)')
    st.caption('For exploratory data analysis and to verify how this model was trained, check this [jupyter notebook](https://github.com/2leog/diabetes-data-analysis/blob/master/diabetes_data_analysis.ipynb)')
    st.caption('For more general information about this project check the [github repo](https://github.com/2leog/diabetes-data-analysis)')
    st.header('Data requirements')
    st.caption('To inference the model you need to upload a dataframe in csv format with eight columns/features.')
    with st.expander('Data format'):
        st.markdown(' - utf-8')
        st.markdown(' - separated by coma')
        st.markdown(' - delimited by "."')
        st.markdown(' - first row - header')
    st.caption('You can download a test set in [here](https://github.com/2leog/diabetes-data-analysis/blob/master/X_test.csv)')
    st.divider()
    st.caption('Developed by Leonardo Guimarães de Oliveira.')

if 'clicked' not in st.session_state:
    st.session_state.clicked = {1:False}

def clicked(button):
    st.session_state.clicked[button] = True

st.button("Let's get started", on_click=clicked, args=[1])

if st.session_state.clicked[1]:
    uploaded_file = st.file_uploader('Choose a file', type='csv')
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, low_memory=True)
        st.header('Uploaded data sample')
        st.write(df.head())
        model = joblib.load(filename='model.pkl')
        pred_proba = model.predict_proba(df)
        pred_proba = pd.DataFrame(pred_proba, columns = ['non_diabetes_probability', 'diabetes_probability'])
        pred_values = model.predict(df)
        pred_values = pd.DataFrame(pred_values, columns=['is_diabetes_pred'])
        pred = dict(
            {
                'non_diabetes_probability': pred_proba['non_diabetes_probability'], 
                'diabetes_probability': pred_proba['diabetes_probability'], 
                'is_diabetes_predicted': pred_values['is_diabetes_pred']
            }
        )
        pred = pd.DataFrame(data=pred)
        st.header('Predictions')
        st.write(pred.head())
        pred = pred.to_csv(index=False).encode('utf-8')
        st.download_button(
            label='Download predictions',
            data=pred,
            file_name='predictions.csv',
            mime='text/csv',
            key='download-csv'
        )
        
        

