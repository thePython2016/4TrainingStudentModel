import pickle as pkl
import pandas as pd
import streamlit as st

model=pkl.load(open("model.pkl","rb"))
encoder=pkl.load(open("encoder.pkl","rb"))
Transformer=pkl.load(open("Transformer.pkl","rb"))

st.title("Student Performance Model")
file=st.file_uploader("Upload File",type="csv")
button=st.button("Predict Student Outcome")


if button:
    if not file:
        st.error("Upload File")
    else:
        fileData=pd.read_csv(file)
        TransformFile=Transformer.transform(fileData)
        withFeatures=pd.DataFrame(TransformFile,columns=Transformer.get_feature_names_out())
        predict=model.predict(withFeatures)
        predict=encoder.inverse_transform(predict)
        # fileData['pre']
        fileData['Predicted Outcome']=predict
      

        st.write(fileData)