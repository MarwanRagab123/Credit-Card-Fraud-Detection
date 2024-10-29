import pickle
import pandas as pd
from sklearn.metrics import accuracy_score

with open('model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

def evaluateModel(test_path):
    test_val=pd.read_csv(test_path)
    X_test=test_val.drop("Class",axis=1)
    y_test=test_val["Class"]
    y_pred=loaded_model.predict(X_test)
    acc=accuracy_score(y_test,y_pred)
    print("Accuracy va =", acc)
    return acc
evaluateModel("split/val.csv")



    


