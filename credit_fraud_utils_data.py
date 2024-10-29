import pandas as pd
from imblearn.over_sampling import SMOTE
def load_data(train_path, test_path):

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    #split_train
    X_train=train_data.drop("Class",axis=1)
    y_train=train_data["Class"]
    #split_test
    X_test=test_data.drop("Class",axis=1)
    y_test=test_data["Class"]
    
    
    #data inblanced in train
    sm=SMOTE(random_state=42)
    X_trainre,y_trainre=sm.fit_resample(X_train,y_train)

    #data inblanced in test
    X_testre,y_testre=sm.fit_resample(X_test,y_test)





    return X_trainre, y_trainre, X_testre,y_testre
    
train="split/train.csv"
test="split/train.csv"
X_train, y_train, X_test,y_test = load_data(train, test)
