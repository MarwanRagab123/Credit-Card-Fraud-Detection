from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix,precision_recall_curve,auc,average_precision_score
from credit_fraud_utils_data import load_data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import pickle

def modLogestic(x_train, y_train, x_test, y_test):
    model = LogisticRegression(max_iter=500)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_predprob=model.predict_log_proba(x_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cn = confusion_matrix(y_test, y_pred)
    av=average_precision_score(y_test,y_predprob)
    
    print(f"Logistic Regression - Accuracy: {accuracy}")
    print(f"Logistic Regression - F1 Score: {f1}")
    print(f"Logistic Regression - Confusion Matrix:\n{cn}")
    print(f"Logistic Regression - Average Precision:\n{av}")



    precision, recall, thresholds = precision_recall_curve(y_test, y_predprob)
    f1_scores = 2 * (precision * recall) / (precision + recall) 
    best_threshold_index = np.argmax(f1_scores)
    best_threshold = thresholds[best_threshold_index] 
    auc_score = auc(recall, precision)

    print(f"Best Threshold for Logistic Regression: {best_threshold:.2f}")

    #   Precision-Recall Curve
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, label=f"Logistic Regression (AUC = {auc_score:.2f})")
    plt.scatter(recall[best_threshold_index], precision[best_threshold_index], marker='o', color='red', label=f"Best Threshold: {best_threshold:.2f}")
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.show()

    with open('model.pkl', 'wb') as file:
        pickle.dump(model, file)
    print("Model saved as model.pkl")
    
    return accuracy, f1, cn

#Random Forest
# def RandomForest(x_train, y_train, x_test, y_test):
#     model2 = RandomForestClassifier(n_estimators=100,random_state=42)
#     model2.fit(x_train, y_train)
#     y_pred = model2.predict(x_test)
#     y_predprob=model2.predict_log_proba(x_test)[:, 1]
    
#     accuracy = accuracy_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)
#     cn = confusion_matrix(y_test, y_pred)
    
#     print(f"Logistic Regression - Accuracy: {accuracy}")
#     print(f"Logistic Regression - F1 Score: {f1}")
#     print(f"Logistic Regression - Confusion Matrix:\n{cn}")

#     return accuracy, f1, cn




x_train, y_train, x_test, y_test = load_data("split/train.csv", "split/test.csv")



print("Logistic Regression Results:")
modLogestic(x_train, y_train, x_test, y_test)


