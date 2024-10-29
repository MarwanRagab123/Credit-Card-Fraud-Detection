import matplotlib.pyplot as plt
import seaborn as sns
from credit_fraud_utils_data import load_data

def eda(data, dataset_name="Dataset"):
   
    print(f"\n--- EDA for {dataset_name} ---")
    
    
    print("First 5 rows of the dataset:")
    print(data.head())
    
  
    print("\nData Info:")
    print(data.info())
    
   
    print("\nDescriptive Statistics:")
    print(data.describe())
    
  
    print("\nMissing Values:")
    print(data.isnull().sum())
    
   
    if 'Class' in data.coulmns:
        plt.figure(figsize=(6, 4))
        sns.countplot(data['target_column'])
        plt.title(f"Target Variable Distribution in {dataset_name}")
        plt.show()

    
   
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(f"Correlation Matrix for {dataset_name}")
    plt.show()


X_train, y_train, X_test,y_test = load_data("split/train.csv","split/test.csv")
    
    
eda(X_train, "X train")
eda(y_train, "y train")

eda(X_test, "X_test")
eda(y_test, "y_test")