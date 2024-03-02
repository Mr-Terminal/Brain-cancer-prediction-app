import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle as pickle


def create_model(data):

    # separating the attributes of datset into target variable and predictor variables
    x = data.drop('diagnosis', axis=1)
    y = data['diagnosis']

    # We need to scale the predictor variables as they are too much vary from each other
    scaler = StandardScaler()
    X = scaler.fit_transform(x)

    # split train test data --> 80% train data and 20% test data
    x_train, x_test, y_train, y_test =  train_test_split(X, y, test_size=0.2, random_state=42)

    # train model
    model = LogisticRegression()
    model.fit(x_train, y_train)

    #test model
    y_predict = model.predict(x_test)

    print("Accuracy of the model: ", accuracy_score(y_test, y_predict))
    print(classification_report(y_test, y_predict))

    return model, scaler


def clean_data():
    data = pd.read_csv('data/data.csv')
    
    #Remove the id and Unnamed column as those are unnecessary
    data = data.drop(['Unnamed: 32', 'id'], axis = 1) 
    
    # Map the diagnosis target variable to a binary classification format i.e 0 and 1
    data['diagnosis'] = data['diagnosis'].map({
        'M' : 1,
        'B' : 0
    })
    
    return data


def main():
    data = clean_data()

    model, scaler = create_model(data)

    print(model)
    print(scaler)
    with open('model/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)



if __name__ == '__main__':
    main()    