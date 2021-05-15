from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pickle

if __name__ == "__main__":
    digits = datasets.load_digits()

    #Datasets: Prep
    x=digits.data
    y=digits.target

    #scaled
    scaler = StandardScaler()
    scaled = scaler.fit_transform(x)

    #Train and test data
    x_train,x_test,y_train,y_test=train_test_split(scaled,y,test_size=.3)

    #ML Model: Model Selection
    knn=neighbors.KNeighborsClassifier()
    knn.fit(x_train, y_train)

    #Predictions
    predictions=knn.predict(x_test)

    #Accuracy
    print(accuracy_score(y_test,predictions))

    #Registro
    with open('./outputs/digits_model.pkl', 'wb') as model_pkl:
        pickle.dump(knn, model_pkl)