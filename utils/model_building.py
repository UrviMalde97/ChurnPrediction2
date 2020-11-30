from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import roc_auc_score
import pickle


def telecom_churn_prediction(algorithm, training_x, testing_x, training_y, testing_y):
    # model
    algorithm.fit(training_x, training_y)
    predictions = algorithm.predict(testing_x)

    print('Algorithm:', type(algorithm).__name__)
    print("\nClassification report:\n", classification_report(testing_y, predictions))
    print("Accuracy Score:", accuracy_score(testing_y, predictions))

    # roc_auc_score
    model_roc_auc = roc_auc_score(testing_y, predictions)
    print("Area under curve:", model_roc_auc, "\n")

    pickle.dump(algorithm, open('pickle/'+type(algorithm).__name__ + '.pkl', 'wb'))




