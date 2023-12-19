import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import shap

data = 'data_all.csv'

df = pd.read_csv(data).fillna(0)

# declare data and target values
X = df.drop(['winning_team'], axis=1)
y = df['winning_team']

# split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# get columns and 
cols = X_train.columns
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train = pd.DataFrame(X_train, columns=[cols])
X_test = pd.DataFrame(X_test, columns=[cols])
X_train.describe()

# instantiate classifier with default hyperparameters
try:# load
    with open('model.pkl', 'rb') as f:
        svc = pickle.load(f)
except FileNotFoundError:
    svc=SVC(probability=True)

    # fit classifier to training set
    svc.fit(X_train,y_train)

    # save the trained model
    with open('model.pkl','wb') as f:
        pickle.dump(svc,f)

# make predictions on test set
y_pred=svc.predict(X_test)

# compute and print accuracy score
print('Model accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

# print the scores on training and test set
print('Training set score: {:.4f}'.format(svc.score(X_train, y_train)))
print('Test set score: {:.4f}'.format(svc.score(X_test, y_test)))

# print confusion matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix\n\n', cm)
print('\nTrue Positives(TP) = ', cm[0,0])
print('\nTrue Negatives(TN) = ', cm[1,1])
print('\nFalse Positives(FP) = ', cm[0,1])
print('\nFalse Negatives(FN) = ', cm[1,0])
plt.figure(1)
cm_matrix = pd.DataFrame(data=cm, columns=['Actual Blue Side', 'Actual Red Side'], 
                                 index=['Predict Blue Side', 'Predict Red Side'])
sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')

# classification metrics
print(classification_report(y_test, y_pred))
TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]
classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)
print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))
classification_error = (FP + FN) / float(TP + TN + FP + FN)
print('Classification error : {0:0.4f}'.format(classification_error))
precision = TP / float(TP + FP)
print('Precision : {0:0.4f}'.format(precision))
recall = TP / float(TP + FN)
print('Recall or Sensitivity : {0:0.4f}'.format(recall))
specificity = TN / (TN + FP)
print('Specificity : {0:0.4f}'.format(specificity))

# use Kernel SHAP to explain test set predictions
plt.figure(2)
explainer = shap.KernelExplainer(svc.predict_proba, shap.sample(X_train,120), link="logit")
shap_values = explainer.shap_values(shap.sample(X_test,30), nsamples=100)
shap.summary_plot(shap_values, features=X_test, feature_names=cols, class_names=["Blue Win", "Red Win"])

print("Done")