import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
csv = pd.read_csv('data_aida_header.csv')
dataframe = pd.DataFrame(csv)
x=dataframe.copy()
y=x.pop('class')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
classes = ['Helm Scoopy','Helm CM','Helm VOG', 'Helm Bogo', 'Helm INK', 'Helm Merah']
n_classes = len(classes)
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
clf = model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# Use score method to get accuracy of the model
score_te = model.score(X_test, y_test)
print('Accuracy Score: ', score_te)
from sklearn.preprocessing import label_binarize
# Binarize the output
y_test = label_binarize(y_test, classes=classes)
from matplotlib import pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc
plt.style.use('ggplot')
# Plotting and estimation of FPR, TPR
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
colors = cycle(['blue', 'red', 'green', 'yellow'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=1.5, label='ROC curve of class {0} (area = {1:0.2f})' ''.format(classes[i], roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k-', lw=1.5)
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for multi-class data')
plt.legend(loc="lower right")
plt.show()
