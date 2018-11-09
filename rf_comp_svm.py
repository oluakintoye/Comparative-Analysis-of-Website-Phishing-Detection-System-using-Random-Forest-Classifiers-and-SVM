# The key to a fair comparison of machine learning algorithms
#  is ensuring that each algorithm is evaluated in
# the same way on the same data

# forcing each algorithm to be evaluated on a consistent test harness

# Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.svm import SVC


# Load Dataset
path = "website_phish.csv"

names = ["has_ip", "long_url", "short_service", "has_at",
         "double_slash_redirect", "pref_suf", "has_sub_domain",
         "ssl_state", "long_domain", "favicon", "port",
         "https_token", "req_url", "url_of_anchor", "tag_links",
         "SFH", "submit_to_email", "abnormal_url", "redirect",
         "mouseover", "right_click", "popup", "iframe", "domain_Age",
         "dns_record", "traffic", "page_rank", "google_index",
         "links_to_page", "stats_report", "target"]

dataframe = pd.read_csv(path, names=names)
array = dataframe.values

# Slicing the features
X = array[:, 0:30]
Y = array[:, 30]

# Seed ensures in having the same sequence of random numbers
seed = 7

# preparing the models
models = []
models.append(('RF', RandomForestClassifier()))
models.append(('SVM', SVC()))

# evaluating each model
results = []
names = []
scoring = 'accuracy'
scoring1 = 'precision'
scoring2 = 'recall'
scoring3 = 'f1'

for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)

    cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print("Accuracy")
    print(msg)

for name, model in models:
    kfold1 = model_selection.KFold(n_splits=10, random_state=seed)

    cv_results1 = model_selection.cross_val_score(model, X, Y, cv=kfold1, scoring=scoring1)
    results.append(cv_results1)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results1.mean(), cv_results1.std())
    print("Precision")
    print(msg)

for name, model in models:
    kfold2 = model_selection.KFold(n_splits=10, random_state=seed)

    cv_results2 = model_selection.cross_val_score(model, X, Y, cv=kfold2, scoring=scoring2)
    results.append(cv_results2)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results2.mean(), cv_results2.std())
    print("Recall")
    print(msg)


for name, model in models:
    kfold3 = model_selection.KFold(n_splits=10, random_state=seed)

    cv_results3 = model_selection.cross_val_score(model, X, Y, cv=kfold3, scoring=scoring3)
    results.append(cv_results3)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results3.mean(), cv_results3.std())
    print("F1")
    print(msg)

#boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison (Accuracy(A), Precision(P), Recall(R), F1-Score(F))')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# plt.figure()
# plt.plot(fpr, tpr, color='darkorange',
#          lw=2, label='Random Forest(area = %0.2f)' % roc_auc)
# plt.plot(fpr, tpr, color='darkgreen',
#          lw=2, label='Support Vector Machine (area = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve')
# plt.legend(loc="lower right")
# plt.show()
