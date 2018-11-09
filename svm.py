__author__ = 'Olumide'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

path = "inputs/website_phishing.csv"

# Assign colum names to the dataset
colnames = ["has_ip", "long_url", "short_service", "has_at",
            "double_slash_redirect", "pref_suf", "has_sub_domain",
            "ssl_state", "long_domain", "favicon", "port",
            "https_token", "req_url", "url_of_anchor", "tag_links",
            "SFH", "submit_to_email", "abnormal_url", "redirect",
            "mouseover", "right_click", "popup", "iframe", "domain_Age",
            "dns_record", "traffic", "page_rank", "google_index",
            "links_to_page", "stats_report", "target"]

def main():
# Read dataset to pandas dataframe
    data = pd.read_csv(path)

    data.shape

    data.head()

    X = data.drop('target', axis=1)
    y = data['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

    svclassifier = SVC(kernel='linear')
    svclassifier.fit(X_train, y_train)

    y_pred = svclassifier.predict(X_test)

    for x in range(0, len(X_test)):
        print("The Actual outcome : {} and Predicted outcome : {}".format(list(y_test)[x], y_pred[x]))

    print("The Train Accuracy : ", accuracy_score(y_train, svclassifier.predict(X_train)))
    print("The Test Accuracy  : ", accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))
