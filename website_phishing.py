__author__ = 'Olumide'

# Python Packages Required

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

# File Paths

PATH = "inputs/website_phishing.csv"


# Headers
HEADERS = ["has_ip", "long_url", "short_service", "has_at",
           "double_slash_redirect", "pref_suf", "has_sub_domain",
           "ssl_state", "long_domain", "favicon", "port",
           "https_token", "req_url", "url_of_anchor", "tag_links",
           "SFH", "submit_to_email", "abnormal_url", "redirect",
           "mouseover", "right_click", "popup", "iframe", "domain_Age",
           "dns_record", "traffic", "page_rank", "google_index",
           "links_to_page", "stats_report", "target"]


# Loading the csv file into a pandas dataframe

def reading_data(path):
    data = pd.read_csv(path)
    return data


# Adding the headers to a loaded dataframe
def adding_headers(dataset, headers):
    dataset.columns = headers
    return dataset

# Getting the dataset headers
def getting_headers(dataset):
    return dataset.columns.values


# Split dataset into train and test dataset
def dataset_split(dataset, train_percentage, feature_headers, target_header):

    train_x, test_x, train_y,test_y = train_test_split(dataset[feature_headers], dataset[target_header],
                                                        train_size=train_percentage)

    return train_x, test_x, train_y, test_y


# To train the random forest classifier with features and target data
def RandomForest_Classifier(features, target):

    clf = RandomForestClassifier()
    clf.fit(features, target)
    return clf


# Analyzing some of the statistics of the dataset
def dataset_statistics(dataset):
    print(dataset.describe())


def main():
    # Load the csv file into pandas dataframe
    dataset = pd.read_csv(PATH)

    # Get basic statistics of the loaded dataset
    dataset_statistics(dataset)

    train_x, test_x, train_y, test_y = dataset_split(dataset, 0.80, HEADERS[0:-1], HEADERS[-1])

    # Train and Test dataset size details
    print("Train_x  :: ", len(train_x))
    print("Train_y  :: ", len(train_y))
    print("Test_x   :: ", len(test_x))
    print("Test_y   :: ", len(test_y))

    print(test_x)
    print(test_y)

    # Creating a random forest classifier instance
    trained_model = RandomForest_Classifier(train_x, train_y)
    print("Trained model : ", trained_model)
    predictions = trained_model.predict(test_x)

    for x in range(0, len(test_x)):
        print("The Actual outcome : {} and Predicted outcome : {}".format(list(test_y)[x], predictions[x]))

    print("The Train Accuracy : ", accuracy_score(train_y, trained_model.predict(train_x)))
    print("The Test Accuracy  : ", accuracy_score(test_y, predictions))
    print(" Confusion matrix ", confusion_matrix(test_y, predictions))
    print(classification_report(test_y, predictions))


if __name__ == "__main__":
    main()
