""" Cluster Heart Disease Data
Author: Tyler Baker
Class:  DAT-310-01
Certification of Authenticity:
I certify that this is entirely my own work, except where I have given fully documented
references to the work of others. I understand the definition and consequences of
plagiarism and acknowledge that the assessor of this assignment may, for the purpose of
assessing this assignment reproduce this assignment and provide a copy to another member
of academic staff and / or communicate a copy of this assignment to a plagiarism checking
service(which may then retain a copy of this assignment on its database for the purpose
of future plagiarism checking).
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
import random


# Data Centering
def visualize_distributions_for_skew(data):
    """ Create Graphs displaying distributions before and after centering the data."""
    for col in data.columns:
        sns_plot = sns.distplot(data.loc[:, col], bins=20)
        fig = sns_plot.get_figure()
        fig.savefig('batch_figures/' + col + ".png")
        fig.clear()

        frequency_log = np.log(data.loc[:, col] + 1)
        log_plot = sns.distplot(frequency_log)
        log_fig = log_plot.get_figure()
        log_fig.savefig('batch_figures/log' + col + ".png")
        plt.show()


def centering(data):
    """ Center the data using logarithms. """
    return data.apply(np.log)


# Standardization
def standardization(data):
    """ Standardize the data using StandardScalar."""
    scaler = StandardScaler()
    scaler.fit(data)
    return pd.DataFrame(scaler.transform(data))


# Decorrelation
def correlation_map(data):
    """ Display the correlation of the data on a heatmap."""
    sns.heatmap(data.corr(), annot=True)


def decorrelation(data):
    """ Decorrelate the data using PCA. """
    pca = PCA()
    pca.fit(data)
    return pd.DataFrame(pca.transform(data))


# Feature Reduction
def feature_variance_check(data):
    """ Check the variance for each PCA feature of the data. """
    pca = PCA()
    pca.fit(data)
    features = range(pca.n_components_)
    plt.bar(features, pca.explained_variance_)
    plt.xlabel('PCA feature')
    plt.ylabel('variance')
    plt.xticks(features)
    plt.show()


def feature_reduction(data, num_com=2):
    """ Reduce the features to the number given, default of 2. """
    pca = PCA(n_components=num_com)
    pca.fit(data)
    pca_features = pca.transform(data)
    print(pca_features.shape)
    return pd.DataFrame(pca_features)


# Finding the optimal number of clusters
def optimal_clusters(data):
    """ Build an elbow graph for the given data. """
    ks = range(1, 8)
    inertia = []

    for k in ks:
        model = KMeans(n_clusters=k)
        model.fit(data)
        inertia.append(model.inertia_)

    # Plot ks vs inertia
    plt.plot(ks, inertia, '-o')
    plt.xlabel('number of clusters, k')
    plt.ylabel('inertia')
    plt.xticks(ks)
    plt.show()


if __name__ == '__main__':
    heart_disease = pd.read_csv('heart_disease_patients.csv')
    hdp = pd.DataFrame(heart_disease)
    print(hdp.head())

    # Drop unneeded columns and null values
    hdp.dropna()
    hdp = hdp.drop(['id', 'cp', 'restecg', 'slope'], axis=1)

    # show current distribution for reference
    hdp.hist()
    plt.show()

    # standardize dataframe and show distribution for reference
    scaled = standardization(hdp)
    scaled.hist()
    plt.show()

    # Display correlation map to determine if there is correlation
    correlation_map(scaled)
    # decorrelate the data anyways
    deco = decorrelation(scaled)

    # restandardize the data and show distribution for reference
    rescaled = standardization(deco)
    rescaled.hist()
    plt.show()

    # check if feature reduction is useful (it is not)
    feature_variance_check(rescaled)

    # set seed and display elbow graph to see what K should be
    random.seed(10)
    optimal_clusters(rescaled)

    # user inputs what K should be
    k = int(input("What is the optimal K?"))
    inertias = []

    # Run through multiple seeds and pick the one with the KMeans model with the least inertia
    for i in range(20):
        random.seed(i)
        kmeans = KMeans(n_clusters=k)
        model = kmeans.fit(rescaled)
        inertias.append(model.inertia_)
    random.seed(inertias.index(min(inertias)))

    # Cluster the data with the seed and K values and sort dataframe by those clusters
    kmeans = KMeans(n_clusters=k)
    model = kmeans.fit(rescaled)
    hdp['Cluster'] = model.predict(rescaled)

    # Display cluster means and std
    pd.options.display.width = 0
    print(hdp.groupby(['Cluster']).mean())
    print(hdp.groupby(['Cluster']).std())
