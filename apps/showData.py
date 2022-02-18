import glob

import streamlit as st
import pandas as pd
import os
import csv
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectPercentile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import VotingClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression


def data_analysis():
    # !/usr/bin/env python
    # coding: utf-8

    # # Data Analysis
    #
    # #### by Davide Gamba, Andrea Wey and Christian Pala
    #

    # ### Libraries:

    # In[4]:
    st.write("func")

    plt.style.use("dark_background")


    # Load datasets

    # In[ ]:

    df = pd.read_csv("/Users/corradopansoni/Desktop/Hackaton/apps/features.csv")

    # In[6]:

    targets = pd.read_csv("/Users/corradopansoni/Desktop/Hackaton/apps/NSCLC_target.csv")
    targets

    # ## Exploratory Data Analysis:

    # In[7]:

    df.describe()

    # We can drop the index, as it was a helper column for the feature extraction phase.

    # In[8]:

    df.drop(["index"], axis=1, inplace=True)

    # We are not considering age and gender as target features, as this should be information available
    # in medical records, we therefore merge them with the rest of our extracted features.

    # In[9]:

    df = targets[["age", "gender"]].join(df)
    df

    # In[10]:

    targets = targets.drop(["PatientID", "age", "gender"], axis=1)

    # We have 422 patients, we are interested in building mathematical models to predict:
    #
    # - Whether the patient is alive or dead, binary classification problem
    # - The life expectancy of the patient, regression problem
    # - Determining the histology of the tumor(s) in the patient, multivariate classification
    # problem
    #
    # The first step is understanding which features we are working with, based on the radiomics documentation and
    # *Radiomics: the facts and the challenges of image analysis* by Rizzo & all: these can be broadly divided into 4
    # categories, in our application case:
    #
    # - first order statistical features
    # - shape features
    # - higher order statistical features
    # - known medical information, we include age and gender in this group.

    # ### First order features:

    # In[11]:

    first_order_features = [col for col in df.columns if 'firstorder' in col]

    for i, col in enumerate(first_order_features):
        plt.figure(i)
        sns.histplot(df[col])

    # Citing *Radiomics: the facts and the challenges of image analysis* by Rizzo & all:
    # First-order statistics features describe the distribution of individual voxel values without concern
    # for spatial relationships. These are histogram-based properties reporting the mean,
    # median, maximum, minimum values of the voxel intensities on the image,
    # as well as their skewness (asymmetry), kurtosis (flatness), uniformity, and randomness (entropy).
    #
    # Our interpretation is these are sets of mathematically relevant features for the single voxel our models
    # may be able to use, given our knowledge base as non-specialists the best approach is using performance based
    # feature selection to determine what we should keep and what not.

    # ### Shape features
    # Citing *Radiomics: the facts and the challenges of image analysis* by Rizzo & all:
    # describe the shape of the traced region of interest (ROI) and its geometric properties such as volume, maximum diameter
    # along different orthogonal directions, maximum surface, tumour compactness, and sphericity.
    # For example, the surface-to-volume ratio of a spiculated tumour will show higher values than that of a round
    # tumour of similar volume.
    #
    # From the description we expect these features to play a strong role when trying to predict tumor histology.

    # ### Higher order features
    # These are features capturing the interaction between voxels, in particular we expect the GLCM matrix
    # indicators to be helpful in detecting anomalies and to be of use to algorithms using
    # boundary regions for their predictions.
    #
    # Our general approach as new to the field is we would rather keep more variables in our hypothesis and do
    # performance based feature selections, rather than mistakenly excluding something important.

    # ## deadstatus.event
    # The first event we will try to model is predicting if the patient is alive or dead given the
    # tomography, it's a good baseline case to start on.
    #
    # Let's begin with the missing values.

    # In[12]:

    df.isnull().sum()

    # We have 22 missing values for age.

    # In[13]:

    sns.histplot(df["age"])

    # As expected for this dataset age is slightly skewed, we prefer imputing with
    # the median value in this case since it's more robust against outliers,
    # dropping with such a small dataset would be an unnecessary loss of information.

    # In[14]:

    df["age"] = df["age"].fillna(df["age"].median())
    df.isnull().sum()

    # In[15]:

    dead_or_alive = targets["deadstatus.event"]
    dead_or_alive.value_counts()

    # we have a pretty unbalanced dataset, we need to check how the model is learning
    # for each class, aggregate measures like accuracy could be misleading.

    # Gender needs to by encoded:

    # In[16]:

    lb_make = LabelEncoder()
    df["gender"] = lb_make.fit_transform(df["gender"])
    df["gender"].value_counts()

    # The dataset is also slightly unbalanced regarding males and females.

    # # Features Selection

    # Since we kept all the features we could extract, this is a very important step to optimize our models.
    #
    # ## Anova

    # In[17]:

    def anova(X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=.3)

        select = SelectPercentile(percentile=50)
        select.fit(X_train, y_train)

        # transform train set
        X_train_selected_anova = select.transform(X_train)
        X_test_selected_anova = select.transform(X_test)

        return X_train_selected_anova, X_test_selected_anova

    # ## K-Best

    # In[18]:

    def k_best(X, y):
        min_max_df = MinMaxScaler().fit_transform(X)
        X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(min_max_df, y, random_state=0, test_size=.3)

        k_best = SelectKBest(k=40)

        fit = k_best.fit(X_train_scaled, y_train)

        # transform training set
        X_train_selected_k_best = k_best.transform(X_train_scaled)
        X_test_selected_k_best = k_best.transform(X_test_scaled)

        return X_train_selected_k_best, X_test_selected_k_best

    # # Models

    # #Binary Classification

    # In[21]:

    X_train, X_test, y_train, y_test = train_test_split(df, targets['deadstatus.event'], random_state=0, test_size=.3)
    X_train_anova, X_test_anova = anova(df, targets['deadstatus.event'])
    X_train_k_best, X_test_k_best = k_best(df, targets['deadstatus.event'])

    # ## Decision Tree

    # In[19]:

    clf = tree.DecisionTreeClassifier()

    # In[22]:

    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # In[33]:

    print(classification_report(y_test, y_pred))

    # #### Anova

    # In[23]:

    clf = clf.fit(X_train_anova, y_train)
    y_pred = clf.predict(X_test_anova)
    print(classification_report(y_test, y_pred))

    # #### K-Best

    # In[24]:

    clf = clf.fit(X_train_k_best, y_train)
    y_pred = clf.predict(X_test_k_best)
    print(classification_report(y_test, y_pred))

    # ## Logistic Regression

    # In[26]:

    log = LogisticRegression(max_iter=100000000)

    # In[27]:

    log.fit(X_train, y_train)
    y_pred = log.predict(X_test)
    print(classification_report(y_test, y_pred, zero_division=False))

    # #### Anova

    # In[28]:

    log.fit(X_train_anova, y_train)
    y_pred = log.predict(X_test_anova)
    print(classification_report(y_test, y_pred, zero_division=False))

    # #### K-Best

    # In[29]:

    log.fit(X_train_k_best, y_train)
    y_pred = log.predict(X_test_k_best)
    print(classification_report(y_test, y_pred, zero_division=False))

    # ## Ensemble

    # In[30]:

    ensemble = VotingClassifier([("tree", clf), ("log", log)])

    ensemble.fit(X_train, y_train)
    y_pred = ensemble.predict(X_test)

    print(classification_report(y_test, y_pred, zero_division=False))

    # ### Anova

    # In[31]:

    ensemble.fit(X_train_anova, y_train)
    y_pred = ensemble.predict(X_test_anova)
    print(classification_report(y_test, y_pred, zero_division=False))

    # ### K-Best

    # In[32]:

    ensemble.fit(X_train_k_best, y_train)
    y_pred = ensemble.predict(X_test_k_best)
    print(classification_report(y_test, y_pred, zero_division=False))

    # #Regression

    # ##Linear Regression

    # In[50]:

    y = targets['Survival.time']

    X_train, X_test, y_train, y_test = train_test_split(df, y, random_state=0, test_size=.3)
    X_train_k_best, X_test_k_best = k_best(df, y)
    X_train_anova, X_test_anova = anova(df, y)

    # In[42]:

    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error

    lin = LinearRegression().fit(X_train, y_train)
    y_pred = lin.predict(X_test)

    print(mean_squared_error(y_test, y_pred))

    # ###Anova

    # In[40]:

    lin = LinearRegression().fit(X_train_anova, y_train)
    y_pred = lin.predict(X_test_anova)

    print(mean_squared_error(y_test, y_pred))

    # In[47]:

    lin = LinearRegression().fit(X_train_k_best, y_train)
    y_pred = lin.predict(X_test_k_best)

    print(mean_squared_error(y_test, y_pred))


def csv_path():
    # Write filenames from folder in csv
    path = '10_patients/*'
    with open('folder_names.csv', 'w') as f:
        writer = csv.writer(f)
        a = glob.glob(os.path.join("*/", "*", ""))
        writer.writerows(zip(a))


def app():
    st.title('Show data')
    csv_path()
    df = pd.read_csv("folder_names.csv")

    with st.form("my_form"):
        st.write("Choose a patient:")
        patient_choose = st.selectbox("Choose a patient", df)
        submit_button = st.form_submit_button(label='Submit')

        if submit_button:
            st.write(f'your choose:  {patient_choose}')
    if submit_button:

        execute_button = st.button(f'execute feature extraction for {patient_choose}')

        prediction_button = st.button(f'prediction model for {patient_choose}')
        view_button = st.button(f'view patient scans for {patient_choose}')

