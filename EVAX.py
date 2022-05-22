import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from objects import Literal, Argument, AF
from methods import *
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning
import time
import statistics

# the different datasets and classifiers
datasets = ['mushroom', 'adult bin', 'iris', 'wine']
clfs = ['Logistic Regression', 'SVM', 'Random Forest', 'Neural Network']


def run_all():
    """This method runs EVAX for all datasets and classifiers"""
    begin_time = time.time()

    avg_fidelity = []
    for dataset_name in datasets:
        df = get_dataset(dataset_name)
        for clf_name in clfs:
            clf = get_fitted_clf(df, clf_name)
            start_time = time.time()
            fidelity = run_EVAX(clf=clf, df=df, normalize=False, test_size=0.2,
                                explanation=False, explained_instance=3, t_select=20,
                                t_explain=2, feature_importance=False, divide_by_class_distribution=False,
                                biasedness=False)
            avg_fidelity.append(fidelity)
            print(dataset_name, clf_name)
            print("--- %s seconds ---" % (time.time() - start_time), '\n\n' )

    print('\n############################ Summary ############################')
    print('After running EVAX on all datasets and classifiers:')
    print('Average fidelity score:', statistics.mean(avg_fidelity))
    print("Total run time: %s seconds" % round((time.time() - begin_time), 2))



def run_once(dataset_name, clf_name):
    """This method runs EVAX for one dataset and one classifier"""
    avg_fidelity = []
    df = get_dataset(dataset_name)
    clf = get_fitted_clf(df, clf_name)
    start_time = time.time()

    """ t_select determines the number of argumentens in the argumentation framework of EVAX
    set explanation=True and explained_instance=i to get an explanation of the i-th instance of the dataset
    t_explain determines the size of the dialectical explanation
    when biasedness is True, a biased explanation (based on abnormality) is added
    when divide_by_class_distribution is True, the argument strength is altered by class distributions
    when normalize is True, the argument strength is normalized between 0 and 1."""

    fidelity = run_EVAX(clf=clf, df=df,
                        t_select=20,
                        explanation=True, explained_instance=0, t_explain=2, biasedness=True,
                        divide_by_class_distribution=False,
                        normalize=True)

    avg_fidelity.append(fidelity)
    print('############################ Summary ############################')
    print('After running EVAX on the', dataset_name,'dataset and the', clf_name, 'classifier:')
    print('Fidelity score:', statistics.mean(avg_fidelity))
    print("Total run time: %s seconds" % round((time.time() - start_time), 2))


def main():

    #run_once('adult bin', 'Random Forest')

    run_all()


if __name__ == '__main__':
    main()



