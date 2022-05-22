from objects import Literal, Argument, AF, Labeler
from statistics import mode
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import sklearn
import numpy as np
from sklearn.inspection import permutation_importance


def run_EVAX(df, clf, normalize=False, test_size=0.2, explanation=False, explained_instance=1,
             t_select=None, t_explain=None, feature_importance=False,
             divide_by_class_distribution=False, biasedness=False):
    """
    The method run_EVAX is the high-level method that calls the sub-methods below.
    It takes as input a dataset (df) and a classifier (clf).
    First, it splits the df into a train set and a test set.
    Then it computes a list of all attribute-value pairs (or literals).
    Afterward, it creates arguments based on these literals.
    Then, for every datapoint in the test set it creates an argumentation framework (AF), called the local_AF.
    Based on that local_AF, it predicts the output class of that datapoint.
    Lastly, it creates an explanation for that prediction.
    """

    # create global variable that counts how many times the grounded extension is empty
    global no_ge
    no_ge = 0

    # get feature names, separate labels (y) from feature-value pairs (X) and split X and y into train set and test set
    feature_names, X, y, n_of_classes = get_variables_from_df(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # set the default prediction as the majority class
    default_prediction = y.value_counts().idxmax()

    # get predictions from black box
    bb_pred_test = clf.predict(X_test)

    # calculate feature permutation importance scores
    if feature_importance:
        r = permutation_importance(clf, X_train, y_train, n_repeats=2, random_state=42)
        print('calculated feature importance...')
    else:
        r = None

    # get a list of all attribute-value pairs (= literals)
    list_of_literals = get_list_of_literals(clf, feature_names, X_train)

    # convert all literals into arguments
    list_of_arguments = convert_literals_to_arguments(list_of_literals=list_of_literals,
                                                      n_of_classes=n_of_classes, normalize=normalize, y_train=y_train,
                                                      divide_by_class_distribution=divide_by_class_distribution,
                                                      len_dataset=len(X_train))
    print('computed arguments...')

    if normalize:
        # convert argument strength into values between 0 and 1
        list_of_arguments = normalize_probability_of_arguments(list_of_arguments)

    af_pred = []
    test_length = len(X_test)

    # for all instances in the test set
    for row_number in range(test_length):
        test_instance = X_test.iloc[row_number:row_number + 1, :].to_numpy().reshape(1, -1)

        # get relevant arguments
        relevant_arguments = get_relevant_arguments(list_of_arguments, test_instance, feature_names, size=t_select)

        # delete all attack relations (otherwise they stack up when iterating)
        for argument in relevant_arguments:
            del argument.attacks[:]
            del argument.is_attacked_by[:]

        # alter argument strength based on permutation feature importance scores
        if feature_importance:
            change_argument_strength(relevant_arguments, permutation_importance=r)

        # define attack relations
        define_attack_relations(relevant_arguments, feature_importance=feature_importance)

        # create the local_AF based on the relevant arguments.
        af = create_AF(relevant_arguments)

        # get the prediction of the AF
        ge_list, prediction_af, predictionstotal = get_single_prediction_AF(af, default_prediction=default_prediction)
        af_pred.append(prediction_af)

    fidelity = get_fidelity(af_pred, bb_pred_test)
    print_results(af_pred, bb_pred_test, fidelity, no_ge, test_length, y_test)
    # path = r'C:\Users\jowan.van.lente\Documents\Thesis\data\results\res.txt'
    # write_results(path, af_pred, bb_pred_test, fidelity, no_ge, test_length, y_test)

    # if explanation is True, print an explanation for the explained instance
    if explanation:
        test_instance = X.iloc[explained_instance:explained_instance + 1, :].to_numpy().reshape(1, -1)
        relevant_arguments = get_relevant_arguments(list_of_arguments, test_instance, feature_names, size=t_select)

        # delete irrelevant attacks
        for argument in relevant_arguments:
            del argument.attacks[:]
            del argument.is_attacked_by[:]
        define_attack_relations(relevant_arguments)
        af = create_AF(relevant_arguments)
        create_explanation(af, t_explain, print_attack=False, biasedness=biasedness)

    return fidelity


def change_argument_strength(list_of_arguments, permutation_importance):
    """This method alters the argument strength based on the permutation importance scores."""
    # alter the argument strength by multiplying with permutation feature importance score
    for argument in list_of_arguments:
        feature_importance = permutation_importance.importances_mean
        current_importance_score = feature_importance[argument.feature_name_digit]
        # print(literal.feature_name_digit, current_importance_score)
        argument.strength_altered = current_importance_score


def write_AF_to_file(file_path: str, af: AF):
    """This method writes an AF to a file."""
    with open(file_path, 'w') as file:
        for argument in af.arguments.values():
            print(argument.name)
            file.write('arg(' + argument.name + ').\n')
        for argument in af.arguments.values():
            for attack in argument.attacks:
                file.write('att(' + argument.name + ',' + attack.name + ').\n')
        file.close()


def write_results(file_path: str, af_pred, bb_pred_test, fidelity, no_ge, test_length, y_test):
    """This method writes results to a file."""
    with open(file_path, 'w') as file:
        file.write('AF predictions: ' + str(af_pred) + '\n' +
                   'BB predictions: ' + str(bb_pred_test.tolist()) + '\n' +
                   'fidelity: ' + str(fidelity) + '\n' +
                   'fraction empty GE:' + str(no_ge / test_length) + '\n' +
                   'BB accuracy' + str(sklearn.metrics.accuracy_score(y_test, bb_pred_test)) + '\n' +
                   'AF accuracy' + str(sklearn.metrics.accuracy_score(y_test, af_pred)))


def print_results(af_pred, bb_pred_test, fidelity, no_ge, test_length, y_test):
    """This method prints the results."""
    print('\n############################ Results ############################')
    # print('AF predictions: ', af_pred)
    # print('BB predictions: ', bb_pred_test.tolist())
    print('Fidelity: ', fidelity)
    print('Fraction empty GE:', no_ge / test_length)
    print('BB accuracy', round(sklearn.metrics.accuracy_score(y_test, bb_pred_test), 2))
    print('AF accuracy', round(sklearn.metrics.accuracy_score(y_test, af_pred), 2), '\n')


def create_explanation(af: AF, explanation_size=None, print_attack=False, biasedness=False):
    """This method creates and prints a dialectical explanation based on an AF.
    When biasedness is set at true, it adds a biased explanation."""

    ge_list, prediction_af, predictionstotal = get_single_prediction_AF(af)
    list_of_arguments = list(af.arguments.values())

    # print the prediction
    print('\n############################ Explanation ############################')
    print('Grounded Extension: ', ge_list, predictionstotal, '\ntherefore the prediction =', prediction_af, '\n')

    # if print attacks is set at True, print them.
    if print_attack:
        print_attacks(af.arguments.values())

    # print dialectical explanation with pro and con arguments.
    pro_arguments = []
    con_arguments = []
    for argument in list_of_arguments:
        if argument.conclusion == prediction_af:
            pro_arguments.append(argument)
        else:
            con_arguments.append(argument)
    print('Dialectical explanation:')
    for i in range(min(len(pro_arguments), len(con_arguments))):
        p = pro_arguments
        o = con_arguments

        # sort pro and con arguments based on strength
        p.sort(key=lambda x: x.strength, reverse=True)
        o.sort(key=lambda x: x.strength, reverse=True)

        print('P: ', p[i].name + ':', p[i].premise_name, '=', p[i].premise_value,
              '-->', p[i].conclusion, '(' + str(p[i].strength) + ')')

        print('O: ', o[i].name + ':', o[i].premise_name, '=', o[i].premise_value,
              '-->', o[i].conclusion, '(' + str(o[i].strength) + ')')

        # break is t_explanation threshold is reached
        if explanation_size is not None:
            if i == explanation_size:
                break

    # print biased explanation
    if biasedness:
        pro_arguments.sort(key=lambda x: x.coverage, reverse=False)
        abnormal_argument = pro_arguments[0]
        print('\nBiased explanation, based on abnormality:')
        o = abnormal_argument

        # abnormality is 1 - coverage.
        abnormality = round(1 - o.coverage, 3)
        print('Most abnormal argument: ', o.name + ':', o.premise_name, '=', o.premise_value,
              '-->', o.conclusion, '(' + str(o.strength) + ') abnormality = ' + str(abnormality) + '\n\n')


def get_dataset(input: str):
    """This method returns a workable dataset"""

    if input == 'stof':
        input_file = r'C:\Users\jowan.van.lente\Documents\Thesis\data\stof\stof.csv'
        col_names = ['wd_dag',
                     'wd_maxuur',
                     'wd_max',
                     'ws_dag',
                     'ws_uurmax',
                     'ws_max',
                     'temp_dag',
                     'prep_dag',
                     'RLH_dag',
                     'NOX_dag',
                     'PM10_dag',
                     'BC_dag',
                     'TSP_dag',
                     'NOX_maxuur',
                     'PM10_maxuur',
                     'BC_maxuur',
                     'BC.NOX_maxuur',
                     'windhoek_max',
                     'windhoek_uur',
                     'windhoek_dag',
                     'Class']

        df = pd.read_csv(input_file, header=None, names=col_names)

        df['Class'] = np.where(df['Class'] > 0, 1, df['Class'])

        le = LabelEncoder()
        for column in df.columns:
            df[column] = le.fit_transform(df[column])

        bins = 10
        for column in col_names[1:]:
            df[column] = pd.cut(df[column], bins)

        one_hot_cols = df.columns.tolist()
        one_hot_cols.remove('Class')
        df = pd.get_dummies(df, columns=one_hot_cols)

    if input == 'wine':
        input_file = r'C:\Users\jowan.van.lente\Documents\Thesis\data\wine\wine.CSV'
        col_names = ['Class', 'Alcohol', 'Malic', 'Ash',
                     'Alcalinity', 'Magnesium', 'Phenols',
                     'Flavanoids', 'Nonflavanoids',
                     'Proanthocyanins', 'Color', 'Hue',
                     'Dilution', 'Proline']

        df = pd.read_csv(input_file, header=None, names=col_names)

        df['Class'] = df['Class'] - 1

        bins = 10
        for column in col_names[1:]:
            df[column] = pd.cut(df[column], bins)

        one_hot_cols = df.columns.tolist()
        one_hot_cols.remove('Class')
        df = pd.get_dummies(df, columns=one_hot_cols)

        first_column = df[list(df.columns)[0]]
        df = df.drop(['Class'], axis=1)
        df.insert(loc=len(df.columns), column='Class', value=first_column)

    if input == 'iris':
        input_file = r'C:\Users\jowan.van.lente\Documents\Thesis\data\iris\iris.CSV'
        col_names = ['sepal length', 'sepal width', 'petal length', 'petal width', 'Class']
        df = pd.read_csv(input_file, header=None, names=col_names)

        le = LabelEncoder()
        df['Class'] = le.fit_transform(df['Class'])

        bins = 10

        df['sepal length'] = pd.cut(df['sepal length'], bins)
        df['sepal width'] = pd.cut(df['sepal width'], bins)
        df['petal length'] = pd.cut(df['petal length'], bins)
        df['petal width'] = pd.cut(df['petal width'], bins)

        one_hot_cols = df.columns.tolist()
        one_hot_cols.remove('Class')
        df = pd.get_dummies(df, columns=one_hot_cols)

        first_column = df[list(df.columns)[0]]
        df = df.drop(['Class'], axis=1)
        df.insert(loc=len(df.columns), column='Class', value=first_column)

        return df

    if input == 'adult bin':
        input_file = r'C:\Users\jowan.van.lente\Documents\Thesis\data\adult\adult.CSV'

        dataset_bin = pd.DataFrame()

        headers = ['age', 'workclass', 'fnlwgt',
                   'education', 'education-num',
                   'marital-status', 'occupation',
                   'relationship', 'race', 'sex',
                   'capital-gain', 'capital-loss',
                   'hours-per-week', 'native-country',
                   'predclass']

        dataset_raw = pd.read_csv(input_file, names=headers, na_values=["?"],
                                  engine='python', sep=',\s')

        dataset_raw.loc[dataset_raw['predclass'] == '>50K', 'predclass'] = 1
        dataset_raw.loc[dataset_raw['predclass'] == '>50K.', 'predclass'] = 1
        dataset_raw.loc[dataset_raw['predclass'] == '<=50K', 'predclass'] = 0
        dataset_raw.loc[dataset_raw['predclass'] == '<=50K.', 'predclass'] = 0

        dataset_bin['predclass'] = dataset_raw['predclass']
        dataset_bin['age'] = pd.cut(dataset_raw['age'], 10)

        dataset_raw.loc[dataset_raw['workclass'] == 'Without-pay', 'workclass'] = 'Not Working'
        dataset_raw.loc[dataset_raw['workclass'] == 'Never-worked', 'workclass'] = 'Not Working'
        dataset_raw.loc[dataset_raw['workclass'] == 'Federal-gov', 'workclass'] = 'Fed-gov'
        dataset_raw.loc[dataset_raw['workclass'] == 'State-gov', 'workclass'] = 'Non-fed-gov'
        dataset_raw.loc[dataset_raw['workclass'] == 'Local-gov', 'workclass'] = 'Non-fed-gov'
        dataset_raw.loc[dataset_raw['workclass'] == 'Self-emp-not-inc', 'workclass'] = 'Self-emp'
        dataset_raw.loc[dataset_raw['workclass'] == 'Self-emp-inc', 'workclass'] = 'Self-emp'

        dataset_bin['workclass'] = dataset_raw['workclass']

        dataset_raw.loc[dataset_raw['occupation'] == 'Adm-clerical', 'occupation'] = 'Admin'
        dataset_raw.loc[dataset_raw['occupation'] == 'Armed-Forces', 'occupation'] = 'Military'
        dataset_raw.loc[dataset_raw['occupation'] == 'Craft-repair', 'occupation'] = 'Manual Labour'
        dataset_raw.loc[dataset_raw['occupation'] == 'Exec-managerial', 'occupation'] = 'Office Labour'
        dataset_raw.loc[dataset_raw['occupation'] == 'Farming-fishing', 'occupation'] = 'Manual Labour'
        dataset_raw.loc[dataset_raw['occupation'] == 'Handlers-cleaners', 'occupation'] = 'Manual Labour'
        dataset_raw.loc[dataset_raw['occupation'] == 'Machine-op-inspct', 'occupation'] = 'Manual Labour'
        dataset_raw.loc[dataset_raw['occupation'] == 'Other-service', 'occupation'] = 'Service'
        dataset_raw.loc[dataset_raw['occupation'] == 'Priv-house-serv', 'occupation'] = 'Service'
        dataset_raw.loc[dataset_raw['occupation'] == 'Prof-specialty', 'occupation'] = 'Professional'
        dataset_raw.loc[dataset_raw['occupation'] == 'Protective-serv', 'occupation'] = 'Military'
        dataset_raw.loc[dataset_raw['occupation'] == 'Sales', 'occupation'] = 'Office Labour'
        dataset_raw.loc[dataset_raw['occupation'] == 'Tech-support', 'occupation'] = 'Office Labour'
        dataset_raw.loc[dataset_raw['occupation'] == 'Transport-moving', 'occupation'] = 'Manual Labour'

        dataset_bin['occupation'] = dataset_raw['occupation']

        dataset_raw.loc[dataset_raw['native-country'] == 'Cambodia', 'native-country'] = 'SE-Asia'
        dataset_raw.loc[dataset_raw['native-country'] == 'Canada', 'native-country'] = 'British-Commonwealth'
        dataset_raw.loc[dataset_raw['native-country'] == 'China', 'native-country'] = 'China'
        dataset_raw.loc[dataset_raw['native-country'] == 'Columbia', 'native-country'] = 'South-America'
        dataset_raw.loc[dataset_raw['native-country'] == 'Cuba', 'native-country'] = 'South-America'
        dataset_raw.loc[dataset_raw['native-country'] == 'Dominican-Republic', 'native-country'] = 'South-America'
        dataset_raw.loc[dataset_raw['native-country'] == 'Ecuador', 'native-country'] = 'South-America'
        dataset_raw.loc[dataset_raw['native-country'] == 'El-Salvador', 'native-country'] = 'South-America'
        dataset_raw.loc[dataset_raw['native-country'] == 'England', 'native-country'] = 'British-Commonwealth'
        dataset_raw.loc[dataset_raw['native-country'] == 'France', 'native-country'] = 'Euro_Group_1'
        dataset_raw.loc[dataset_raw['native-country'] == 'Germany', 'native-country'] = 'Euro_Group_1'
        dataset_raw.loc[dataset_raw['native-country'] == 'Greece', 'native-country'] = 'Euro_Group_2'
        dataset_raw.loc[dataset_raw['native-country'] == 'Guatemala', 'native-country'] = 'South-America'
        dataset_raw.loc[dataset_raw['native-country'] == 'Haiti', 'native-country'] = 'South-America'
        dataset_raw.loc[dataset_raw['native-country'] == 'Holand-Netherlands', 'native-country'] = 'Euro_Group_1'
        dataset_raw.loc[dataset_raw['native-country'] == 'Honduras', 'native-country'] = 'South-America'
        dataset_raw.loc[dataset_raw['native-country'] == 'Hong', 'native-country'] = 'China'
        dataset_raw.loc[dataset_raw['native-country'] == 'Hungary', 'native-country'] = 'Euro_Group_2'
        dataset_raw.loc[dataset_raw['native-country'] == 'India', 'native-country'] = 'British-Commonwealth'
        dataset_raw.loc[dataset_raw['native-country'] == 'Iran', 'native-country'] = 'Euro_Group_2'
        dataset_raw.loc[dataset_raw['native-country'] == 'Ireland', 'native-country'] = 'British-Commonwealth'
        dataset_raw.loc[dataset_raw['native-country'] == 'Italy', 'native-country'] = 'Euro_Group_1'
        dataset_raw.loc[dataset_raw['native-country'] == 'Jamaica', 'native-country'] = 'South-America'
        dataset_raw.loc[dataset_raw['native-country'] == 'Japan', 'native-country'] = 'APAC'
        dataset_raw.loc[dataset_raw['native-country'] == 'Laos', 'native-country'] = 'SE-Asia'
        dataset_raw.loc[dataset_raw['native-country'] == 'Mexico', 'native-country'] = 'South-America'
        dataset_raw.loc[dataset_raw['native-country'] == 'Nicaragua', 'native-country'] = 'South-America'
        dataset_raw.loc[
            dataset_raw['native-country'] == 'Outlying-US(Guam-USVI-etc)', 'native-country'] = 'South-America'
        dataset_raw.loc[dataset_raw['native-country'] == 'Peru', 'native-country'] = 'South-America'
        dataset_raw.loc[dataset_raw['native-country'] == 'Philippines', 'native-country'] = 'SE-Asia'
        dataset_raw.loc[dataset_raw['native-country'] == 'Poland', 'native-country'] = 'Euro_Group_2'
        dataset_raw.loc[dataset_raw['native-country'] == 'Portugal', 'native-country'] = 'Euro_Group_2'
        dataset_raw.loc[dataset_raw['native-country'] == 'Puerto-Rico', 'native-country'] = 'South-America'
        dataset_raw.loc[dataset_raw['native-country'] == 'Scotland', 'native-country'] = 'British-Commonwealth'
        dataset_raw.loc[dataset_raw['native-country'] == 'South', 'native-country'] = 'Euro_Group_2'
        dataset_raw.loc[dataset_raw['native-country'] == 'Taiwan', 'native-country'] = 'China'
        dataset_raw.loc[dataset_raw['native-country'] == 'Thailand', 'native-country'] = 'SE-Asia'
        dataset_raw.loc[dataset_raw['native-country'] == 'Trinadad&Tobago', 'native-country'] = 'South-America'
        dataset_raw.loc[dataset_raw['native-country'] == 'United-States', 'native-country'] = 'United-States'
        dataset_raw.loc[dataset_raw['native-country'] == 'Vietnam', 'native-country'] = 'SE-Asia'
        dataset_raw.loc[dataset_raw['native-country'] == 'Yugoslavia', 'native-country'] = 'Euro_Group_2'

        dataset_bin['native-country'] = dataset_raw['native-country']

        dataset_raw.loc[dataset_raw['education'] == '10th', 'education'] = 'Dropout'
        dataset_raw.loc[dataset_raw['education'] == '11th', 'education'] = 'Dropout'
        dataset_raw.loc[dataset_raw['education'] == '12th', 'education'] = 'Dropout'
        dataset_raw.loc[dataset_raw['education'] == '1st-4th', 'education'] = 'Dropout'
        dataset_raw.loc[dataset_raw['education'] == '5th-6th', 'education'] = 'Dropout'
        dataset_raw.loc[dataset_raw['education'] == '7th-8th', 'education'] = 'Dropout'
        dataset_raw.loc[dataset_raw['education'] == '9th', 'education'] = 'Dropout'
        dataset_raw.loc[dataset_raw['education'] == 'Assoc-acdm', 'education'] = 'Associate'
        dataset_raw.loc[dataset_raw['education'] == 'Assoc-voc', 'education'] = 'Associate'
        dataset_raw.loc[dataset_raw['education'] == 'Bachelors', 'education'] = 'Bachelors'
        dataset_raw.loc[dataset_raw['education'] == 'Doctorate', 'education'] = 'Doctorate'
        dataset_raw.loc[dataset_raw['education'] == 'HS-Grad', 'education'] = 'HS-Graduate'
        dataset_raw.loc[dataset_raw['education'] == 'Masters', 'education'] = 'Masters'
        dataset_raw.loc[dataset_raw['education'] == 'Preschool', 'education'] = 'Dropout'
        dataset_raw.loc[dataset_raw['education'] == 'Prof-school', 'education'] = 'Professor'
        dataset_raw.loc[dataset_raw['education'] == 'Some-college', 'education'] = 'HS-Graduate'

        dataset_bin['education'] = dataset_raw['education']

        dataset_raw.loc[dataset_raw['marital-status'] == 'Never-married', 'marital-status'] = 'Never-Married'
        dataset_raw.loc[dataset_raw['marital-status'] == 'Married-AF-spouse', 'marital-status'] = 'Married'
        dataset_raw.loc[dataset_raw['marital-status'] == 'Married-civ-spouse', 'marital-status'] = 'Married'
        dataset_raw.loc[dataset_raw['marital-status'] == 'Married-spouse-absent', 'marital-status'] = 'Not-Married'
        dataset_raw.loc[dataset_raw['marital-status'] == 'Separated', 'marital-status'] = 'Separated'
        dataset_raw.loc[dataset_raw['marital-status'] == 'Divorced', 'marital-status'] = 'Separated'
        dataset_raw.loc[dataset_raw['marital-status'] == 'Widowed', 'marital-status'] = 'Widowed'

        dataset_bin['marital-status'] = dataset_raw['marital-status']

        dataset_bin['fnlwgt'] = pd.cut(dataset_raw['fnlwgt'], 10)

        dataset_bin['education-num'] = pd.cut(dataset_raw['education-num'], 10)

        dataset_bin['hours-per-week'] = pd.cut(dataset_raw['hours-per-week'], 10)

        dataset_bin['capital-gain'] = pd.cut(dataset_raw['capital-gain'], 5)

        dataset_bin['capital-loss'] = pd.cut(dataset_raw['capital-loss'], 5)

        dataset_bin['sex'] = dataset_raw['sex']
        dataset_bin['race'] = dataset_raw['race']
        dataset_bin['relationship'] = dataset_raw['relationship']

        one_hot_cols = dataset_bin.columns.tolist()
        one_hot_cols.remove('predclass')
        dataset_bin_enc = pd.get_dummies(dataset_bin, columns=one_hot_cols)

        return dataset_bin_enc

    if input == 'adult':
        input_file = r'C:\Users\jowan.van.lente\Documents\Thesis\data\adult\adult.CSV'
        col_names = ["Age", "Workclass", "fnlwgt", "Education",
                     "Education-Num", "Marital Status", "Occupation",
                     "Relationship", "Race", "Sex", "Capital Gain",
                     "Capital Loss", "Hours per week", "Country", 'Class']

        df = pd.read_csv(input_file, names=col_names)

        df['Country'] = df['Country'].replace(' ?', np.nan)
        df['Workclass'] = df['Workclass'].replace(' ?', np.nan)
        df['Occupation'] = df['Occupation'].replace(' ?', np.nan)

        df.dropna(how='any', inplace=True)
        df = df.fillna(0)

        df.drop(['Education-Num', 'Age', 'Hours per week', 'fnlwgt', 'Capital Gain', 'Capital Loss', 'Country'], axis=1,
                inplace=True)

        df.replace({'Class': {' <=50K': 0, ' >50K': 1}}, inplace=True)
        df.replace({'Sex': {' Male': 0, ' Female': 1}}, inplace=True)
        df.replace(
            {'Race': {' Black': 0, ' Asian-Pac-Islander': 1, ' Other': 2, ' White': 3, ' Amer-Indian-Eskimo': 4}},
            inplace=True)
        df.replace({'Marital Status': {' Married-spouse-absent': 0, ' Widowed': 1, ' Married-civ-spouse': 2,
                                       ' Separated': 3, ' Divorced': 4, ' Never-married': 5, ' Married-AF-spouse': 6}},
                   inplace=True)
        df.replace(
            {'Workclass': {' Self-emp-inc': 0, ' State-gov': 1, ' Federal-gov': 2, ' Without-pay': 3, ' Local-gov': 4,
                           ' Private': 5, ' Self-emp-not-inc': 6}}, inplace=True)
        df.replace({'Education': {' Some-college': 0, ' Preschool': 1, ' 5th-6th': 2, ' HS-grad': 3, ' Masters': 4,
                                  ' 12th': 5, ' 7th-8th': 6,
                                  ' Prof-school': 7, ' 1st-4th': 8, ' Assoc-acdm': 9, ' Doctorate': 10, ' 11th': 11,
                                  ' Bachelors': 12,
                                  ' 10th': 13, ' Assoc-voc': 14, ' 9th': 15}}, inplace=True)
        df.replace(
            {'Occupation': {' Farming-fishing': 1, ' Tech-support': 2, ' Adm-clerical': 3, ' Handlers-cleaners': 4,
                            ' Prof-specialty': 5, ' Machine-op-inspct': 6, ' Exec-managerial': 7, ' Priv-house-serv': 8,
                            ' Craft-repair': 9, ' Sales': 10,
                            ' Transport-moving': 11, ' Armed-Forces': 12, ' Other-service': 13,
                            ' Protective-serv': 14}}, inplace=True)
        df.replace({'Relationship': {' Not-in-family': 0, ' Wife': 1, ' Other-relative': 2, ' Unmarried': 3,
                                     ' Husband': 4, ' Own-child': 5}}, inplace=True)

    if input == 'mushroom':
        input_file = r'C:\Users\jowan.van.lente\Documents\Thesis\data\mushroom\mushrooms.csv'
        df = pd.read_csv(input_file)

        # change class column to last column
        first_column = df[list(df.columns)[0]]
        df = df.drop(['class'], axis=1)
        df.insert(loc=len(df.columns), column='Class', value=first_column)

        le = LabelEncoder()
        for column in df.columns:
            df[column] = le.fit_transform(df[column])

    return df


def get_fitted_clf(df, clf_type: str):
    """This method returns a fitted classifier"""

    feature_names = df.columns.values[:-1]
    X = df[feature_names]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    if clf_type == 'Random Forest':
        clf = RandomForestClassifier(n_estimators=10, n_jobs=3, random_state=42, max_depth=20, max_features=4)
        clf.fit(X=X_train, y=y_train)
        print('Random Forest classifier fitted...')

    elif clf_type == 'Logistic Regression':
        clf = LogisticRegression(solver='lbfgs', random_state=42, max_iter=1000)
        clf.fit(X=X_train, y=y_train)
        print('Logistic Regression classifier fitted...')

    elif clf_type == 'SVM':
        clf = SVC()
        clf.fit(X=X_train, y=y_train)
        print('SVM classifier fitted...')

    elif clf_type == 'Neural Network':
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=42)
        clf.fit(X=X_train, y=y_train)
        print('Neural Network classifier fitted...')

    else:
        print('clf-type invalid')

    return clf


def get_variables_from_df(df: pd.DataFrame):
    """This method gets feature names, splits labels and unlabeled data an extracts number of classes."""
    feature_names = df.columns.values[:-1]
    X = df[feature_names]
    y = df.iloc[:, -1]
    n_of_classes = y.nunique()

    return feature_names, X, y, n_of_classes


def ask_oracle_for_single_prediction(row_number, clf, X):
    """This method looks up prediction of black box for specific instance"""
    instance = X.iloc[row_number:row_number+1, :].to_numpy().reshape(1, -1)
    prediction = clf.predict(instance)
    return int(prediction)


def get_list_of_literals(clf, feature_names, X):
    """This method produces a list of feature-value pairs (or literals) based on the train set.
    Alter max_iter to decide how many datapoints are handled"""

    list_of_literals = []
    max_iter = 2000

    if len(X) > max_iter:
        length = max_iter
    else:
        length = len(X)

    bb_pred = clf.predict(X)
    # for all data points
    for i in range(length):
        # for all features
        count = -1
        for name in feature_names:
            # for all possible values
            size = X[name].nunique()
            count += 1
            for j in range(size):
                value = j
                # if value corresponds with current feature
                if X.iloc[i][name] == value:
                    # construct a literal object with that name and value
                    for literal in list_of_literals:
                        if literal.name == name and literal.value == value:
                            current_literal = literal
                            break
                    else:
                        current_literal = Literal(name, value, feature_name_digit=count)
                        list_of_literals.append(current_literal)

                    outcome = bb_pred[i]
                    current_literal.add_outcome(outcome)
    return list_of_literals


def print_arguments(list_of_arguments):
    """This method prints arguments."""
    for argument in list_of_arguments:
        print(argument.name + ':', argument.premise_name, '=', argument.premise_value,
              '-->', argument.conclusion, 'P:', argument.strength)
    print('\n')


def convert_literals_to_arguments(list_of_literals, n_of_classes, len_dataset, permutation_importance=None,
                                  normalize=False,
                                  y_train=None, divide_by_class_distribution=False):
    """This method converst the list of literals to a list of arguments."""
    list_of_arguments = []
    count = 0

    # create an argument for all (literal, output class) pairs
    for literal in list_of_literals:
        for c in range(n_of_classes):
            count += 1

            # get precision and coverage of the decision rule that describes (literal) --> output class
            # precision (prec) describes the accuracy of the decision rule
            # coverage (cov) describes the amount of data points the rule 'rules over'
            prec, cov = get_precision_coverage(literal, c,
                                               y_train=y_train,
                                               divide_by_class_distribution=divide_by_class_distribution)
            cov = cov/len_dataset
            argument = Argument('a' + str(count), literal, c, prec, cov)
            list_of_arguments.append(argument)
            c += 1

    return list_of_arguments


def get_relevant_arguments(list_of_arguments, input_instance, feature_names, size):
    """This method creates a list of relevant arguments, which will be the basis for the local_AF"""
    relevant_arguments = []
    feature_value_pairs = {}

    # match values of input instance to corresponding feature name
    for i in range(len(feature_names)):
        feature_value_pairs[feature_names[i]] = input_instance[0, i]

    # add relevant arguments to list if it corresponds to a feature-value pair of the input instance
    for argument in list_of_arguments:
        for key in feature_value_pairs:
            if argument.premise_name == key and argument.premise_value == feature_value_pairs[key]:
                relevant_arguments.append(argument)

    relevant_arguments.sort(key=lambda x: x.strength, reverse=True)

    # if threshold t_select is given, select only the top t_select arguments.
    if size is not None:
        relevant_arguments = relevant_arguments[:size]

    return relevant_arguments


def define_attack_relations(list_of_arguments, feature_importance=None):
    """This method defines the attack relations."""
    # define attack relation between a and b if a has a different conclusion and is stronger or equally strong as b
    # if permutation feature importance is used, use the altered argument strength
    if feature_importance:
        for argument1 in list_of_arguments:
            for argument2 in list_of_arguments:
                if argument1.name != argument2.name:
                    if (argument1.conclusion != argument2.conclusion
                            and argument1.strength_altered >= argument2.strength_altered):
                        argument1.add_attack(argument2)
    else:
        for argument1 in list_of_arguments:
            for argument2 in list_of_arguments:
                if argument1.name != argument2.name:
                    if (argument1.conclusion != argument2.conclusion
                            and argument1.strength >= argument2.strength):
                        argument1.add_attack(argument2)


def get_fidelity(pred_af, pred_bb):
    """This method calculates the fidelity."""
    # fidelity = fraction of datapoints that is assigned to the same output class (by both black box and EVAX)
    n_predictions = len(pred_bb)
    n_same = 0
    for i in range(n_predictions):
        if pred_af[i] == pred_bb[i]:
            n_same += 1

    return round(n_same/n_predictions, 2)


def get_precision_coverage(literal: Literal, outcome: int,
                           y_train, divide_by_class_distribution=False):
    """This method calculates the precision and coverage of the decision rules (literal --> outcome)"""
    # result is the precision of the decision rule, total the coverage
    total = len(literal.outcomes)
    count = literal.outcomes.count(outcome)
    result = count/total

    if divide_by_class_distribution:
        total_class = len(y_train[y_train == outcome])
        result = result / total_class

    return round(result, 5), total


def normalize_probability_of_arguments(list_of_arguments: [Argument]):
    """This method normalizes all strength scores of arguments between 0 and 1.
    This is necessary when argument strength is altered based on permutation importance scores."""
    list_of_arguments = list_of_arguments
    mini = min(list_of_arguments, key=lambda x: x.strength).strength
    maxi = max(list_of_arguments, key=lambda x: x.strength).strength

    for argument in list_of_arguments:
        max_min = maxi - mini
        if maxi - mini == 0:
            max_min = 0.01

        argument.strength = round((argument.strength - mini) / max_min, 2)

    return list_of_arguments


def print_attacks(list_of_arguments):
    """This method prints the attacks of a list of arguments"""
    for argument in list_of_arguments:
        for argument1 in argument.attacks:
            print(argument.name + ' attacks ' + argument1.name)
    print('\n')


def create_AF(list_of_arguments):
    """This method creates an AF-object based on a list of arguments."""
    af = AF()
    for argument in list_of_arguments:
        af.add_argument(argument)
    return af


def get_grounded_extension(af: AF):
    """This method computes the grounded extension of an AF"""
    labeler = Labeler()
    grounded_extension = labeler.get_extension(af, 'grounded')

    return grounded_extension


def get_single_prediction_AF(af: AF, default_prediction=0):
    """This method computes the prediction of an AF"""
    predictions = []
    ge = get_grounded_extension(af)
    ge_list = []
    for argument in ge:
        predictions.append(argument.conclusion)
        ge_list.append(argument.name)
    if not predictions == []:
        # print('grounded extension = ' + str(ge_list))
        predictionstotal = predictions
        predictions = mode(predictions)
    else:
        predictionstotal = predictions
        predictions = default_prediction
        global no_ge
        no_ge += 1

    return ge_list, predictions, predictionstotal
