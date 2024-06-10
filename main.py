import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import tree, __all__, metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, balanced_accuracy_score
import inspect
import warnings
warnings.filterwarnings("ignore")
RAND = 111


def converted_labels(pred, labels):
    global Y_true_labels
    values = Y_true_labels.copy()

    def convert(series, values):
        series[series == values[0]] = 1
        series[series == values[1]] = 0
        return series

    return convert(pred, values), convert(labels, values)


def plot_metrics(set1, set2, labels, title):
    metrics = [el[0] for el in set1]
    values_list1 = [el[1] for el in set1]
    values_list2 = [el[1] for el in set2]

    bar_width = 0.35

    index = np.arange(len(metrics))

    subplots = 1
    fig, axs = plt.subplots(subplots, 1, figsize=(12, 8))

#     for i in range(subplots):
    axs.bar(index - bar_width/2, values_list1, bar_width, label=labels[0], color='#ffa781')
    axs.bar(index + bar_width/2, values_list2, bar_width, label=labels[1], color='#5b0e2d')

    axs.set_ylabel('Values')
    axs.set_title(title)
    axs.set_xticks(index)
    axs.set_xticklabels(metrics, rotation=45)
    axs.set_yticks(np.linspace(0, 1, 11))
    axs.legend()

    plt.tight_layout()
    plt.show()





def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]


def print_metrics(y_pred, y_true, name, with_print=False):
    y_pred, y_true = converted_labels(pd.Series(y_pred), pd.Series(y_true))
    y_pred = y_pred.values.astype(int)
    y_true = y_true.values.astype(int)

    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)
    F1_score = metrics.f1_score(y_true, y_pred)
    BA = metrics.balanced_accuracy_score(y_true, y_pred)
    MCC = metrics.matthews_corrcoef(y_true, y_pred)

    lst = [(retrieve_name(accuracy)[0], accuracy),
           (retrieve_name(precision)[0], precision),
           (retrieve_name(recall)[0], recall),
           (retrieve_name(F1_score)[0], F1_score),
           (retrieve_name(BA)[0], BA),
           (retrieve_name(MCC)[0], MCC)]

    if with_print:
        print(f'Metrics for {name}:')
        for el in lst:
            print(f'\t{el[0]}: {el[1]}')

    return lst


# 1
file_path = 'dataset3.csv'
data = pd.read_csv(file_path)
Y_true_labels = data.iloc[:, -1].unique().tolist()
print(f'Список True позначок класів: {Y_true_labels}')

# 2
num_records = len(data)
num_fields = len(data.columns)

print(f"к-кість записів: {num_records}")
print(f"к-кість полів: {num_fields}")

# 3
first_10_records = data.head(10)
print("\nперші 10 рядків:")
print(first_10_records)

# 4
data = data.sample(frac=1)
data

eighty = int(len(data) * 0.8)

train_data = data.iloc[:eighty]
test_data = data.iloc[eighty:]

# 5
X_train = pd.DataFrame(train_data.iloc[:, :-1])
Y_train = pd.Series(train_data.iloc[:, -1])

dt_entropy = tree.DecisionTreeClassifier(max_depth=5, criterion='entropy', random_state=RAND)
dt_entropy = dt_entropy.fit(X_train, Y_train)
dt_entropy

DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=111)

# 6
image1 = export_graphviz(
    dt_entropy,
    feature_names=X_train.columns,
    class_names=Y_train.unique().astype(str).tolist()
    )

graph1 = graphviz.Source(image1)
graph1.render("decision_tree1")



# 7

train_entropy = print_metrics(dt_entropy.predict(X_train), Y_train.copy(), 'DecisionTree on training set (entropy)', with_print=True)
X_test = pd.DataFrame(test_data.iloc[:, :-1])
Y_test = pd.Series(test_data.iloc[:, -1])
test_entropy = print_metrics(dt_entropy.predict(X_test), Y_test.copy(), 'DecisionTree on test set (entropy)', with_print=True)
plot_metrics(train_entropy, test_entropy, labels=['Training Set', 'Test Set'], title='DecisionTree (entropy): Training set vs Test set')

dt_gini = tree.DecisionTreeClassifier(max_depth=5, criterion='gini', random_state=RAND)
dt_gini = dt_gini.fit(X_train, Y_train)
dt_gini
train_gini = print_metrics(dt_gini.predict(X_train), Y_train.copy(), 'DecisionTree on training set (gini)', with_print=True)
test_gini = print_metrics(dt_gini.predict(X_test), Y_test.copy(), 'DecisionTree on test set (gini)', with_print=True)
plot_metrics(train_entropy, train_gini, labels=['entropy', 'gini'], title='DecisionTree on training set: entropy vs gini')
plot_metrics(test_entropy, test_gini, labels=['entropy', 'gini'], title='DecisionTree on test set: entropy vs gini')

# 8
Xs = data.iloc[:, :-1]
Ys = data.iloc[:, -1]


def max_leaf_nodes(criterion, n):
    grouped_metrics = dict()

    for i in range(2, 2 * n + 1):
        some_tree = tree.DecisionTreeClassifier(criterion=criterion, max_leaf_nodes=i, random_state=RAND)
        some_tree = some_tree.fit(X_train, Y_train)

        metrics = print_metrics(some_tree.predict(Xs), Ys.copy(), '')

        for j in range(len(metrics)):
            grouped_metrics.update({metrics[j][0]: grouped_metrics.get(metrics[j][0], []) + [metrics[j][1]]})

    return grouped_metrics


def min_samples_leaf(criterion, n):
    grouped_metrics = dict()

    for i in range(2, 2 * n + 1):
        some_tree = tree.DecisionTreeClassifier(criterion=criterion, min_samples_leaf=i, random_state=RAND)
        some_tree = some_tree.fit(X_train, Y_train)

        metrics = print_metrics(some_tree.predict(Xs), Ys.copy(), '')

        for j in range(len(metrics)):
            grouped_metrics.update({metrics[j][0]: grouped_metrics.get(metrics[j][0], []) + [metrics[j][1]]})

    return grouped_metrics


def plot_changes(tree, no_of_nodes, name, title):
    plt.figure(figsize=(12, 10))
    for k, v in tree.items():
        plt.plot(np.arange(2, 2 * no_of_nodes + 1), v, label=k)
    plt.xlabel(name)
    plt.ylabel('Значення метрики')
    plt.title(title)
    plt.legend()
    plt.show()


entropy_1 = max_leaf_nodes('entropy', dt_entropy.tree_.node_count)

plot_changes(entropy_1, dt_entropy.tree_.node_count, 'Максимальна кількість листів', title='entropy')

entropy_2 = min_samples_leaf('entropy', dt_entropy.tree_.node_count)
plot_changes(entropy_2, dt_entropy.tree_.node_count, 'Мінімальна кількість елементів для розбиття', title='entropy')

# 9
feature_importance = list(zip(X_train.columns, dt_gini.feature_importances_))

xs = [el[0] for el in feature_importance]
ys = [el[1] for el in feature_importance]

plt.figure(figsize=(12, 6))
plt.xticks(np.arange(len(xs)))
plt.bar(xs, ys)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.tight_layout()
plt.show()


