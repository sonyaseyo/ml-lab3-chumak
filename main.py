import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz

# 1
file_path = 'dataset3.csv'
data = pd.read_csv(file_path)

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
x = data.iloc[:, :-1]  # Вхідні ознаки (усі стовпці крім останнього)
y = data.iloc[:, -1]   # Цільова змінна (останній стовпець)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 5
train_model_entropy = DecisionTreeClassifier(max_depth=5, criterion='entropy') #max_leafs додати, прибрати max_depth
train_model_entropy.fit(x_train, y_train)

train_model_gini = DecisionTreeClassifier(max_depth=5, criterion='gini')
train_model_gini.fit(x_train, y_train)


# 6
image1 = export_graphviz(
    train_model_entropy,
    feature_names=x_train.columns,
    class_names=y_train.unique().astype(str).tolist()
    )

graph1 = graphviz.Source(image1)
graph1.render("decision_tree1")

image2 = export_graphviz(
    train_model_gini,
    feature_names=x_train.columns,
    class_names=y_train.unique().astype(str).tolist()
    )

graph2 = graphviz.Source(image2)
graph2.render("decision_tree2")
