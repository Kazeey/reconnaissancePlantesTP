from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn import neighbors
from ipywidgets import interact

iris = datasets.load_iris()
target = iris.target
data = iris.data

for i in [0, 1, 2]:
    print("Classe : %s (%s), nb exemplaires : %s" % (i, iris.target_names[i], len(target[target == i])))

print(type(data), data.ndim, data.shape)

fig = plt.figure(figsize=(8, 4))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
ax1 = plt.subplot(1, 2, 1)

clist = ['violet', 'yellow', 'blue']
colors = [clist[c] for c in target]

# Formate les données du tableau avec tous les index 0 et 1
ax1.scatter(data[:, 0], data[:, 1], c=colors)
plt.xlabel('Longueur du sepal (cm)')
plt.ylabel('Longueur du sepal (cm)')

ax2 = plt.subplot(1, 2, 2)
# Formate les données du tableau avec tous les index 2 et 3
ax2.scatter(data[:, 2], data[:, 3], c=colors)

plt.xlabel('Longueur du petal (cm)')
plt.ylabel('Longueur du petal (cm)')

# Légende
for ind, s in enumerate(iris.target_names):
    plt.scatter([], [], label=s, c=clist[ind])

plt.legend(scatterpoints=1, frameon=False, labelspacing=1, bbox_to_anchor=(1.8, .5), loc="center right", title="Espèces")

sns.set()
df = pd.DataFrame(data, columns=iris['feature_names'])
df['target'] = target
df['label'] = df.apply(lambda x:iris['target_names'][int(x.target)], axis=1)
df.head()

sns.pairplot(df, hue='label', vars=iris['feature_names'], height=2)

clf = GaussianNB()

clf.fit(data, target)
result = clf.predict(data)

errors = sum(result != target)
print("Nb erreurs :", errors)
print("Pourcentage de prédiction juste: ", accuracy_score(result, target)*100, "%")

conf = confusion_matrix(target, result)
sns.heatmap(conf, square=True, annot=True, cbar=False
            , xticklabels=list(iris.target_names)
            , yticklabels=list(iris.target_names))
plt.xlabel('valeurs prédites')
plt.ylabel('valeurs réelles')
plt.show()

plt.matshow(conf, cmap='rainbow')

data_test, target_test = data[::2], target[::2]
data_train, target_train = data[1::2], target[1::2]

data_test = train_test_split(data, target, random_state=0, train_size=0.5)
data_train, data_test, target_train, target_test = data_test

clf = GaussianNB()
clf.fit(data_train, target_train)
result = clf.predict(data_test)

print("Pourcentage de prédiction juste: ", accuracy_score(result, target_test)*100, "%")

conf = confusion_matrix(target_test, result)

sns.heatmap(conf, square=True, annot=True, cbar=False
            , xticklabels=list(iris.target_names)
            , yticklabels=list(iris.target_names))
plt.xlabel('valeurs prédites')
plt.ylabel('valeurs réelles')
plt.show()

data = iris.data[:, :2]
target = iris.target

clf = GaussianNB()
clf.fit(data, target)
h = .15
x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1

x = np.arange(x_min, x_max, h)
y = np.arange(y_min, y_max, h)

print("xMax : ", x_max, " / xMin : ", x_min," / yMax :", y_max ," / yMin : ", y_min)

xx, yy = np.meshgrid(x, y)
data_samples = list(zip(xx.ravel(), yy.ravel()))

z = clf.predict(data_samples)
plt.figure(1)

colors = ['violet', 'yellow', 'red']
C = [colors[x] for x in z]

plt.scatter(xx.ravel(), yy.ravel(), c=C)
plt.xlim(xx.min() - .1, xx.max() + .1)
plt.ylim(yy.min() - .1, yy.max() + .1)
plt.xlabel("Longueur du sépal (cm)")
plt.xlabel("Largeur du sépal (cm)")

plt.show()

plt.figure(1)
plt.pcolormesh(xx, yy, z.reshape(xx.shape), shading='flat')
colors = ['violet', 'yellow', 'red']
C = [colors[x] for x in target]
plt.scatter(data[:, 0], data[:, 1], c=C)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xlabel('Longueur du sepal (cm)')
plt.ylabel('Largueur du sepal (cm)')

plt.show()

clf = neighbors.KNeighborsClassifier()

@interact(n=(0, 20))
def n_change(n=5):
    clf = neighbors.KNeighborsClassifier(n_neighbors=n)
    clf.fit(data, target)
    Z = clf.predict(data_samples)
    plt.figure(1)
    plt.pcolormesh(xx, yy, Z.reshape(xx.shape), shading='flat')
    C = [colors[x] for x in target]
    plt.scatter(data[:, 0], data[:, 1], c=C)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel('Longueur du sepal (cm)')
    plt.ylabel('Longueur du sepal (cm)')

data_test, target_test = iris.data[::2], iris.target[::2]
data_train, target_train = iris.data[1::2], iris.target[1::2]

result = []
n_values = range(1, 20)

for n in n_values:
    clf = neighbors.KNeighborsClassifier(n_neighbors=n)
    clf.fit(data_train, target_train)
    Z = clf.predict(data_test)
    score = accuracy_score(Z, target_test)
    result.append(score)

plt.plot(list(n_values), result)

plt.show()



