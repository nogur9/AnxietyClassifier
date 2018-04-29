import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from DataImporting.DataFromExcel import get_data,refactor_labels

dataset = refactor_labels(get_data("C:\\Users\\user\\PycharmProjects\\AnxietyClassifier\\ExtractedFeatures\\formatted_features_SAD.xlsx", "Sheet1"),"group")
3:62
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
pca = PCA(n_components=2)
pca.fit(X)
X = pca.transform(X)
# Reorder the labels to have colors matching the cluster results
fig = plt.figure(1, figsize=(4, 3))
y = np.choose(y, [1, 2, 0]).astype(np.float)
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.spectral,
           edgecolor='k')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])

plt.show()