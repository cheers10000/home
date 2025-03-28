import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from k_means import KMeans

data=pd.read_csv(r"D:\Iris数据集\iris.csv")
iris_types=["setosa","versicolor","virginica"]
x_axis="Petal.Length"
y_axis="Petal.Width"

plt.figure(figsize=(10,8))
plt.subplot(1,2,1)
for i in iris_types:
    plt.scatter(data[x_axis][data["Species"]==i],data[y_axis][data["Species"]==i],label=i)
plt.title("Riginal Iris Data")
plt.legend()
plt.show()

num_of_data=data.shape[0]
x_train=data[[x_axis,y_axis]].values
max_iter=100
kmeans=KMeans(data=x_train,k=3)
center,closest_center_dis=kmeans.train(max_iter)
plt.figure(figsize=(10,8))
plt.subplot(1,2,1)
for i in iris_types:
    plt.scatter(data[x_axis][data["Species"]==i],data[y_axis][data["Species"]==i],label=i)
plt.title("Iris Data")
plt.legend()

plt.subplot(1,2,2)
for i,j in enumerate(center):
    current_index=np.where(closest_center_dis==i)
    plt.scatter(x_train[current_index,0],x_train[current_index,1],label=iris_types[i])
plt.scatter(center[:,0],center[:,1],c="red",marker="x")
plt.title("K-Means Clustering")
plt.legend()
plt.show()

