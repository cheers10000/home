import numpy as np
class KMeans:
    def __init__(self, data,k):
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be a numpy array")
        self.data = data
        self.k = k
    def train(self,max_iter):
        center=KMeans.center_chooser(self.data,self.k)
        num_of_data = self.data.shape[0]
        closest_center_dis = np.zeros(num_of_data,dtype=int)
        for i in range(max_iter):
            # 更新每个点的最近中心索引
            for j in range(num_of_data):
                closest_center_dis[j] = KMeans.min_dis(self.data[j],center)
            # 更新聚类中心
            new_center = KMeans.center_compute(self.data,closest_center_dis,self.k)
            if np.all(center==new_center):
                break
            center = new_center
        return center,closest_center_dis
    @staticmethod
    def center_chooser(data,k):
        num_of_data = data.shape[0]
        random_index = np.random.choice(num_of_data,k,replace=False)
        return data[random_index]
    @staticmethod
    def min_dis(point,centers):
        # num_of_data = data.shape[0]
        # k = len(center)
        # closest_center_dis = np.zeros((k, 1))
        # for i in range(k):
        #     distance = np.zeros((k, 1))
        #     for j in range(k):
        #         distance_diff = data[i,:] - center[j,:]
        #         distance[j] = np.sum(distance_diff ** 2)
        #     closest_center_dis[i] = np.argmin(distance)
        # return closest_center_dis
        return np.argmin(np.linalg.norm(centers-point,axis=1))
    @staticmethod
    def center_compute(data,labels,k):
        num_features = data.shape[1]
        centers = np.zeros((k,num_features))
        for j in range(k):
             cluster_data=data[labels==j]
             if len(cluster_data)==0:
                 # 如果某个簇没有数据点，随机选择一个数据点作为中心
                 centers[j] = data[np.random.randint(0,data.shape[0])]
             else:
                 centers[j] = np.mean(cluster_data,axis=0)

        return centers
