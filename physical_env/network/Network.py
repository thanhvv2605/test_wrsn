import copy
import numpy as np
import numpy as np
from sklearn.cluster import KMeans
import math
from sklearn.metrics import silhouette_score
from ChargingLocation import ChargingLocation

class Network:
    def __init__(self, env, listNodes, baseStation, listTargets, max_time):
        self.env = env
        self.listNodes = listNodes
        self.baseStation = baseStation
        self.listTargets = listTargets
        self.targets_active = [1 for _ in range(len(self.listTargets))]
        self.alive = 1
        # Setting BS and Node environment and network
        baseStation.env = self.env
        baseStation.net = self
        self.max_time = max_time

        self.frame = np.array([self.baseStation.location[0], self.baseStation.location[0], self.baseStation.location[1], self.baseStation.location[1]], np.float64)
        it = 0
        for node in self.listNodes:
            node.env = self.env
            node.net = self
            node.id = it
            it += 1
            self.frame[0] = min(self.frame[0], node.location[0])
            self.frame[1] = max(self.frame[1], node.location[0])
            self.frame[2] = min(self.frame[2], node.location[1])
            self.frame[3] = max(self.frame[3], node.location[1])
        self.nodes_density = len(self.listNodes) / ((self.frame[1] - self.frame[0]) * (self.frame[3] - self.frame[2]))
        it = 0

        # Setting name for each target
        for target in listTargets:
            target.id = it
            it += 1
        list_charging_locations, charging_location_detail = self.create_charging_location()
        self.listChargingLocations = list_charging_locations
        self.charging_location_detail = charging_location_detail
        # print(len(self.listChargingLocations))
        # for chargingLocation in self.listChargingLocations:
        #     print(chargingLocation.charging_location)
         

    # Function is for setting nodes' level and setting all targets as covered
    def setLevels(self):
        for node in self.listNodes:
            node.level = -1
        tmp1 = []
        tmp2 = []
        for node in self.baseStation.direct_nodes:
            if node.status == 1:
                node.level = 1
                tmp1.append(node)

        for i in range(len(self.targets_active)):
            self.targets_active[i] = 0

        while True:
            if len(tmp1) == 0:
                break
            # For each node, we set value of target covered by this node as 1
            # For each node, if we have not yet reached its neighbor, then level of neighbors equal this node + 1
            for node in tmp1:
                for target in node.listTargets:
                    self.targets_active[target.id] = 1
                for neighbor in node.neighbors:
                    if neighbor.status == 1 and neighbor.level == -1:
                        tmp2.append(neighbor)
                        neighbor.level = node.level + 1

            # Once all nodes at current level have been expanded, move to the new list of next level
            tmp1 = tmp2[:]
            tmp2.clear()
        return

    def operate(self, t=1):
        
        for node in self.listNodes:
            self.env.process(node.operate(t=t))
        self.env.process(self.baseStation.operate(t=t))
        while True:
            yield self.env.timeout(t / 10.0)
            self.setLevels()
            self.alive = self.check_targets()
            yield self.env.timeout(9.0 * t / 10.0)
            if self.alive == 0 or self.env.now >= self.max_time:
                break         
        return

    # If any target dies, value is set to 0
    def check_targets(self):
        return min(self.targets_active)
    
    def check_nodes(self):
        tmp = 0
        for node in self.listNodes:
            if node.status == 0:
                tmp += 1
        return tmp

    # def create_charging_location(self):
    #
    #     def find_list_node_locations(self):
    #         list_node_locations = []
    #         for node in self.listNodes:
    #             list_node_locations.append(node.location)
    #         return np.array(list_node_locations)
    #
    #     def find_n_clusters_optimal(self):
    #
    #         def check_valid_cluster(clusterer, data, n_clusters, n_nodes):
    #             """
    #             Ensure the maximum of the diameter of the charging location
    #             and that all nodes are included in the clusters.
    #             """
    #             cluster_labels = clusterer.fit_predict(data)
    #             centers = clusterer.cluster_centers_
    #             cluster = [[] for _ in range(n_clusters)]
    #
    #             # Gán các node vào cụm tương ứng
    #             for index_node in range(n_nodes):
    #                 cluster[cluster_labels[index_node]].append(data[index_node])
    #
    #             # Kiểm tra khoảng cách các node trong từng cụm
    #             for index_cluster in range(n_clusters):
    #                 for node in cluster[index_cluster]:
    #                     if math.dist(node, centers[index_cluster]) > 27:
    #                         return False
    #
    #             # Kiểm tra rằng tất cả các node đều nằm trong cụm
    #             total_nodes_in_clusters = sum(len(c) for c in cluster)
    #             if total_nodes_in_clusters != n_nodes:
    #                 return False  # Một số node không được gán vào cụm
    #
    #             return True
    #
    #         list_node_locations = find_list_node_locations(self)
    #         range_n_clusters = np.arange(2, int(len(list_node_locations)))
    #         max_silhouette_score = 0
    #         n_clusters_optimal = 0
    #
    #         for n_clusters in range_n_clusters:
    #             clusterer = KMeans(n_clusters=n_clusters, init="k-means++", algorithm='elkan', n_init="auto")
    #             cluster_labels = clusterer.fit_predict(list_node_locations)
    #             silhouette_avg = silhouette_score(list_node_locations, cluster_labels)
    #
    #             if check_valid_cluster(clusterer, list_node_locations, n_clusters, len(list_node_locations)) is True:
    #                 # if max_silhouette_score < silhouette_avg:
    #                 #     max_silhouette_score = silhouette_avg
    #                 #     n_clusters_optimal = n_clusters
    #                 n_clusters_optimal = n_clusters
    #                 return n_clusters_optimal
    #         return n_clusters_optimal
    #
    #     def define_charging_location(self):
    #         list_node_locations = find_list_node_locations(self)
    #         n_clusters_optimal = find_n_clusters_optimal(self)
    #         clusterer = KMeans(n_clusters=n_clusters_optimal, init="k-means++",  algorithm='elkan')
    #         cluster_labels = clusterer.fit_predict(list_node_locations)
    #         silhouette_avg = silhouette_score(list_node_locations, cluster_labels)
    #         centers = clusterer.cluster_centers_
    #         charging_locations = []
    #         for i in range(0, n_clusters_optimal):
    #             cluster = ChargingLocation(i, centers[i])
    #             charging_locations.append(cluster)
    #         return charging_locations
    #
    #     return define_charging_location(self)
    def create_charging_location(self):

        def find_list_node_locations(self):
            list_node_locations = []
            for node in self.listNodes:
                list_node_locations.append(node.location)
            return np.array(list_node_locations)

        def find_n_clusters_optimal(self):

            def check_valid_cluster(clusterer, data, n_clusters, n_nodes):
                """
                Ensure the maximum of the diameter of the charging location
                and that all nodes are included in the clusters.
                """
                cluster_labels = clusterer.fit_predict(data)
                centers = clusterer.cluster_centers_
                cluster = [[] for _ in range(n_clusters)]

                # Gán các node vào cụm tương ứng
                for index_node in range(n_nodes):
                    cluster[cluster_labels[index_node]].append(data[index_node])

                # Kiểm tra khoảng cách các node trong từng cụm
                for index_cluster in range(n_clusters):
                    for node in cluster[index_cluster]:
                        if math.dist(node, centers[index_cluster]) > 27:
                            return False

                # Kiểm tra rằng tất cả các node đều nằm trong cụm
                total_nodes_in_clusters = sum(len(c) for c in cluster)
                if total_nodes_in_clusters != n_nodes:
                    return False  # Một số node không được gán vào cụm

                return True

            list_node_locations = find_list_node_locations(self)
            range_n_clusters = np.arange(2, int(len(list_node_locations)))
            max_silhouette_score = 0
            n_clusters_optimal = 0

            for n_clusters in range_n_clusters:
                clusterer = KMeans(n_clusters=n_clusters, init="k-means++", algorithm='elkan', n_init="auto")
                cluster_labels = clusterer.fit_predict(list_node_locations)
                silhouette_avg = silhouette_score(list_node_locations, cluster_labels)

                if check_valid_cluster(clusterer, list_node_locations, n_clusters, len(list_node_locations)) is True:
                    n_clusters_optimal = n_clusters
                    return n_clusters_optimal
            return n_clusters_optimal

        def define_charging_location(self):
            list_node_locations = find_list_node_locations(self)
            n_clusters_optimal = find_n_clusters_optimal(self)
            clusterer = KMeans(n_clusters=n_clusters_optimal, init="k-means++", algorithm='elkan')
            cluster_labels = clusterer.fit_predict(list_node_locations)
            silhouette_avg = silhouette_score(list_node_locations, cluster_labels)
            centers = clusterer.cluster_centers_

            charging_locations = []  # Danh sách các vị trí sạc
            cluster_details = []  # Danh sách chi tiết cụm và các node thuộc cụm

            for i in range(0, n_clusters_optimal):
                # Tạo đối tượng ChargingLocation cho cụm
                cluster = ChargingLocation(i, centers[i])
                charging_locations.append(cluster)

                # Thu thập danh sách các node thuộc cụm i
                node_indices = np.where(cluster_labels == i)[0].tolist()  # Lấy ID các node thuộc cụm i
                cluster_details.append({
                    "cluster_id": i,
                    "nodes": node_indices
                })

            return charging_locations, cluster_details

        return define_charging_location(self)