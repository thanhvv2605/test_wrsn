
import torch
from sympy.printing.tree import print_node
from torch_geometric.data import Data
import numpy as np
from scipy.spatial.distance import euclidean
import math 
import torch.nn.functional as F
import os

from physical_env.network.Network import Network
from physical_env.network.Node import Node
from physical_env.network.BaseStation import BaseStation
from rl_env.state_representation.GNN import GCN
root_dir = os.getcwd()
class GraphRepresentation:
    def __init__(self):
        self.GNN_model = None
    
    @staticmethod
    def get_graph_representation(net: Network, load_model=False):
        model_path = os.path.join(root_dir, "rl_env", "grap_model.pth")
        data = GraphRepresentation.create_graph(net)
        num_features = data.x.size(1)
        num_classes = len(net.listChargingLocations) + 1
        hidden_dim = 512
        output_dim = 83  # chưa tối ưu, lấy bằng số node trong đồ thị

        GNN_model = GCN(num_features, hidden_dim, output_dim, num_classes)

        # Load the model if `load_model` is set to True
        if load_model:
            GNN_model.load_state_dict(torch.load(model_path))
            GNN_model.eval()  # Set to evaluation mode for inference
        else:
            optimizer = torch.optim.Adam(GNN_model.parameters(), lr=0.0005)  # Define optimizer.

            # Train the model
            for epoch in range(1000):
                loss = GraphRepresentation.train(GNN_model, optimizer, data)
                if epoch % 10 == 0:
                    acc = GraphRepresentation.test(GNN_model, data)
                    print(f'Epoch {epoch}, Loss: {loss:.4f}, Test Accuracy: {acc:.4f}')

            # Save the model after training
            torch.save(GNN_model.state_dict(), model_path)

        # Return embeddings after training or loading
        GNN_model.eval()  # Set to evaluation mode
        with torch.no_grad():
            _, embeddings = GNN_model(data.x, data.edge_index)

        # print(embeddings)
        return embeddings


    @staticmethod
    def train(model, optimizer, data):
        model.train()
        optimizer.zero_grad()  # Reset gradient
        out, h = model(data.x, data.edge_index)  # Đầu ra của mô hình và nhúng node
        loss = F.cross_entropy(out, data.y)  # Tính toán hàm mất mát dựa trên nhãn và dự đoán
        loss.backward()  # Lan truyền ngược
        optimizer.step()  # Cập nhật trọng số
        return loss.item()
    
    @staticmethod
    def test(model, data):
        model.eval()
        out, _ = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)  # Dự đoán nhãn
        correct = pred.eq(data.y).sum().item()  # Số lượng dự đoán đúng
        acc = correct / data.num_nodes  # Độ chính xác
        print(f'Số lượng node đoán đúng: {correct:.4f}')
        return acc
        
    @staticmethod
    def create_graph(net: Network) -> Data:
        node_features = GraphRepresentation.create_vertices(net)
        edges = GraphRepresentation.create_edges(net)
        labels = GraphRepresentation.create_labels(net)

        data = Data(x=node_features, edge_index=edges, y=labels)    
        return data
    
    @staticmethod
    def create_vertices(net: Network):
        """_summary_
        This function aims to create vertices, including sensor nodes and the base station
        Args:
            net (Network): _description_
        """
        
        def make_node_features(node):
            """_summary_
            This function aims to transform sensor characteristics into node features in graph, by using tensor torch
            Args:
                node (_type_): _description_

            Returns:
                _type_: tensor
            """
            location = node.location
            num_target = len(node.listTargets)
            node_feature = torch.tensor([*location, num_target], dtype=torch.float)
            return node_feature
        
        vertices = []
        for node in net.listNodes:
            node_feature = make_node_features(node)
            vertices.append(node_feature)
        # add base station as the last node of the graph
        location = net.baseStation.location
        vertices.append(torch.tensor([*location, 0], dtype=torch.float))
        return torch.stack(vertices)
    
    @staticmethod
    def create_edges(net: Network):
        for node in net.listNodes:
            node.probe_neighbors()
        net.baseStation.probe_neighbors()
        net.setLevels()
        edges = []
        for node in net.listNodes:
            neighbor = node.find_receiver()
            if neighbor.__class__.__name__ == "Node":
                edges.append([node.id, neighbor.id])
            elif euclidean(node.location, net.baseStation.location) <= node.com_range:
                edges.append([node.id, len(net.listNodes)])
        edges = torch.tensor(edges, dtype=torch.long).t()
        return edges
    
    @staticmethod
    def create_labels(net: Network):
        labels = []
        for i in range(0, len(net.listNodes)):
            min_distance = math.inf
            label = 0
            for j in range(0, len(net.listChargingLocations)):
                if euclidean(net.listNodes[i].location, net.listChargingLocations[j].charging_location) < min_distance:
                    label = net.listChargingLocations[j].id
                    min_distance = euclidean(net.listNodes[i].location, net.listChargingLocations[j].charging_location)
            labels.append(label)
        labels.append(len(net.listChargingLocations))
        return torch.tensor(labels, dtype=torch.long)