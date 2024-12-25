import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from scipy.spatial.distance import euclidean
import math
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
        hidden_dim = net.hidden_dim
        output_dim = net.output_dim

        GNN_model = GCN(num_features, hidden_dim, output_dim, num_classes)
        optimizer = torch.optim.Adam(GNN_model.parameters(), lr=0.0005)

        for epoch in range(5000):
            loss = GraphRepresentation.train(GNN_model, optimizer, data)
            if epoch % 100 == 0:
                acc = GraphRepresentation.test(GNN_model, data)
                print(f'Epoch {epoch}, Loss: {loss:.4f}, Test Accuracy: {acc:.4f}')

        torch.save(GNN_model.state_dict(), model_path)
        print('Model saved successfully!')

        GNN_model.eval()
        with torch.no_grad():
            _, embeddings = GNN_model(data.x, data.edge_index)
        return embeddings

    @staticmethod
    def train(model, optimizer, data):
        model.train()
        optimizer.zero_grad()

        out, embeddings = model(data.x, data.edge_index)

        # Primary loss: classification
        classification_loss = F.cross_entropy(out, data.y)

        # Additional loss: Contrastive Loss to encourage diverse embeddings
        contrastive_loss = GraphRepresentation.compute_contrastive_loss(embeddings, data.y, margin=1.0)

        # Combined loss
        loss = classification_loss + 0.1 * contrastive_loss  # Weight the contrastive loss

        loss.backward()
        optimizer.step()
        return loss.item()

    @staticmethod
    def compute_contrastive_loss(embeddings, labels, margin=1.0):
        pairwise_distances = torch.cdist(embeddings, embeddings)
        labels = labels.unsqueeze(1)
        label_diff = labels != labels.t()
        loss_positive = pairwise_distances[~label_diff].pow(2).mean()  # Pull similar labels closer
        loss_negative = F.relu(margin - pairwise_distances[label_diff]).pow(2).mean()  # Push different labels apart
        return loss_positive + loss_negative

    @staticmethod
    def test(model, data):
        model.eval()
        out, _ = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        correct = pred.eq(data.y).sum().item()
        acc = correct / data.num_nodes
        print(f'Số lượng node đoán đúng: {correct}')
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
        """Create node features without using energy."""

        def make_node_features(node):
            location = node.location
            num_target = len(node.listTargets)
            num_neighbors = len(node.neighbors)

            # Thêm nhiễu với kích thước phù hợp
            feature_vector = torch.tensor([*location, num_target, num_neighbors], dtype=torch.float)
            noise = torch.normal(0, 0.01, size=feature_vector.shape)  # Nhiễu có cùng kích thước
            node_feature = feature_vector + noise
            return node_feature

        vertices = [make_node_features(node) for node in net.listNodes]
        location = net.baseStation.location
        vertices.append(torch.tensor([*location, 0, 0], dtype=torch.float))  # Base station
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
        for i, node in enumerate(net.listNodes):
            min_distance = math.inf
            label = 0
            for j, loc in enumerate(net.listChargingLocations):
                distance = euclidean(node.location, loc.charging_location)
                if distance < min_distance:
                    label = loc.id
                    min_distance = distance
            labels.append(label)
        labels.append(len(net.listChargingLocations))
        return torch.tensor(labels, dtype=torch.long)


# import torch
# import torch.nn.functional as F
# from torch_geometric.data import Data
# from scipy.spatial.distance import euclidean
# import math
# import os
# from sklearn.metrics import f1_score

# from physical_env.network.Network import Network
# from physical_env.network.Node import Node
# from physical_env.network.BaseStation import BaseStation
# from rl_env.state_representation.GAT import GAT

# root_dir = os.getcwd()

# class GraphRepresentation:
#     def __init__(self):
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         self.GNN_model = None

#     @staticmethod
#     def get_graph_representation(net: Network, load_model=False):
#         model_path = os.path.join(root_dir, "rl_env", "graph_model_gat.pth")
#         data = GraphRepresentation.create_graph(net)
#         num_features = data.x.size(1)
#         num_classes = len(net.listChargingLocations) + 1
#         hidden_dim = net.hidden_dim
#         output_dim = net.output_dim

#         GNN_model = GAT(num_features, hidden_dim, output_dim, num_classes).to('cuda' if torch.cuda.is_available() else 'cpu')

#         # Load the model if `load_model` is set to True
#         if load_model:
#             GNN_model.load_state_dict(torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu'))
#             GNN_model.eval()
#         else:
#             optimizer = torch.optim.Adam(GNN_model.parameters(), lr=0.001, weight_decay=1e-4)
#             early_stopping = 0
#             best_loss = float('inf')

#             for epoch in range(500):
#                 loss, f1 = GraphRepresentation.train(GNN_model, optimizer, data)
#                 if epoch % 50 == 0:
#                     acc = GraphRepresentation.test(GNN_model, data)
#                     print(f"Epoch {epoch}, Loss: {loss:.4f}, Test Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")

#                 # Early stopping
#                 if loss < best_loss:
#                     best_loss = loss
#                     early_stopping = 0
#                     torch.save(GNN_model.state_dict(), model_path)
#                 else:
#                     early_stopping += 1

#                 if early_stopping >= 50:  # Stop early if no improvement
#                     print("Early stopping triggered.")
#                     break

#             print("Model training completed and saved!")

#         GNN_model.eval()
#         with torch.no_grad():
#             _, embeddings = GNN_model(data.x, data.edge_index)
#         return embeddings

#     @staticmethod
#     def train(model, optimizer, data):
#         model.train()
#         optimizer.zero_grad()

#         out, embeddings = model(data.x, data.edge_index)

#         # Compute class weights
#         class_counts = torch.bincount(data.y)
#         class_weights = 1.0 / (class_counts.float() + 1e-5)
#         class_weights = class_weights / class_weights.sum()

#         # Classification loss with weighted cross entropy
#         classification_loss = F.cross_entropy(out, data.y, weight=class_weights.to(out.device))

#         # Backward pass
#         loss = classification_loss
#         loss.backward()
#         optimizer.step()

#         # Calculate F1-Score
#         pred = out.argmax(dim=1).cpu()
#         f1 = f1_score(data.y.cpu(), pred, average='weighted')

#         return loss.item(), f1

#     @staticmethod
#     def test(model, data):
#         model.eval()
#         out, _ = model(data.x, data.edge_index)
#         pred = out.argmax(dim=1)
#         correct = pred.eq(data.y).sum().item()
#         acc = correct / data.num_nodes
#         return acc

#     @staticmethod
#     def create_graph(net: Network) -> Data:
#         node_features = GraphRepresentation.create_vertices(net)
#         edges = GraphRepresentation.create_edges(net)
#         labels = GraphRepresentation.create_labels(net)

#         data = Data(x=node_features, edge_index=edges, y=labels)
#         return data

#     @staticmethod
#     def create_vertices(net: Network):
#         """
#         Tích hợp thông tin năng lượng vào đặc trưng của node.
#         """
#         def make_node_features(node):
#             location = node.location
#             num_target = len(node.listTargets)
#             num_neighbors = len(node.neighbors)
#             energy_level = node.energy  # Thêm thông tin năng lượng

#             feature_vector = torch.tensor([*location, num_target, num_neighbors, energy_level], dtype=torch.float)
#             noise = torch.normal(0, 0.01, size=feature_vector.shape)
#             node_feature = feature_vector + noise
#             return node_feature

#         vertices = [make_node_features(node) for node in net.listNodes]
#         location = net.baseStation.location
#         vertices.append(torch.tensor([*location, 0, 0, 1.0], dtype=torch.float))  # Base station
#         return torch.stack(vertices)

#     @staticmethod
#     def create_edges(net: Network):
#         for node in net.listNodes:
#             node.probe_neighbors()
#         net.baseStation.probe_neighbors()
#         net.setLevels()

#         edges = []
#         for node in net.listNodes:
#             neighbor = node.find_receiver()
#             if neighbor.__class__.__name__ == "Node":
#                 edges.append([node.id, neighbor.id])
#             elif euclidean(node.location, net.baseStation.location) <= node.com_range:
#                 edges.append([node.id, len(net.listNodes)])
#         edges = torch.tensor(edges, dtype=torch.long).t()
#         return edges

#     @staticmethod
#     def create_labels(net: Network):
#         labels = []
#         for i, node in enumerate(net.listNodes):
#             min_distance = math.inf
#             label = 0
#             for j, loc in enumerate(net.listChargingLocations):
#                 distance = euclidean(node.location, loc.charging_location)
#                 if distance < min_distance:
#                     label = loc.id
#                     min_distance = distance
#             labels.append(label)
#         labels.append(len(net.listChargingLocations))
#         return torch.tensor(labels, dtype=torch.long)
