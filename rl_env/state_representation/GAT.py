import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
class GAT(torch.nn.Module):
    """
    Graph Attention Network với 2 tầng và cơ chế attention.
    """
    def __init__(self, num_features, hidden_dim, output_dim, num_classes):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_features, hidden_dim, heads=4, concat=True)  # Attention với nhiều head
        self.conv2 = GATConv(hidden_dim * 4, output_dim, heads=1, concat=False)
        self.classifier = torch.nn.Linear(output_dim, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)  # Hàm kích hoạt ELU
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        out = self.classifier(x)
        return out, x  # Trả về cả output và embeddings
