import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_classes):
        super(GCN, self).__init__()
        torch.manual_seed(1234)
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)  
        self.classifier = Linear(output_dim, num_classes)  

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.tanh() 
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()  

        out = self.classifier(h)

        return out, h