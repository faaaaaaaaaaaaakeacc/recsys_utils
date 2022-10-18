import pytest
import numpy as np
import tf_geometric as tfg
from recsys.models.ngcf import NGCFHead
from recsys.models.ngcf import NGCFModule


@pytest.fixture
def sample_ui_graph():
    """Sample tf graph with 3 users and 2 items."""
    x = np.arange(5)
    edge_index = np.array([
        [0, 0, 1, 2],
        [3, 4, 4, 3]
    ])
    edge_weight = np.array([0.9, 0.8, 0.1, 0.2]).astype(np.float32)
    graph = tfg.Graph(x=x, edge_index=edge_index, edge_weight=edge_weight).to_directed()
    mask_users = np.array([True, True, True, False, False])
    return graph, mask_users


@pytest.fixture
def sample_batch(sample_ui_graph):
    graph, mask_users = sample_ui_graph
    return tfg.BatchGraph.from_graphs([graph, graph, graph, graph, graph])


def test_head(sample_batch):
    graph_batch = sample_batch
    model = NGCFHead(num_ids=5, embedding_dim=20, hidden_dim=32, num_layers=3, dropout=0.1)
    x, edge_index, edge_weight = graph_batch.x, graph_batch.edge_index, graph_batch.edge_weight
    output = model(x, edge_index, edge_weight)
    assert output.shape == (25, 32)


def test_model(sample_ui_graph):
    graph_batch, users_mask = sample_ui_graph
    model = NGCFModule(num_ids=5, embedding_dim=20, hidden_dim=32, num_layers=3, dropout=0.1)
    output = model(graph_batch, users_mask, 0, 1).numpy()
    assert 0 < output
    assert output < 1
