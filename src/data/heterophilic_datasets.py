"""
Heterophilic graph dataset loaders.

Heterophilic graphs have low homophily - connected nodes tend to have different labels.
These datasets are challenging for standard message-passing GNNs.
"""

import torch
from torch_geometric.datasets import (
    WikipediaNetwork,  # Chameleon, Squirrel
    Actor,  # Actor network
    WebKB,  # Cornell, Texas, Wisconsin
)
from torch_geometric.utils import to_undirected, remove_self_loops
import os


def load_heterophilic_dataset(name, root='./data'):
    """
    Load heterophilic datasets.

    Args:
        name: Dataset name. Options:
            - 'chameleon': Wikipedia page-page network (2,277 nodes, 5 classes)
            - 'squirrel': Wikipedia page-page network (5,201 nodes, 5 classes)
            - 'actor': Actor co-occurrence network (7,600 nodes, 5 classes)
            - 'cornell': WebKB Cornell (183 nodes, 5 classes)
            - 'texas': WebKB Texas (183 nodes, 5 classes)
            - 'wisconsin': WebKB Wisconsin (251 nodes, 5 classes)
        root: Data directory

    Returns:
        dataset: PyG dataset
    """
    name = name.lower()

    if name == 'chameleon':
        dataset = WikipediaNetwork(root=os.path.join(root, 'WikipediaNetwork'),
                                   name='chameleon')
    elif name == 'squirrel':
        dataset = WikipediaNetwork(root=os.path.join(root, 'WikipediaNetwork'),
                                   name='squirrel')
    elif name == 'actor':
        dataset = Actor(root=os.path.join(root, 'Actor'))
    elif name == 'cornell':
        dataset = WebKB(root=os.path.join(root, 'WebKB'), name='Cornell')
    elif name == 'texas':
        dataset = WebKB(root=os.path.join(root, 'WebKB'), name='Texas')
    elif name == 'wisconsin':
        dataset = WebKB(root=os.path.join(root, 'WebKB'), name='Wisconsin')
    else:
        raise ValueError(f"Unknown heterophilic dataset: {name}")

    return dataset


def compute_homophily_ratio(data):
    """
    Compute edge homophily ratio: fraction of edges connecting same-class nodes.

    Args:
        data: PyG Data object with edge_index and y

    Returns:
        homophily: Float in [0, 1]. Higher = more homophilic.
    """
    edge_index = data.edge_index
    y = data.y

    # Remove self-loops
    edge_index, _ = remove_self_loops(edge_index)

    # Get labels of source and target nodes
    src_labels = y[edge_index[0]]
    dst_labels = y[edge_index[1]]

    # Count same-class edges
    same_class = (src_labels == dst_labels).sum().item()
    total_edges = edge_index.shape[1]

    homophily = same_class / total_edges if total_edges > 0 else 0.0

    return homophily


def get_dataset_stats(dataset):
    """
    Get statistics for a dataset.

    Returns:
        dict: Statistics including nodes, edges, classes, features, homophily
    """
    data = dataset[0]

    stats = {
        'num_nodes': data.num_nodes,
        'num_edges': data.edge_index.shape[1],
        'num_classes': dataset.num_classes,
        'num_features': dataset.num_features,
        'homophily': compute_homophily_ratio(data),
    }

    return stats


# Dataset information
HETEROPHILIC_DATASETS = {
    'chameleon': {
        'nodes': 2277,
        'edges': 36101,
        'classes': 5,
        'features': 2325,
        'homophily': 0.23,  # Low homophily
        'description': 'Wikipedia page-page network about chameleons',
    },
    'squirrel': {
        'nodes': 5201,
        'edges': 217073,
        'classes': 5,
        'features': 2089,
        'homophily': 0.22,  # Low homophily
        'description': 'Wikipedia page-page network about squirrels',
    },
    'actor': {
        'nodes': 7600,
        'edges': 33544,
        'classes': 5,
        'features': 931,
        'homophily': 0.22,  # Low homophily
        'description': 'Actor co-occurrence network from film-actor graphs',
    },
    'cornell': {
        'nodes': 183,
        'edges': 298,
        'classes': 5,
        'features': 1703,
        'homophily': 0.30,  # Low homophily
        'description': 'WebKB Cornell university webpage dataset',
    },
    'texas': {
        'nodes': 183,
        'edges': 325,
        'classes': 5,
        'features': 1703,
        'homophily': 0.11,  # Very low homophily
        'description': 'WebKB Texas university webpage dataset',
    },
    'wisconsin': {
        'nodes': 251,
        'edges': 515,
        'classes': 5,
        'features': 1703,
        'homophily': 0.21,  # Low homophily
        'description': 'WebKB Wisconsin university webpage dataset',
    },
}


if __name__ == '__main__':
    """Test heterophilic dataset loading."""
    print("Testing Heterophilic Dataset Loaders")
    print("=" * 60)

    for name in HETEROPHILIC_DATASETS.keys():
        print(f"\nLoading {name}...")
        try:
            dataset = load_heterophilic_dataset(name)
            stats = get_dataset_stats(dataset)

            print(f"  Nodes: {stats['num_nodes']}")
            print(f"  Edges: {stats['num_edges']}")
            print(f"  Classes: {stats['num_classes']}")
            print(f"  Features: {stats['num_features']}")
            print(f"  Homophily: {stats['homophily']:.3f}")
            print(f"  ✅ Loaded successfully")
        except Exception as e:
            print(f"  ❌ Error: {e}")

    print("\n" + "=" * 60)
    print("All heterophilic datasets tested!")
