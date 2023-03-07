from torch import Tensor 
from gnn_tracking.models.edge_classifier import symmetrize_edge_weights

def test_symmetrize_edge_weights():
    # Example 1
    edge_indices = Tensor([[1, 2], [2, 1]])
    edge_weights = Tensor([1, 3])
    expected_output = Tensor([2, 2])
    assert (symmetrize_edge_weights(edge_indices, edge_weights) == expected_output).all()

    # Example 2
    edge_indices = Tensor([[1, 2], [3, 4], [2, 1]])
    edge_weights = Tensor([1, 2, 3])
    expected_output = Tensor([2, 2, 2])
    assert (symmetrize_edge_weights(edge_indices, edge_weights) == expected_output).all()

    # Test with empty input
    edge_indices = Tensor([])
    edge_weights = Tensor([])
    expected_output = Tensor([])
    assert (symmetrize_edge_weights(edge_indices, edge_weights) == expected_output).all()

    # Test with single edge
    edge_indices = Tensor([[1, 2]])
    edge_weights = Tensor([1])
    expected_output = Tensor([1])
    assert (symmetrize_edge_weights(edge_indices, edge_weights) == expected_output).all()

    # Test with no symmetrized edges
    edge_indices = Tensor([[1, 2], [3, 4], [5, 6]])
    edge_weights = Tensor([1, 2, 3])
    expected_output = Tensor([1, 2, 3])
    assert (symmetrize_edge_weights(edge_indices, edge_weights) == expected_output).all()

    # Test with all symmetrized edges
    edge_indices = Tensor([[1, 2], [2, 1], [3, 4], [4, 3]])
    edge_weights = Tensor([1, 2, 3, 4])
    expected_output = Tensor([1.5, 1.5, 3.5, 3.5])
    assert (symmetrize_edge_weights(edge_indices, edge_weights) == expected_output).all()

    print("All unit tests pass")


