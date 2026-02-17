"""
File: __init__.py
Module: graph
Description: Graph Neural Network package for sector correlation modelling.
    Provides dynamic sector graph construction and GNN-based sector embedding
    generation used by the Macro agent.
Author: HRL-SARP Framework
"""

from graph.sector_graph import SectorGraph
from graph.gnn_encoder import GNNEncoder

__all__ = [
    "SectorGraph",
    "GNNEncoder",
]
