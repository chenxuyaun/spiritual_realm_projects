"""
Routing module for intelligent workflow selection.

This module provides multiple router implementations:
- RouterV1: Rule-based routing with keyword matching
- RouterV2: Classifier-based routing with TF-IDF + LogisticRegression
- RouterV3: Cost-aware routing with mode features
"""

from mm_orch.routing.router_v1 import RouterV1, RoutingRule
from mm_orch.routing.router_v2 import RouterV2
from mm_orch.routing.router_v3 import RouterV3

__all__ = ['RouterV1', 'RouterV2', 'RouterV3', 'RoutingRule']
