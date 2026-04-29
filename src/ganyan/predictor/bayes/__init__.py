"""Hierarchical Bayesian Plackett-Luce race predictor.

Prototype alongside the LightGBM ensemble. The goal is to determine
whether posterior credible intervals on win probability are better
calibrated than softmax point estimates from the tree-based ranker.
"""
