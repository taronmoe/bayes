# Exact and Approximate Inference in Bayesian Networks

This project implements and evaluates two probabilistic inference algorithms for
Bayesian Networks: Variable Elimination (exact inference) and Gibbs Sampling
(approximate inference).

The goal is to understand the tradeoffs between accuracy, computational cost,
and scalability as network size and evidence constraints increase.

## Implemented Algorithms
- **Variable Elimination (VE)**  
  Exact inference using factor construction, restriction, multiplication,
  marginalization, and normalization under a chosen elimination order.

- **Gibbs Sampling (GS)**  
  Approximate inference using stochastic sampling with burn-in and thinning
  to estimate marginal probability distributions.

Both algorithms were implemented from scratch in Python without external
inference libraries.

## System Overview
- Parses Bayesian networks from `.bif` files into an internal graph structure
- Applies evidence consistently across inference methods
- Outputs normalized marginal distributions in a standardized CSV format
- Tracks runtime and accuracy metrics for evaluation

## Evaluation
The algorithms were evaluated on three Bayesian networks of increasing size:
- Child (small)
- Insurance (medium)
- Win95pts (large)

Results confirm that Variable Elimination produces near-perfect accuracy but
becomes computationally expensive as network complexity grows. Gibbs Sampling
maintains competitive accuracy while scaling far more efficiently, making it
better suited for large or densely connected networks.

## Key Takeaway
Exact inference is ideal when accuracy is paramount and the network is small.
Approximate inference offers a practical alternative when scalability and
runtime constraints dominate.

## Notes
This project emphasizes correctness, reproducibility, and clear algorithmic
structure, and was developed as part of an Artificial Intelligence course.
