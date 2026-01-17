# Lithology Identification from Well Logs via Meta-Information Tensors and Quality-Aware Weighting

This repository contains the core implementation of the Robust Feature Engineering (RFE) framework proposed in the paper: "Lithology Identification from Well Logs via Meta-Information Tensors and Quality-Aware Weighting", submitted to Big Data and Cognitive Computing.

##  Data & Code Availability Policy
Due to project confidentiality and proprietary restrictions regarding the full workflow, we are unable to release the complete executable pipeline. However, to facilitate reproducibility and scientific transparency, we provide the source code for the core Feature Engineering module below.

This module implements the key contributions of our work:
1.Linear-time Window Aggregation (O(N))
2.Meta-Information Tensor Construction (handling missing patterns)

## Core Implementation: Robust Feature Engineering

The following Python code demonstrates how to transform raw well logs into the high-dimensional feature space used in our XGBoost model.

