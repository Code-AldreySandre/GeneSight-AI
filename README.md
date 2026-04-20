# Apollo Solutions: Genetic Syndrome Classification 

## Overview
This repository contains the solution for the Apollo Solutions Machine Learning Developer Test. The objective is to analyze 320-dimensional image embeddings to classify genetic syndromes. 

Moving beyond standard data exploration, this project implements a complete, robust, and reproducible Machine Learning pipeline. It features automated data flattening, dimensionality reduction (t-SNE), a heavily evaluated K-Nearest Neighbors (KNN) classifier, and manual mathematical implementations of key evaluation metrics.

## Key Engineering Decisions
To ensure code quality, maintainability, and execution safety, the following architectural choices were made:
- **Separation of Concerns (SoC):** The pipeline is divided into specialized modules (`data_processor`, `model_pipeline`, `visualizer`, `metrics`).
- **Reproducibility:** Strict dependency management ensures the environment is highly reproducible.
- **Observability:** A centralized custom logger (`src/utils/logger.py`) tracks all execution steps, outputting to both the console and a persistent `.log` file.
- **Reliability:** A dedicated test suite (`pytest`) uses generated mock data to validate core logic without loading the entire dataset.

## Repository Structure

```text
apollo_ml_test/
├── data/     
├── deliverables/ # PDF Reports 
├── logs/    # Auto-generated execution logs
├── results/                          
│   ├── figures/    # Auto-generated t-SNE and ROC Curves
│   └── tables/     # Auto-generated evaluation metrics 
├── src/            # Core pipeline modules
│   ├── utils/logger.py
│   ├── data_processor.py
│   ├── metrics.py
│   ├── model_pipeline.py
│   └── visualizer.py
├── main.py                     # Application entry point
├── requirements.txt            # Dependency list
└── README.md