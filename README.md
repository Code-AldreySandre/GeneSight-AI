# GeneSight-AI 
### Genetic Syndrome Classifier and Analytics Pipeline

GeneSight-AI is a professional-grade Machine Learning pipeline designed to classify rare genetic syndromes from high-dimensional image embeddings (320-D). Built with a focus on modularity, scalability, and clinical interpretability, this project demonstrates a complete AI Engineering workflow, from automated Exploratory Data Analysis (EDA) to an interactive diagnostic dashboard.

---

## Features

- **Decoupled Architecture:** Strict separation of concerns between data processing, statistical analysis, visualization, and model training.
- **Automated EDA:** A dedicated Python script for data quality auditing, outlier detection via L2 Norm analysis, and class imbalance reporting.
- **Robust Classification:** KNN implementation using Stratified 10-Fold Cross-Validation, evaluating both Euclidean and Cosine distances.
- **Interactive Dashboard:** A Streamlit-based UI for real-time data ingestion, cluster visualization (t-SNE), and performance monitoring.


## Repository Structure

```text
GENESIGHT-AI/
├── data/     
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
├── main.py  # Application entry point
├── app.py
├── poetry.lock
├── pyproject.toml    # Dependency list
├── requirements.txt  # Dependency list
└── README.md
```



## Installation & Setup

This project supports both standard `pip` (ideal for quick evaluations) and `Poetry` (ideal for strict dependency management). Choose the method that best fits your environment.

### Method 1: Standard Python (venv + pip)
*Recommended for quick setup and broad compatibility.*

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/GeneSight-AI.git

   cd GeneSight-AI
   ```

2. **Create a virtual environment:**
   ```bash
    # On Linux/macOS
    python3 -m venv .venv
    ```
    ```bash
    # On Windows
    python -m venv .venv
    ```

3. **Activate the virtual environment:**
    ```bash
    # On Linux/macOS
    source .venv/bin/activate
    ```

    ```bash
    # On Windows
    .venv\Scripts\activate
    ```

4. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

### Method 2: Modern Workflow (Poetry)
*Recommended for development and exact reproducibility..*

1. **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/GeneSight-AI.git

    cd GeneSight-AI
    ```

2. **Install Poetry (if not already installed):**


    Using the official secure installer:
    ```bash
    curl -sSL [https://install.python-poetry.org](https://install.python-poetry.org) | python3 -
    ```

    Alternative: If your system uses pipx, you can run:
    
    ```bash
     pipx install poetry
    ```
3. **Install dependencies (automatically creates the environment):**

    ```bash
    poetry install
    ```

4. **Activate the Poetry shell:**

    ```bash
    poetry shell
    ```


## Execution Workflow (Decoupled Architecture)

GeneSight-AI was built with a decoupled architecture, allowing it to be used via the terminal for automated batch processing, or via a web dashboard for interactive visual analysis.

### 1. Via Terminal: The Processing Engine (main.py)

This mode focuses on Backend / Batch Processing. It is ideal for CI/CD pipelines or running automated experiments without a graphical interface.


Command: 

    python main.py

Under the Hood:

- **Orchestration:** The script loads the raw data and applies L2 Normalization (essential for stabilizing distances in high-dimensional vector spaces).

- **Training:** Executes the KNN model using Stratified 10-Fold Cross-Validation, ensuring performance metrics are robust and statistically sound.

- **Silent Export:** Instead of blocking the server with pop-up windows, it silently saves all generated plots (PNG) and metrics tables (CSV) directly into the results/ directory.

Ideal for automated pipelines.

### 2. Via Dashboard: The Interactive UI (app.py)

This mode transforms the core logic into a HealthTech Product. It empowers end-users (without programming knowledge) to interact directly with the AI models.

Command:  

    streamlit run app.py

Under the Hood:

- **Web Server:** Launches a local Streamlit web server in your browser.

- **Dynamic Ingestion:** Allows the user to upload new .p embedding files and process them in real-time.

- **Multidimensional Visualization:** Renders interactive t-SNE maps and plots on the fly, showing how the AI clusters the genetic traits in a 2D space

Its ideal for clinicians, geneticists, and product demonstrations.

#### User Guide:  Interpreting the Results

When using the interactive Dashboard (app.py), navigate through the three main tabs to understand the data:

- **Exploratory Data:** Check the Outlier Detection plot. It demonstrates how L2 Normalization successfully treated anomalous facial vectors before model training.

- **Spatial Clusters:** Explore the t-SNE map. Look for distinct color groupings—the more isolated a cluster is, the better the model naturally separates that specific genetic syndrome.

- **Performance Results:** Analyze the hard metrics. Pay special attention to the Top-5 Accuracy, which is crucial in clinical genetics as it allows the tool to act as a "triage assistant", suggesting the 5 most probable syndromes to the physician.