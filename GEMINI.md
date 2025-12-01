# BPINNs - Bayesian Physics-Informed Neural Networks

## Project Overview
**BPINNs** is a comprehensive library for Physics-Informed Deep Learning under uncertainty. It implements Bayesian Physics-Informed Neural Networks (B-PINNs) to solve forward and inverse Partial Differential Equation (PDE) problems.

**Key Features:**
*   **Algorithms:** Supports multiple training algorithms including:
    *   **ADAM:** Adaptive Moment Estimation (standard optimization).
    *   **HMC:** Hamiltonian Monte Carlo.
    *   **SVGD:** Stein Variational Gradient Descent.
    *   **VI:** Variational Inference.
*   **Problem Support:** Includes implementations for Laplace (1D/2D), Oscillator, and Regression problems.
*   **Architecture:** Modular design separating data generation, network architecture, algorithms, and post-processing.

## Tech Stack
*   **Language:** Python 3.10+
*   **Deep Learning Framework:** TensorFlow 2.9.1 (CPU), Keras 2.9.0
*   **Scientific Computing:** NumPy, SciPy
*   **Visualization:** Matplotlib

## Directory Structure
*   **`config/`**: Contains `.json` configuration files defining parameters for various test cases.
*   **`data/`**: Stores datasets (input domains, solutions, parametric fields) as `.npy` files.
*   **`outs/`**: Output directory for experiment results, including:
    *   `log/`: Loss history and experiment summaries.
    *   `plot/`: Generated plots.
    *   `thetas/`: Saved network parameters.
    *   `values/`: Computed solutions.
*   **`src/`**: Source code directory.
    *   **`main.py`**: The main executable script for running experiments.
    *   `algorithms/`: Implementations of training algorithms (ADAM, HMC, SVGD, VI).
    *   `networks/`: Bayesian Neural Network architectures (`BayesNN`, `CoreNN`, `Theta`).
    *   `equations/`: PDE definitions and differential operators.
    *   `setup/`: Argument parsing and parameter management.

## Setup & Installation

### Prerequisites
*   Python 3.10 or higher
*   `virtualenv` or `conda`

### Installation Steps
1.  **Create a Virtual Environment:**
    ```bash
    # Using virtualenv
    virtualenv venv
    source venv/bin/activate

    # OR using conda
    conda create -n bpinns python=3.10
    conda activate bpinns
    ```
2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage
The project is executed via the `src/main.py` script. You can configure experiments using command-line arguments or by specifying a configuration file.

### Running with a Configuration File
The easiest way to run an experiment is to use a pre-defined JSON configuration file from the `config/` directory.

```bash
python src/main.py --config <config_name>
```
*   `<config_name>`: The name of the file in `config/` (e.g., `HMC_lap_sin` for `config/best_models/HMC_lap_sin.json` or just the filename if it's in the root of `config`). *Note: The code searches for the config file.*

### Running with Command Line Arguments
You can also specify parameters directly via the CLI, which overrides config file settings.

```bash
python src/main.py --problem laplace1D --case_name cos --method HMC
```

**Key Arguments:**
*   `--config`: JSON configuration file name.
*   `--problem`: Physical problem to solve (e.g., `laplace1D`, `laplace2D`).
*   `--case_name`: Specific data case (e.g., `cos`).
*   `--method`: Training algorithm (`HMC`, `SVGD`, `VI`).
*   `--epochs`: Number of training epochs.
*   `--gen_flag`: Set to `True` to generate new datasets.
*   `--save_flag`: Set to `True` to save results.

### Example
To run the Hamiltonian Monte Carlo method on the 1D Laplace problem with a sine solution:

```bash
python src/main.py --config HMC_lap_sin
```

## Development Conventions
*   **Code Style:** Follows standard Python practices.
*   **Configuration:** Uses a `Param` class to merge arguments from the CLI and JSON files.
*   **Data Handling:** Datasets are generated or loaded as `.npy` files. The `main_data.py` script can be used for independent data generation.
