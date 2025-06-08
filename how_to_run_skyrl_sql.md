# How to Run SkyRL-SQL-7B

This guide provides step-by-step instructions to reproduce the results for the SkyRL-SQL-7B model.

## 1. Prerequisites

Before you begin, ensure you have the following installed:

*   **CUDA Toolkit 12.4**: You can download it from the [NVIDIA website](https://developer.nvidia.com/cuda-12-4-0-download-archive).
*   **build-essential**: Required for building some dependencies. You can install it on Debian-based systems using `sudo apt-get install build-essential`.
*   **uv**: A fast Python package installer. Follow the [official installation guide](https://docs.astral.sh/uv/getting-started/installation).
*   **Python 3.12**
*   **Ray 2.43.0**

After installing the prerequisites, configure Ray to use `uv` by setting the following environment variable:

```bash
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
```

## 2. Installation

1.  Clone the repository.
2.  Navigate to the root of the project directory.
3.  Install the required dependencies for the SQL task using `uv`:

    ```bash
    uv run --isolated --extra sql --frozen python -c 'import ray; ray.init(); print("Success!")'
    ```
    This command will create an isolated environment with all the necessary packages for the SQL task.

## 3. Data Preparation

The training process requires two sets of data: `SkyRL-SQL-653-data` and `OmniSQL-datasets`.

1.  **Download SkyRL-SQL-653-data**:
    This dataset is available on Hugging Face. The `run_skyrl_sql.sh` script expects this data to be in a specific directory. By default, it is set to `/SkyRL-SQL-653-data`.

    ```bash
    # Download the dataset
    huggingface-cli download NovaSky-AI/SkyRL-SQL-653-data --local-dir /SkyRL-SQL-653-data --repo-type dataset
    ```

2.  **Download OmniSQL-datasets**:
    This dataset contains the databases required for training. The `run_skyrl_sql.sh` script expects the databases to be at `/data`.

    ```bash
    # Download the dataset from Hugging Face
    huggingface-cli download seeklhy/OmniSQL-datasets data.zip --repo-type dataset --local-dir .

    # Unzip the data to the required directory
    sudo mkdir -p /data
    sudo unzip data.zip -d /data
    rm data.zip
    ```
    The script specifically needs the `synsql-2.5m` and `spider` databases from this dataset.

## 4. Environment Setup

1.  **WANDB API Key**:
    The project uses Weights & Biases for logging. You need to provide your API key. Create a file named `.env.sql` in the root of the project with the following content:

    ```
    WANDB_API_KEY=your_wandb_api_key_here
    ```
    Replace `your_wandb_api_key_here` with your actual Weights & Biases API key.

2.  **Review the Run Script**:
    Open the `examples/sky/sql/run_skyrl_sql.sh` script and review the environment variables at the top. You might need to adjust them based on your setup. Key variables include:
    *   `CUDA_VISIBLE_DEVICES`: Set the GPUs to be used.
    *   `DATA_DIR`: Path to `SkyRL-SQL-653-data`.
    *   `DB_PATH`: Path to the OmniSQL databases.
    *   `BASE_MODEL`: The base model to be used for training.
    *   `CKPT_PATH`: Directory to save checkpoints.
    *   `PROJECT_NAME`: Name of the W&B project.
    *   `EXPERIMENT_NAME`: Name of the W&B experiment.

## 5. Execution

1.  **Start a Ray Cluster**:
    Before running the script, you need to start a Ray cluster. For a local setup, you can simply run:

    ```bash
    ray start --head
    ```
    For a multi-node setup, follow the [Ray documentation](https://docs.ray.io/en/latest/ray-core/starting-ray.html).

2.  **Run the script**:
    Execute the training script from the root of the project:

    ```bash
    bash examples/sky/sql/run_skyrl_sql.sh
    ```
This will start the training process. You can monitor the progress on the console and on Weights & Biases.