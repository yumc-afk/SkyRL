Installation
============

Requirements
------------
- CUDA version >=12.4
- `uv <https://docs.astral.sh/uv/>`_

We use `uv <https://docs.astral.sh/uv/>`_ to manage dependencies. We also make use of the `uv` and `ray` integration to manage dependencies for ray workers. 

If you're running on an existing Ray cluster, make sure to use Ray 2.44.0 and Python 3.12.


Docker (recommended)
---------------------

We provide a docker image with the base dependencies ``sumanthrh/skyrl-train-ray-2.44.0-py3.12-cu12.4`` for quick setup. 

1. Make sure to have `NVIDIA Container Runtime <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>`_ installed.

2. You can launch the container using the following command:

.. code-block:: bash

    docker run -it  --runtime=nvidia --gpus all  --name skyrl-train sumanthrh/skyrl-train-ray-2.44.0-py3.12-cu12.4 /bin/bash

3. Inside the launched container, setup the latest version of the project:

.. code-block:: bash

    git clone https://github.com/novasky-ai/SkyRL.git
    cd SkyRL/skyrl-train


That is it! You should now be to able to run our :doc:`quick start example <quickstart>`.

Install without Dockerfile
--------------------------

For installation without the Dockerfile, make sure you meet the pre-requisities: 

- CUDA 12.4
- `uv <https://docs.astral.sh/uv/>`_
- `ray <https://docs.ray.io/en/latest/>`_ 2.44.0

All project dependencies are managed by `uv`.

Clone the repo and `cd` into the `skyrl` directory:

.. code-block:: bash

    git clone https://github.com/novasky-ai/SkyRL.git
    cd SkyRL/skyrl-train 

Base environment
~~~~~~~~~~~~~~~~

We recommend having a base virtual environment for the project.

With ``uv``: 

.. code-block:: bash

    uv venv --python 3.12 <path_to_venv>

If ``<path_to_venv>`` is not specified, the virtual environment will be created in the current directory at ``.venv``.

.. tip::
    Because of how Ray ships content in the `working directory <https://docs.ray.io/en/latest/ray-core/handling-dependencies.html>`_, we recommend that the base environment is created *outside* the package directory. For example, ``~/venvs/skyrl-train``.

Then activate the virtual environment and install the dependencies.

.. code-block:: bash

    source <path_to_venv>/bin/activate
    uv sync --active --extra vllm

With ``conda``: 

.. code-block:: bash

    conda create -n skyrl-train python=3.12
    conda activate skyrl-train

After activating the virtual environment, make sure to configure Ray to use `uv`:

.. code-block:: bash

    export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
    # or add to your .bashrc
    # echo 'export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook' >> ~/.bashrc


Initialize Ray cluster
----------------------

Finally, you can initialize a Ray cluster using the following command (for single-node):

.. code-block:: bash

    ray start --head 
    # sanity check
    # ray status


.. note::
    For multi-node clusters, please follow the `Ray documentation <https://docs.ray.io/en/latest/cluster/getting-started.html>`_.

You should now be to able to run our :doc:`quick start example <quickstart>`.


Development 
-----------

For development, make sure to use ``--extra dev`` so that the dev dependencies are included.

.. code-block:: bash

    uv run --isolated --extra dev pytest -s tests/cpu