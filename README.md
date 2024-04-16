# WeatherPrediction450
Used as a means of predicting weather patterns for our intro to ai class



# Setting Up a Python Virtual Environment for Our Project

## Introduction

This guide will walk you through the process of setting up a Python virtual environment. A virtual environment helps to manage dependencies for your project, ensuring that the libraries and versions you use do not conflict with those of other Python projects.

## Requirements

You will need Python installed on your machine. Python 3.6 or higher is recommended. You can download Python from [python.org](https://www.python.org/downloads/).

## Steps to Set Up the Virtual Environment

### 1. Install Python

Ensure Python is installed and accessible from your command line. You can check this by running:

\```bash
python --version
\```
or
\```bash
python3 --version
\```

### 2. Create a Virtual Environment

Navigate to your project directory where you want the virtual environment to be set up:

\```bash
cd path/to/your/project
\```

Create the virtual environment using the following command:

\```bash
python -m venv myenv
\```
or
\```bash
python3 -m venv myenv
\```

*`myenv` is the name of the virtual environment and can be anything you choose.*

### 3. Activate the Virtual Environment

Activating the virtual environment will ensure that all Python and pip commands will use the environment's packages and settings.

#### On Windows:

\```bash
myenv\Scripts\activate
\```

#### On macOS and Linux:

\```bash
source myenv/bin/activate
\```

### 4. Install Required Packages

Once the environment is activated, install the required packages using the `requirements.txt` file provided by the project lead.

First, ensure your `requirements.txt` file is in the current directory (or provide the path to it), then run:

\```bash
pip install -r requirements.txt
\```

## Working with the Virtual Environment

- **Activating the environment**: Each time you work on the project, activate the environment using the commands shown in step 3.
- **Deactivating the environment**: When you are done, you can deactivate the environment by simply running:

\```bash
deactivate
\```

## Conclusion

You are now set up with a Python virtual environment specifically for our AI weather predictive model project. This setup helps maintain a clean working environment for Python projects and ensures compatibility and reproducibility of our project's results.
