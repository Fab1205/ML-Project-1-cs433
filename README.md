
# Project Title: Cardiovascular Disease Prediction
*Team Members: Fabian Roulin, Léa Goffinet, Samuel Mouny*

## Project Overview
This project aims to predict the likelihood of cardiovascular disease based on individual health factors. The analysis involves data exploration, feature engineering, and implementing machine learning algorithms for binary classification. This work is part of the **EPFL Machine Learning Course, Fall 2024**.

## Repository Structure
The repository is organized as follows:
- **data/**: Placeholder for dataset files (`data/raw/` for raw data).
- **doc/**: Contains project description and data codebook.
- **notebooks/**: Jupyter notebooks for data exploration, processing, model training, etc.
- **report/**: Final report (LaTeX format) and accompanying figures.
- **src/**: Custom package with modules:
  - `data_exploration`, `data_loading`, `data_processing`
  - `models`, `train_pipeline`, `utils`
- **tests/**: Test files provided to validate the functions.

**Root Files:**
- `helpers.py`: Helper functions for submission creation.
- `implementations.py`: Project-required implementations of functions.
- `requirements.txt`: Environment dependencies.
- `run.py`: Script to produce the best predictions, might take a while to train the model.
- `setup.py`: Script to install the `src` package.

## Getting Started
### Setup Instructions
Follow these steps to clone the repository, set up the environment, and install the package.

#### 1. Clone the Repository
To clone this repository, use the following command:
```bash
git clone <repository_link>
cd <repository_name>
```

#### 2. Install Dependencies
Set up the environment by installing required packages from `requirements.txt`:
```bash
pip install -r requirements.txt
```

#### 3. Install the `src` Package
To install the custom `src` package, run:
```bash
pip install .
```
For editable mode, allowing modifications without reinstalling:
```bash
pip install -e .
```

### Running the Project
To execute the project pipeline and generate predictions:
```bash
python run.py
```

## Usage
1. **Data Preparation**: Place raw data files in `data/raw/` for smooth execution.
2. **Exploration and Modeling**: Use the notebooks in `notebooks/` for step-by-step analysis and model development.
3. **Final Report**: The final analysis is documented in `report/`.


---

**For further resources, visit**:
- [Course GitHub Repository](https://github.com/epfml/ML_course/tree/main/projects/project1)
- [Competition Platform on AICrowd](https://www.aicrowd.com/challenges/epfl-machine-learning-project-1)
