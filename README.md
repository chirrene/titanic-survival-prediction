
# Titanic Survival Prediction

## Project Overview
This project uses Python and R to predict passenger survival on the Titanic dataset using logistic regression.

## Repository Structure
- `src/python/main.py`: Python data analysis and prediction
- `src/python/requirements.txt`: Python dependencies
- `src/python/Dockerfile`: Python Docker configuration
- `src/r/main.R`: R data analysis and prediction
- `src/r/install_packages.R`: R package installation
- `src/r/Dockerfile`: R Docker configuration
- `src/data/`: Titanic datasets (excluded from Git)

## Dataset
Download the Titanic dataset from:
https://www.kaggle.com/competitions/titanic/data

Place `train.csv` and `test.csv` inside `src/data/`.

## Python Analysis
The Python script:
1. Loads the training dataset.
2. Explores and cleans missing values.
3. Trains a logistic regression model.
4. Calculates training accuracy.
5. Predicts survival on the Kaggle test dataset.
6. Evaluates accuracy using a holdout split.

## R Analysis
The R script repeats the analysis using R's `glm()` function.

## Run with Docker

Run these commands from the repository root.

### Python

```bash
docker build -f src/python/Dockerfile -t titanic-python src
docker run --rm titanic-python
```

### R

```bash
docker build -f src/r/Dockerfile -t titanic-r src
docker run --rm titanic-r
```

## Model Evaluation
Python training accuracy: 80.02%.

The Kaggle test dataset does not contain true survival labels. Therefore, a separate holdout split of the labeled training dataset is used to estimate performance on unseen data.

## Testing Status

- Python analysis executed successfully locally.
- R analysis executed successfully locally using R 4.6.1.
- Python training accuracy: 80.02%.
- R training accuracy: 79.80%.
- R holdout accuracy: 75.98%.
- Docker builds and container execution have not yet been verified.

## Notes
The dataset is excluded from Git using `.gitignore`, so it must be downloaded before building the Docker images.
