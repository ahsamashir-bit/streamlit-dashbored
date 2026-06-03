# End-to-End Machine Learning Project

This project demonstrates a complete data science workflow using the Telco Customer Churn dataset:

- Exploratory Data Analysis (EDA)
- Data cleaning and preparation
- Training and comparing multiple classification models
- Generating evaluation metrics
- Serving an interactive Streamlit dashboard

## Structure

- `data/` - Telco churn CSV dataset
- `notebooks/` - EDA notebook
- `src/data.py` - data loading and preprocessing
- `src/modeling.py` - model training, evaluation, and persistence
- `app.py` - Streamlit dashboard
- `requirements.txt` - project dependencies

## Getting Started

1. Create a Python virtual environment:

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Train the models and generate metrics:

```bash
py -m src.modeling
```

4. Start the Streamlit app:

```bash
py -m streamlit run app.py
```

## Notes

This project uses the supplied Telco churn dataset from `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`.

The dashboard compares logistic regression, random forest, and gradient boosting classifiers, and predicts churn for a custom customer profile.
