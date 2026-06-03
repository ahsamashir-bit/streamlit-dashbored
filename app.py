import joblib
import pandas as pd
import streamlit as st

from src.data import clean_dataset, load_dataset


MODEL_FILE_MAP = {
    "Logistic Regression": "logistic_regression.joblib",
    "Random Forest": "random_forest_classifier.joblib",
    "Gradient Boosting": "gradient_boosting_classifier.joblib",
}


@st.cache_data
def load_models():
    models = {}
    for name, path in MODEL_FILE_MAP.items():
        try:
            models[name] = joblib.load(path)
        except FileNotFoundError:
            models[name] = None
    return models


@st.cache_data
def get_data():
    df = load_dataset(save_csv=True)
    return clean_dataset(df)


def show_sidebar_input(df):
    st.sidebar.header("Prediction Input")
    inputs = {}
    for column in df.columns.drop("Churn"):
        if df[column].dtype == "float64" or df[column].dtype == "int64":
            min_val = float(df[column].min())
            max_val = float(df[column].max())
            mean_val = float(df[column].mean())
            inputs[column] = st.sidebar.slider(
                column,
                min_value=min_val,
                max_value=max_val,
                value=mean_val,
                step=max((max_val - min_val) / 100, 1.0),
            )
        else:
            options = df[column].dropna().unique().tolist()
            inputs[column] = st.sidebar.selectbox(column, sorted(options))
    return pd.DataFrame([inputs])


def main():
    st.title("Telco Customer Churn Prediction")
    st.markdown(
        "This dashboard uses the Telco customer churn dataset to compare classification models and predict churn for a custom customer profile."
    )

    df = get_data()
    models = load_models()

    st.header("Dataset Overview")
    st.write(df.head())
    st.write("Shape:", df.shape)
    st.write(df.describe(include="all"))

    st.header("Churn Distribution")
    st.bar_chart(df["Churn"].value_counts())

    st.header("Model Comparison")
    metrics_file = "model_metrics.csv"
    try:
        metrics = pd.read_csv(metrics_file)
        sort_column = None
        if "ROC AUC" in metrics.columns:
            sort_column = "ROC AUC"
        elif "Accuracy" in metrics.columns:
            sort_column = "Accuracy"
        else:
            numeric_cols = metrics.select_dtypes(include=["number"]).columns.tolist()
            sort_column = numeric_cols[0] if numeric_cols else None

        if sort_column:
            st.dataframe(metrics.sort_values(by=sort_column, ascending=False))
        else:
            st.dataframe(metrics)
            st.info(
                "The metric file does not contain a standard sort column."
            )

        if "ROC AUC" not in metrics.columns:
            st.warning(
                "The metrics file appears to use an older format. Run `py -m src.modeling` to regenerate classification metrics."
            )
    except FileNotFoundError:
        st.warning("Run `py -m src.modeling` first to train models and generate metrics.")

    st.header("Predict Churn for a New Customer")
    input_df = show_sidebar_input(df)
    st.write(input_df)

    for name, model in models.items():
        if model is None:
            st.info(f"Model file for {name} not found. Train models before using this app.")
            continue

        if not hasattr(model, "predict_proba"):
            st.warning(
                f"{name} does not support probability predictions. This may be an older regression artifact; please retrain using `py -m src.modeling`."
            )
            continue

        probability = model.predict_proba(input_df)[0][1]
        prediction = model.predict(input_df)[0]
        st.write(
            f"**{name}**: churn probability = {probability:.3f}, prediction = {'Yes' if prediction == 1 else 'No'}"
        )

    st.header("Saved Data Files")
    st.write("- data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    st.write("- data/telco_customer_churn.csv")
    st.write("- model_metrics.csv")
    st.write(
        "- model artifacts: logistic_regression.joblib, random_forest_classifier.joblib, gradient_boosting_classifier.joblib"
    )


if __name__ == "__main__":
    main()
