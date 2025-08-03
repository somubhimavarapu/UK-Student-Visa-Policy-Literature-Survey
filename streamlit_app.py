# app.py — Streamlit App for UK Student Visa Analysis

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px


from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from textblob import TextBlob
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest


st.set_page_config(page_title="UK Visa Policy Dashboard", layout="wide")
st.title("UK Student Visa Policy Analysis Dashboard")


# --- Load & preprocess dataset ---
@st.cache_data
def load_data():
    df = pd.read_csv("education-visas-datasets-mar-2025.csv")
    df.columns = df.iloc[1].values
    df = df.iloc[2:].copy()
    df.columns = ['Year', 'Quarter', 'Nationality', 'Region', 'VisaType', 'CourseLevel', 'Grants']
    df['Grants'] = pd.to_numeric(df['Grants'], errors='coerce')
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df.dropna(subset=['Grants', 'Year'], inplace=True)
    for col in ['Quarter', 'Nationality', 'Region', 'VisaType', 'CourseLevel']:
        df[col] = df[col].astype(str).str.strip()
    return df

df = load_data()

# Sidebar filters
st.sidebar.header("Filters")
year_range = st.sidebar.slider("Select Year Range", int(df['Year'].min()), int(df['Year'].max()), (2018, 2025))
selected_course_levels = st.sidebar.multiselect("Select Course Level(s)", df['CourseLevel'].unique(), default=df['CourseLevel'].unique())

df_filtered = df[
    (df['Year'] >= year_range[0]) & 
    (df['Year'] <= year_range[1]) & 
    (df['CourseLevel'].isin(selected_course_levels))
]

# --- TABS ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Forecasting", 
    "🎯 Classification", 
    "🧠 Clustering", 
    "🗣️ Sentiment", 
    "📊 Exploratory Analysis",
     "❗ Anomaly Detection"
])

# --- TAB 1: Time Series Forecasting ---
with tab1:
    st.subheader("📈 Visa Forecasting (Holt-Winters Model)")

    ts = df.groupby("Year")["Grants"].sum()
    model = ExponentialSmoothing(ts, trend="add", seasonal=None, initialization_method="estimated")
    fit = model.fit()
    forecast = fit.forecast(3)

    fig, ax = plt.subplots()
    ts.plot(label="Actual", ax=ax)
    forecast.plot(label="Forecast", ax=ax, linestyle="--")
    plt.title("Student Visa Forecast")
    plt.legend()
    st.pyplot(fig)
    st.write("Forecasted Visa Grants:", forecast)

# --- TAB 2: Classification Model ---
with tab2:
    st.subheader("🎯 Predict Policy Impact (High/Medium/Low)")

    df_class = df.groupby("Year")["Grants"].sum().reset_index()
    avg = df_class["Grants"].mean()
    df_class["Impact"] = np.select(
        [df_class["Grants"] > avg + 0.15*avg, df_class["Grants"] < avg - 0.15*avg],
        ["High", "Low"], default="Medium"
    )

    # Train classifier
    X = df_class[["Year"]]
    y = df_class["Impact"]
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X, y)
    df_class["Predicted Impact"] = clf.predict(X)

    st.dataframe(df_class)
    st.bar_chart(df_class.set_index("Year")["Grants"])

# --- TAB 3: Clustering ---
with tab3:
    st.subheader("🧠 Cluster Student Populations by Nationality & Course Level")

    df_cluster = df.groupby(["Nationality", "CourseLevel"])["Grants"].sum().reset_index()
    df_pivot = df_cluster.pivot(index="Nationality", columns="CourseLevel", values="Grants").fillna(0)

    scaler = (df_pivot - df_pivot.mean()) / df_pivot.std()
    kmeans = KMeans(n_clusters=3, random_state=42)
    df_pivot["Cluster"] = kmeans.fit_predict(scaler)

    st.dataframe(df_pivot.head(10))
    st.write("Cluster Counts:", df_pivot['Cluster'].value_counts().to_dict())

# --- TAB 4: Sentiment Analysis ---
with tab4:
    st.subheader("🗣️ Sentiment Analysis of Policy Statements")

    policy_statements = [
        "The Graduate Route provides two years post-study stay, supporting talent retention.",
        "Restrictions on dependent visas may discourage enrollment.",
        "New changes aim to streamline application process for students.",
        "Visa fee hikes could negatively affect applicant numbers."
    ]

    for i, statement in enumerate(policy_statements):
        blob = TextBlob(statement)
        st.markdown(f"**Policy {i+1}:** {statement}")
        st.write(f"→ Sentiment Polarity: `{blob.sentiment.polarity:.2f}`")
        st.write(f"→ Subjectivity: `{blob.sentiment.subjectivity:.2f}`")

# --- TAB 5: Basic EDA ---
with tab5:
    st.subheader("📊 Exploratory Data Analysis")

    st.write("### Total Grants by Year")
    grants_by_year = df_filtered.groupby("Year")["Grants"].sum()
    st.line_chart(grants_by_year)

    st.write("### Top Nationalities")
    top_nat = df_filtered.groupby("Nationality")["Grants"].sum().sort_values(ascending=False).head(10)
    st.bar_chart(top_nat)

    st.write("### Grants by Course Level")
    grants_by_course = df_filtered.groupby("CourseLevel")["Grants"].sum()
    st.bar_chart(grants_by_course)



# --- Tab 6: Anomaly Detection ---
with tab6:
    st.subheader("❗ Anomaly Detection on Yearly Grants")
    anomaly_df = df.groupby("Year")["Grants"].sum().reset_index()
    model = IsolationForest(contamination=0.2)
    anomaly_df["Anomaly"] = model.fit_predict(anomaly_df[["Grants"]])

    fig5 = px.scatter(anomaly_df, x="Year", y="Grants", color=anomaly_df["Anomaly"].map({1: "Normal", -1: "Anomaly"}))
    st.plotly_chart(fig5)
    st.dataframe(anomaly_df)