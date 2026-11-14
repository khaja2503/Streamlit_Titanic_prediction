import streamlit as st
import psycopg2

try:
    conn = psycopg2.connect(
        host=st.secrets["postgres"]["host"],
        database=st.secrets["postgres"]["database"],
        user=st.secrets["postgres"]["user"],
        password=st.secrets["postgres"]["password"],
        port=st.secrets["postgres"]["port"],
        sslmode=st.secrets["postgres"]["sslmode"]
    )

    print("Neon PostgreSQL Connected Successfully!")

    cursor = conn.cursor()

    cursor.execute("""
        SELECT COUNT(*)
        FROM titanic_project.passengers;
    """)

    result = cursor.fetchone()
    print("Total Passengers:", result[0])

    cursor.close()
    conn.close()

except Exception as e:
    print("Connection Error:", e)

import streamlit as st
import pandas as pd
import psycopg2
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Page configuration
st.set_page_config(
    page_title="Titanic Survival Dashboard",
    page_icon="🚢",
    layout="wide"
)

# Database connection
@st.cache_data(ttl=60)
def load_data():

    conn = psycopg2.connect(
        host=st.secrets["postgres"]["host"],
        database=st.secrets["postgres"]["database"],
        user=st.secrets["postgres"]["user"],
        password=st.secrets["postgres"]["password"],
        port=st.secrets["postgres"]["port"]
    )

    query = """
        SELECT *
        FROM titanic_project.passengers
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    return df


# Load dataset
try:
    df = load_data()

except Exception as e:
    st.error(f"Database connection failed: {e}")
    st.stop()


# Dashboard title
st.title("🚢 Titanic Survival Analysis Dashboard")
st.write("Explore Titanic passenger data and survival patterns.")

st.divider()

# Sidebar filters
st.sidebar.header("Dashboard Filters")

classes = st.sidebar.multiselect(
    "Select Passenger Class",
    options=sorted(df["pclass"].dropna().unique()),
    default=sorted(df["pclass"].dropna().unique())
)

genders = st.sidebar.multiselect(
    "Select Gender",
    options=sorted(df["sex"].dropna().unique()),
    default=sorted(df["sex"].dropna().unique())
)

# Filter data
filtered_df = df[
    (df["pclass"].isin(classes)) &
    (df["sex"].isin(genders))
]

# KPI metrics
total = len(filtered_df)
survivors = int(filtered_df["survived"].sum())
not_survived = total - survivors

survival_rate = (
    survivors / total * 100 if total > 0 else 0
)

avg_age = filtered_df["age"].mean()
avg_fare = filtered_df["fare"].mean()

st.subheader("Passenger Statistics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Passengers", total)
col2.metric("Survivors", survivors)
col3.metric("Not Survived", not_survived)
col4.metric("Survival Rate", f"{survival_rate:.2f}%")

col5, col6 = st.columns(2)

col5.metric(
    "Average Age",
    f"{avg_age:.2f}" if pd.notna(avg_age) else "N/A"
)

col6.metric(
    "Average Fare",
    f"{avg_fare:.2f}" if pd.notna(avg_fare) else "N/A"
)

st.divider()

# Survival analysis
st.subheader("Survival Analysis")

chart1, chart2 = st.columns(2)

with chart1:

    survival_data = (
        filtered_df["survived"]
        .value_counts()
        .rename(index={0: "Not Survived", 1: "Survived"})
        .reset_index()
    )

    survival_data.columns = ["Status", "Count"]

    fig1 = px.pie(
        survival_data,
        names="Status",
        values="Count",
        title="Survival Distribution",
        hole=0.4
    )

    st.plotly_chart(fig1, use_container_width=True)


with chart2:

    class_data = (
        filtered_df.groupby(
            ["pclass", "survived"]
        )
        .size()
        .reset_index(name="Count")
    )

    class_data["Status"] = class_data["survived"].map({
        0: "Not Survived",
        1: "Survived"
    })

    fig2 = px.bar(
        class_data,
        x="pclass",
        y="Count",
        color="Status",
        barmode="group",
        title="Survival by Passenger Class",
        labels={
            "pclass": "Passenger Class",
            "Count": "Passengers"
        }
    )

    st.plotly_chart(fig2, use_container_width=True)


# Gender analysis
st.subheader("Gender Analysis")

gender_data = (
    filtered_df.groupby(["sex", "survived"])
    .size()
    .reset_index(name="Count")
)

gender_data["Status"] = gender_data["survived"].map({
    0: "Not Survived",
    1: "Survived"
})

fig3 = px.bar(
    gender_data,
    x="sex",
    y="Count",
    color="Status",
    barmode="group",
    title="Survival by Gender"
)

st.plotly_chart(fig3, use_container_width=True)


# Age distribution
st.subheader("Passenger Age Distribution")

fig4 = px.histogram(
    filtered_df.dropna(subset=["age"]),
    x="age",
    color="sex",
    nbins=30,
    title="Age Distribution by Gender"
)

st.plotly_chart(fig4, use_container_width=True)


# Data explorer
st.subheader("Passenger Data Explorer")

st.dataframe(
    filtered_df,
    use_container_width=True,
    hide_index=True
)

st.caption(f"Displaying {len(filtered_df)} passenger records.")


# Download filtered data
st.subheader("Download Passenger Data")

csv = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Passenger Data CSV",
    data=csv,
    file_name="titanic_passengers.csv",
    mime="text/csv"
)


# Advanced Analysis
st.divider()

st.header("Advanced Passenger Analysis")

# Survival by gender
st.subheader("Survival Rate by Gender")

gender_survival = (
    filtered_df.groupby("sex")["survived"]
    .mean()
    .mul(100)
    .reset_index()
)

gender_survival.columns = ["Gender", "Survival Rate"]

fig5 = px.bar(
    gender_survival,
    x="Gender",
    y="Survival Rate",
    color="Gender",
    title="Gender-wise Survival Percentage",
    text_auto=".2f"
)

st.plotly_chart(fig5, use_container_width=True)


# Survival by age group
st.subheader("Survival by Age Group")

age_df = filtered_df.dropna(subset=["age"]).copy()

age_df["Age Group"] = pd.cut(
    age_df["age"],
    bins=[0, 12, 18, 35, 60, 100],
    labels=[
        "Child",
        "Teenager",
        "Adult",
        "Middle Age",
        "Senior"
    ]
)

age_survival = (
    age_df.groupby("Age Group", observed=False)["survived"]
    .mean()
    .mul(100)
    .reset_index()
)

age_survival.columns = ["Age Group", "Survival Rate"]

fig6 = px.bar(
    age_survival,
    x="Age Group",
    y="Survival Rate",
    title="Survival Percentage by Age Group",
    text_auto=".2f"
)

st.plotly_chart(fig6, use_container_width=True)


# Fare comparison
st.subheader("Average Fare by Passenger Class")

fare_data = (
    filtered_df.groupby("pclass")["fare"]
    .mean()
    .reset_index()
)

fig7 = px.bar(
    fare_data,
    x="pclass",
    y="fare",
    title="Average Ticket Fare by Class",
    labels={
        "pclass": "Passenger Class",
        "fare": "Average Fare"
    },
    text_auto=".2f"
)

st.plotly_chart(fig7, use_container_width=True)


# =====================================
# MACHINE LEARNING SURVIVAL PREDICTION
# =====================================

st.divider()

st.header("🤖 Titanic Survival Prediction")
st.write("Predict passenger survival using Machine Learning.")

@st.cache_resource
def train_model(data):

    features = [
        "pclass", "sex", "age",
        "sibsp", "parch", "fare", "embarked"
    ]

    model_data = data.dropna(subset=["survived"]).copy()

    X = model_data[features]
    y = model_data["survived"].astype(int)

    numeric_features = [
        "pclass", "age", "sibsp", "parch", "fare"
    ]

    categorical_features = ["sex", "embarked"]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    return model, accuracy


model, accuracy = train_model(df)

st.metric("Model Accuracy", f"{accuracy * 100:.2f}%")

st.subheader("Enter Passenger Details")

with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        pclass = st.selectbox(
            "Passenger Class",
            [1, 2, 3]
        )

        sex = st.selectbox(
            "Gender",
            ["male", "female"]
        )

        age = st.number_input(
            "Age",
            min_value=0.0,
            max_value=100.0,
            value=25.0
        )

        fare = st.number_input(
            "Ticket Fare",
            min_value=0.0,
            value=32.0
        )

    with col2:
        sibsp = st.number_input(
            "Number of Siblings/Spouses",
            min_value=0,
            max_value=10,
            value=0
        )

        parch = st.number_input(
            "Number of Parents/Children",
            min_value=0,
            max_value=10,
            value=0
        )

        embarked = st.selectbox(
            "Port of Embarkation",
            ["S", "C", "Q"]
        )

    submit = st.form_submit_button(
        "Predict Survival"
    )

    if submit:

        input_data = pd.DataFrame([{
            "pclass": pclass,
            "sex": sex,
            "age": age,
            "sibsp": sibsp,
            "parch": parch,
            "fare": fare,
            "embarked": embarked
        }])

        prediction = model.predict(input_data)[0]

        survival_probability = model.predict_proba(
            input_data
        )[0][list(model.classes_).index(1)]

        st.divider()

        if prediction == 1:
            st.success("Prediction: Passenger may Survive!")
        else:
            st.error("Prediction: Passenger may Not Survive!")

        st.metric(
            "Estimated Survival Probability",
            f"{survival_probability * 100:.2f}%"
        )