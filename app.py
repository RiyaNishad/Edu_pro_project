import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="EduPro Demand Dashboard", layout="wide")

@st.cache_data
def load_data():
    file_path = r"C:\Users\riyan\Downloads\edu_pro_project\EduPro Online Platform.xlsx"
    sheets = pd.read_excel(file_path, sheet_name=None, engine="openpyxl")

    users = sheets["Users"]
    teachers = sheets["Teachers"]
    courses = sheets["Courses"]
    transactions = sheets["Transactions"]

    transactions["TransactionDate"] = pd.to_datetime(transactions["TransactionDate"], errors="coerce")

    course_metrics = transactions.groupby("CourseID").agg(
        enrollment_count=("TransactionID", "count"),
        total_revenue=("Amount", "sum"),
        avg_transaction_amount=("Amount", "mean"),
        active_teachers=("TeacherID", "nunique"),
        latest_transaction=("TransactionDate", "max")
    ).reset_index()

    teacher_metrics = transactions.groupby("CourseID").agg(
        unique_teachers=("TeacherID", "nunique")
    ).reset_index()

    course_teacher_map = (
        transactions[["CourseID", "TeacherID"]]
        .drop_duplicates()
        .merge(
            teachers[["TeacherID", "Expertise", "YearsOfExperience", "TeacherRating"]],
            on="TeacherID",
            how="left"
        )
    )

    teacher_agg = course_teacher_map.groupby("CourseID").agg(
        avg_teacher_experience=("YearsOfExperience", "mean"),
        avg_teacher_rating=("TeacherRating", "mean")
    ).reset_index()

    df = courses.merge(course_metrics, on="CourseID", how="left")
    df = df.merge(teacher_metrics, on="CourseID", how="left")
    df = df.merge(teacher_agg, on="CourseID", how="left")

    df["enrollment_count"] = df["enrollment_count"].fillna(0)
    df["total_revenue"] = df["total_revenue"].fillna(0)
    df["avg_transaction_amount"] = df["avg_transaction_amount"].fillna(0)
    df["unique_teachers"] = df["unique_teachers"].fillna(0)
    df["avg_teacher_experience"] = df["avg_teacher_experience"].fillna(df["avg_teacher_experience"].median())
    df["avg_teacher_rating"] = df["avg_teacher_rating"].fillna(df["avg_teacher_rating"].median())

    df["price_band"] = pd.cut(
        df["CoursePrice"],
        bins=[-0.1, 0, 100, 300, 1000],
        labels=["Free", "Low", "Medium", "High"]
    )

    df["duration_bucket"] = pd.cut(
        df["CourseDuration"],
        bins=[0, 10, 25, 40, 60],
        labels=["Short", "Medium", "Long", "Very Long"]
    )

    df["rating_tier"] = pd.cut(
        df["CourseRating"],
        bins=[0, 2.5, 3.5, 5],
        labels=["Low", "Average", "High"]
    )

    df["price_per_hour"] = df["CoursePrice"] / (df["CourseDuration"] + 1)
    df["revenue_per_enrollment"] = df["total_revenue"] / (df["enrollment_count"] + 1)

    return df, transactions, teachers, users

@st.cache_resource
def train_model(df):
    model_df = df.copy()

    model_df = pd.get_dummies(
        model_df,
        columns=["CourseCategory", "CourseType", "CourseLevel", "price_band", "duration_bucket", "rating_tier"],
        drop_first=True
    )

    feature_cols = [
        "CoursePrice",
        "CourseDuration",
        "CourseRating",
        "price_per_hour",
        "avg_teacher_experience",
        "avg_teacher_rating",
        "unique_teachers"
    ] + [c for c in model_df.columns if c.startswith("CourseCategory_")
         or c.startswith("CourseType_")
         or c.startswith("CourseLevel_")
         or c.startswith("price_band_")
         or c.startswith("duration_bucket_")
         or c.startswith("rating_tier_")]

    X = model_df[feature_cols]
    y_enroll = model_df["enrollment_count"]
    y_revenue = model_df["total_revenue"]

    X_train, X_test, y_enroll_train, y_enroll_test = train_test_split(
        X, y_enroll, test_size=0.2, random_state=42
    )
    _, _, y_revenue_train, y_revenue_test = train_test_split(
        X, y_revenue, test_size=0.2, random_state=42
    )

    enroll_model = RandomForestRegressor(n_estimators=200, random_state=42)
    revenue_model = RandomForestRegressor(n_estimators=200, random_state=42)

    enroll_model.fit(X_train, y_enroll_train)
    revenue_model.fit(X_train, y_revenue_train)

    enroll_pred = enroll_model.predict(X_test)
    revenue_pred = revenue_model.predict(X_test)

    metrics = {
        "enroll_mae": mean_absolute_error(y_enroll_test, enroll_pred),
        "enroll_r2": r2_score(y_enroll_test, enroll_pred),
        "revenue_mae": mean_absolute_error(y_revenue_test, revenue_pred),
        "revenue_r2": r2_score(y_revenue_test, revenue_pred),
        "feature_cols": feature_cols,
        "dummy_columns": X.columns.tolist()
    }

    return enroll_model, revenue_model, metrics

def build_input_df(df, user_input, feature_cols, dummy_columns):
    input_df = pd.DataFrame([user_input])

    input_df["price_band"] = pd.cut(
        input_df["CoursePrice"],
        bins=[-0.1, 0, 100, 300, 1000],
        labels=["Free", "Low", "Medium", "High"]
    )
    input_df["duration_bucket"] = pd.cut(
        input_df["CourseDuration"],
        bins=[0, 10, 25, 40, 60],
        labels=["Short", "Medium", "Long", "Very Long"]
    )
    input_df["rating_tier"] = pd.cut(
        input_df["CourseRating"],
        bins=[0, 2.5, 3.5, 5],
        labels=["Low", "Average", "High"]
    )
    input_df["price_per_hour"] = input_df["CoursePrice"] / (input_df["CourseDuration"] + 1)

    input_df = pd.get_dummies(
        input_df,
        columns=["CourseCategory", "CourseType", "CourseLevel", "price_band", "duration_bucket", "rating_tier"],
        drop_first=True
    )

    for col in dummy_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df.reindex(columns=dummy_columns, fill_value=0)
    return input_df

df, transactions, teachers, users = load_data()
enroll_model, revenue_model, metrics = train_model(df)

st.title("EduPro Course Demand & Revenue Dashboard")
st.caption("Predictive dashboard for course planning, pricing, and instructor strategy.")

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Category Insights", "Prediction", "Model Performance"])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Courses", int(df["CourseID"].nunique()))
    c2.metric("Total Transactions", int(transactions["TransactionID"].nunique()))
    c3.metric("Total Revenue", f"${df['total_revenue'].sum():,.2f}")
    c4.metric("Total Enrollments", int(df["enrollment_count"].sum()))

    top_rev = df.groupby("CourseCategory", as_index=False)["total_revenue"].sum().sort_values("total_revenue", ascending=False)
    fig1 = px.bar(top_rev, x="CourseCategory", y="total_revenue", title="Revenue by Category")
    st.plotly_chart(fig1, use_container_width=True)

    top_enroll = df.groupby("CourseCategory", as_index=False)["enrollment_count"].sum().sort_values("enrollment_count", ascending=False)
    fig2 = px.bar(top_enroll, x="CourseCategory", y="enrollment_count", title="Enrollments by Category")
    st.plotly_chart(fig2, use_container_width=True)

with tab2:
    category_summary = df.groupby("CourseCategory", as_index=False).agg(
        courses=("CourseID", "count"),
        revenue=("total_revenue", "sum"),
        enrollments=("enrollment_count", "sum"),
        avg_price=("CoursePrice", "mean"),
        avg_rating=("CourseRating", "mean")
    ).sort_values("revenue", ascending=False)

    st.dataframe(category_summary, use_container_width=True)

    fig3 = px.scatter(
        df,
        x="CoursePrice",
        y="enrollment_count",
        color="CourseCategory",
        size="total_revenue",
        hover_name="CourseName",
        title="Price vs Enrollment by Course"
    )
    st.plotly_chart(fig3, use_container_width=True)

with tab3:
    st.subheader("Predict Future Course Performance")

    col1, col2, col3 = st.columns(3)
    with col1:
        category = st.selectbox("Course Category", sorted(df["CourseCategory"].dropna().unique()))
        course_type = st.selectbox("Course Type", sorted(df["CourseType"].dropna().unique()))
        course_level = st.selectbox("Course Level", sorted(df["CourseLevel"].dropna().unique()))
    with col2:
        price = st.number_input("Course Price", min_value=0.0, value=199.0, step=10.0)
        duration = st.number_input("Course Duration", min_value=1.0, value=25.0, step=1.0)
        rating = st.slider("Course Rating", min_value=1.0, max_value=5.0, value=4.0, step=0.1)
    with col3:
        avg_teacher_experience = st.number_input("Avg Teacher Experience", min_value=0.0, value=6.0, step=1.0)
        avg_teacher_rating = st.slider("Avg Teacher Rating", min_value=1.0, max_value=5.0, value=4.0, step=0.1)
        unique_teachers = st.number_input("Number of Teachers", min_value=1, value=1, step=1)

    user_input = {
        "CoursePrice": price,
        "CourseDuration": duration,
        "CourseRating": rating,
        "avg_teacher_experience": avg_teacher_experience,
        "avg_teacher_rating": avg_teacher_rating,
        "unique_teachers": unique_teachers,
        "CourseCategory": category,
        "CourseType": course_type,
        "CourseLevel": course_level
    }

    if st.button("Predict"):
        X_new = build_input_df(df, user_input, metrics["feature_cols"], metrics["dummy_columns"])
        predicted_enrollment = enroll_model.predict(X_new)[0]
        predicted_revenue = revenue_model.predict(X_new)[0]

        p1, p2 = st.columns(2)
        p1.metric("Predicted Enrollments", f"{predicted_enrollment:.0f}")
        p2.metric("Predicted Revenue", f"${predicted_revenue:,.2f}")

with tab4:
    m1, m2 = st.columns(2)
    m1.metric("Enrollment MAE", f"{metrics['enroll_mae']:.2f}")
    m1.metric("Enrollment R²", f"{metrics['enroll_r2']:.3f}")
    m2.metric("Revenue MAE", f"{metrics['revenue_mae']:.2f}")
    m2.metric("Revenue R²", f"{metrics['revenue_r2']:.3f}")

    importance_df = pd.DataFrame({
        "Feature": metrics["dummy_columns"],
        "Importance": revenue_model.feature_importances_
    }).sort_values("Importance", ascending=False).head(15)

    fig4 = px.bar(importance_df, x="Importance", y="Feature", orientation="h", title="Top Revenue Drivers")
    st.plotly_chart(fig4, use_container_width=True)