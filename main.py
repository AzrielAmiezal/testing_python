import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_option_menu import option_menu

# Use st.cache_data to cache the data-loading function
@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv")
    df["ADMISSION DATE"] = pd.to_datetime(df["ADMISSION DATE"])
    df["DATE_OF_FIRST_SYMPTOM"] = pd.to_datetime(df["DATE_OF_FIRST_SYMPTOM"])
    df["DATE_OF_DEATH"] = pd.to_datetime(df["DATE_OF_DEATH"])
    return df

df = load_data()

# Sidebar Navigation Menu
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Data Overview", "Age & Gender Analysis", "Intubation & ICU", "Diseases & Mortality"],
        icons=["info-circle", "bar-chart", "activity", "heart"],
        menu_icon="cast",
        default_index=0,
    )

# Page: Data Overview
if selected == "Data Overview":
    st.title("COVID-19 Data Overview")
    st.write("This section provides a general overview of the dataset.")
    st.write(df.head(10))

    st.subheader("Filter Data by Age")
    min_age, max_age = st.slider("Select Age Range", 0, 100, (20, 60))
    filtered_data = df[(df["AGE"] >= min_age) & (df["AGE"] <= max_age)]
    st.write(f"Displaying data for ages between {min_age} and {max_age}")
    st.write(filtered_data)

# Page: Age & Gender Analysis
elif selected == "Age & Gender Analysis":
    st.title("Age & Gender Analysis")

    age_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    age_labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100']
    df['AGE_GROUP'] = pd.cut(df['AGE'], bins=age_bins, labels=age_labels, right=False)

    tab1, tab2 = st.tabs(["Age Group Distribution", "Gender by Age Group"])

    with tab1:
        st.subheader("COVID-19 Susceptibility by Age Group")
        age_group_counts = df['AGE_GROUP'].value_counts().sort_index()
        st.bar_chart(age_group_counts)

    with tab2:
        st.subheader("Distribution by Gender and Age Group")
        age_gender_distribution = df.groupby(['AGE_GROUP', 'SEX']).size().unstack()
        st.bar_chart(age_gender_distribution)

# Page: Intubation & ICU Analysis
elif selected == "Intubation & ICU":
    st.title("Intubation & ICU Analysis")
    
    tab1, tab2 = st.tabs(["Intubation Status", "ICU Correlation"])

    with tab1:
        st.subheader("Intubation Status Distribution")
        intubation_counts = df['INTUBATED'].value_counts()
        st.bar_chart(intubation_counts)

    with tab2:
        st.subheader("Correlation between Diseases and ICU Admission")
        diseases = ['DIABETES', 'COPD', 'ASTHMA', 'INMUSUPR', 'HYPERTENSION', 'CARDIOVASCULAR', 'OBESITY', 'CHRONIC_KIDNEY', 'TOBACCO']
        df_diseases_icu = df[diseases + ['ICU']].apply(pd.to_numeric, errors='coerce').fillna(0)
        correlation_matrix = df_diseases_icu.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
        st.pyplot(fig)

# Page: Diseases & Mortality
elif selected == "Diseases & Mortality":
    st.title("Diseases & Mortality Analysis")
    
    tab1, tab2 = st.tabs(["Common Diseases", "Diseases among Deceased"])

    with tab1:
        st.subheader("Common Diseases Distribution")
        disease_counts = df[['DIABETES', 'COPD', 'ASTHMA', 'INMUSUPR', 'HYPERTENSION', 'CARDIOVASCULAR', 'OBESITY', 'CHRONIC_KIDNEY', 'TOBACCO']].apply(pd.Series.value_counts).loc[1]
        st.bar_chart(disease_counts)

    with tab2:
        st.subheader("Common Diseases among Deceased Patients")
        deceased_df = df[df['OUTCOME'] == 1]
        deceased_disease_counts = deceased_df[['DIABETES', 'COPD', 'ASTHMA', 'INMUSUPR', 'HYPERTENSION', 'CARDIOVASCULAR', 'OBESITY', 'CHRONIC_KIDNEY', 'TOBACCO']].apply(lambda x: (x == 1).sum())
        st.bar_chart(deceased_disease_counts)
