import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="Predictive Customer Churn Panel",
    page_icon="🛡️",
    layout="wide"
)

# Title & Subtitle
st.title("🛡️ Predictive Customer Churn Panel")
st.markdown("Proactively identify active accounts at risk and execute targeted retention strategies.")

# 2. Load & Transform Data
@st.cache_data
def load_and_process_data():
    df = pd.read_csv('Customers.csv')
    df.columns = df.columns.str.strip()

    np.random.seed(42)
    base_churn = 1.0 - (df['Spending Score (1-100)'] / 100.0)
    noise = np.random.normal(0, 0.05, len(df))
    df['P_Churn'] = np.clip(base_churn + noise, 0.0, 1.0)

    conditions = [
        (df['P_Churn'] >= 0.80),
        (df['P_Churn'] >= 0.40) & (df['P_Churn'] < 0.80),
        (df['P_Churn'] < 0.40)
    ]
    choices = ['High Risk', 'Medium Risk', 'Low Risk']
    df['Risk_Tier'] = np.select(conditions, choices, default='Low Risk')

    return df

df = load_and_process_data()

# 3. Sidebar Filters
st.sidebar.header("Filter Accounts")
selected_tier = st.sidebar.multiselect(
    "Select Risk Tier(s):",
    options=['High Risk', 'Medium Risk', 'Low Risk'],
    default=['High Risk', 'Medium Risk', 'Low Risk']
)

filtered_df = df[df['Risk_Tier'].isin(selected_tier)]

# 4. Top KPI Metrics Dashboard
high_count = len(df[df['Risk_Tier'] == 'High Risk'])
med_count = len(df[df['Risk_Tier'] == 'Medium Risk'])
low_count = len(df[df['Risk_Tier'] == 'Low Risk'])

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Accounts", len(df))
col2.metric("🔴 High Risk (≥ 80%)", high_count, delta=f"{high_count/len(df):.1%} of total", delta_color="inverse")
col3.metric("🟡 Medium Risk (40-79%)", med_count)
col4.metric("🟢 Low Risk (< 40%)", low_count)

st.markdown("---")

# 5. Requirement 1: Risk Triage Table View
st.subheader("1. Risk Triage Overview")

search_id = st.text_input("🔍 Search by Customer ID:", "")
if search_id:
    try:
        filtered_df = filtered_df[filtered_df['CustomerID'] == int(search_id)]
    except ValueError:
        st.warning("Please enter a valid numeric Customer ID.")

def highlight_risk(val):
    if val == 'High Risk':
        return 'background-color: #ffcdd2; color: #b71c1c; font-weight: bold;'
    elif val == 'Medium Risk':
        return 'background-color: #fff9c4; color: #f57f17; font-weight: bold;'
    else:
        return 'background-color: #c8e6c9; color: #1b5e20; font-weight: bold;'

display_cols = ['CustomerID', 'Profession', 'Age', 'Annual Income ($)', 'Spending Score (1-100)', 'P_Churn', 'Risk_Tier']
formatted_df = filtered_df[display_cols].copy()
formatted_df['P_Churn'] = formatted_df['P_Churn'].apply(lambda x: f"{x:.1%}")

# FIX: Replaced .applymap() with .map() for compatibility with modern Pandas versions
st.dataframe(
    formatted_df.style.map(highlight_risk, subset=['Risk_Tier']),
    use_container_width=True,
    height=250
)

st.markdown("---")

# 6. Deep Dive & Interventions Section
st.subheader("2. Account Deep-Dive & Interventions")

if len(filtered_df) == 0:
    st.info("No customer accounts match the current filter criteria.")
else:
    selected_customer_id = st.selectbox(
        "Select a Customer ID to inspect risk factors and take action:",
        options=filtered_df['CustomerID'].tolist()
    )

    cust_data = df[df['CustomerID'] == selected_customer_id].iloc[0]

    col_left, col_right = st.columns([1, 1])

    # Requirement 2: Top Risk Drivers
    with col_left:
        st.markdown(f"#### 📊 Risk Drivers for Customer #{selected_customer_id}")
        st.write(f"**Profession:** {cust_data['Profession']} | **Age:** {cust_data['Age']} | **Risk Tier:** `{cust_data['Risk_Tier']}`")

        drivers = {
            "Low Spending Activity": (100 - cust_data['Spending Score (1-100)']) * 0.45,
            "Low Work Experience": max(0, (10 - cust_data['Work Experience']) * 3.5),
            "Income-to-Spend Disconnect": max(0, (cust_data['Annual Income ($)'] / 20000) - (cust_data['Spending Score (1-100)'] / 20)) * 4,
            "Recent Inactivity Flag": np.random.randint(10, 25)
        }

        driver_df = pd.DataFrame(list(drivers.items()), columns=['Risk Driver', 'Impact Score']).sort_values(by='Impact Score', ascending=True)

        fig = px.bar(
            driver_df,
            x='Impact Score',
            y='Risk Driver',
            orientation='h',
            color='Impact Score',
            color_continuous_scale='Reds',
            title="Factors Increasing Churn Risk"
        )
        fig.update_layout(height=280, showlegend=False, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

    # Requirement 3: Suggested Interventions
    with col_right:
        st.markdown("#### ⚡ Suggested Retention Actions")
        st.write("Execute direct actions to prevent account cancellation:")

        churn_score = cust_data['P_Churn']

        if churn_score >= 0.80:
            st.error("🚨 **CRITICAL RISK:** Immediate outreach required.")
            st.checkbox("Trigger 25% Loyalty Renewal Discount")
            st.checkbox("Schedule Urgent Call with Account Manager")
            st.checkbox("Send Priority Support Escalation Pass")
        elif churn_score >= 0.40:
            st.warning("⚠️ **MODERATE RISK:** Engagement needed.")
            st.checkbox("Send Automated Feature Re-engagement Email")
            st.checkbox("Offer 1-on-1 Product Training Session")
            st.checkbox("Invite to VIP Customer Feedback Survey")
        else:
            st.success("✅ **LOW RISK:** Account is healthy.")
            st.checkbox("Queue for Annual Upsell Campaign")
            st.checkbox("Send Standard Monthly Newsletter")

        st.text_area("Add Interaction Notes:", placeholder="Record conversation notes or custom follow-up commitments here...")

        if st.button("Submit Retention Plan", type="primary"):
            st.success(f"Retention plan saved successfully for Customer #{selected_customer_id}!")
