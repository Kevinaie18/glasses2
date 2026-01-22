
# Lapaire Dashboard - Historical Code

```python
import pandas as pd
import streamlit as st
import plotly.express as px

# Load data
@st.cache_data
def load_data():
    df = pd.read_excel("Copie de Lapaire_Vente mensuelles usd shop v2.xlsx")
    return df

df = load_data()

st.title("Lapaire Sales Dashboard")

# Country filter
countries = st.multiselect("Select countries", options=df["Country"].unique(), default=df["Country"].unique())
filtered_df = df[df["Country"].isin(countries)]

# Date selection
date_cols = [col for col in df.columns if col.startswith("202") or col.startswith("01/")]
selected_months = st.slider("Select months range", 0, len(date_cols)-1, (0, len(date_cols)-1))
month_range = date_cols[selected_months[0]:selected_months[1]+1]

# Aggregate per country
country_sales = filtered_df.groupby("Country")[month_range].sum().T
country_sales_plot = country_sales.melt(ignore_index=False).reset_index()
country_sales_plot.columns = ["Month", "Country", "Sales"]

fig_country = px.line(country_sales_plot, x="Month", y="Sales", color="Country", title="Monthly Sales by Country")
st.plotly_chart(fig_country)

# Heatmap for top boutiques
top_boutiques = (
    filtered_df.assign(Total=filtered_df[month_range].sum(axis=1))
    .sort_values("Total", ascending=False)
    .head(10)
)

heatmap_data = top_boutiques.set_index("Outlet")[month_range]
fig_heatmap = px.imshow(heatmap_data, labels=dict(x="Month", y="Outlet", color="Sales (USD)"), aspect="auto")
st.plotly_chart(fig_heatmap)

# Metrics
st.subheader("Key Metrics")
total_sales = filtered_df[month_range].sum().sum()
st.metric("Total Sales (USD)", f"{total_sales:,.0f}")
```

