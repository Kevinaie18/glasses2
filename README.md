## Lapaire Sales Performance Dashboard

This is a modular Streamlit app for analyzing and forecasting Lapaire retail performance, built to satisfy the PRD in `Lapaire_Dashboard_PRD.md` and compatible with the historical Excel format.

### Project structure

```
app.py
data_processing.py
analytics.py
visualizations.py
forecasting.py
utils/export_utils.py
data/  # place Excel file(s) here
outputs/  # Excel/PDF/image exports
```

### Data input
- Expected Excel structure (columns): `Outlet`, `Country`, monthly columns like `01/01/2023`, `Total général`, `Nbr de mois d'activité`, `Monthly average (USD)`.
- Default filename looked up: `data/Copie de Lapaire_Vente mensuelles usd shop v2.xlsx`.
- You can also upload a file via the app sidebar.

### Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Features
- Trend analysis (MoM, YoY) by country and by boutique
- Heatmaps for Top 5 boutiques per country over last 12 months
- Monthly average revenue per active boutique and country
- Top/Bottom 10% ranking by year
- Probable closures detection (>= 3 months inactivity, configurable)
- “Rythme de croisière” detection (3-month moving average, variation ≤ 10%)
- Automated recommendations (Renforcer, Maintenir, Corriger, Relocaliser, Fermer)
- Forecasts for 2025–2026 with adjustable sliders (growth, openings, target per boutique)
- Interactive KPIs and Plotly visualizations
- Export analyses to Excel and PDF

### Deploy on Streamlit Cloud

1. **Prepare your repository:**
   - Ensure all files are committed to your Git repository
   - The app is already configured with proper `.streamlit/config.toml`
   - Sample data is included in the `data/` directory

2. **Deploy to Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account
   - Click "New app"
   - Select your repository: `your-username/glasses`
   - Set the main file path: `app.py`
   - Set the app URL (optional): `your-app-name`
   - Click "Deploy!"

3. **Configuration:**
   - The app will automatically install dependencies from `requirements.txt`
   - Sample data will be available immediately for demonstration
   - Users can upload their own Excel files via the sidebar

4. **Customization:**
   - To use your own data as default, replace `data/sample_data.xlsx` with your file
   - Update `DEFAULT_DATA_PATH` in `data_processing.py` if needed
   - Modify `.streamlit/config.toml` for custom theming

5. **Troubleshooting:**
   - Check the logs in the Streamlit Cloud interface if deployment fails
   - Ensure all dependencies are listed in `requirements.txt`
   - Verify the app runs locally before deploying

### Notes
- Date columns are detected automatically and parsed using `dayfirst=True` to match the Excel examples.
- PDF export uses `reportlab`. If restricted, the app will continue without PDF support.


