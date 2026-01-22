# Streamlit Cloud Deployment Checklist

## Pre-deployment Checklist ✅

- [x] **Configuration**: `.streamlit/config.toml` created with proper settings
- [x] **Dependencies**: `requirements.txt` updated with version ranges for stability
- [x] **Sample Data**: `data/sample_data.xlsx` created for immediate demo
- [x] **Git Ignore**: `.gitignore` configured to exclude unnecessary files
- [x] **Documentation**: README.md updated with deployment instructions
- [x] **Testing**: App tested locally and imports successfully

## Deployment Steps

1. **Initialize Git Repository** (if not already done):
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Lapaire Dashboard ready for Streamlit Cloud"
   ```

2. **Push to GitHub**:
   ```bash
   git remote add origin https://github.com/your-username/glasses.git
   git branch -M main
   git push -u origin main
   ```

3. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Repository: `your-username/glasses`
   - Main file path: `app.py`
   - App URL: `your-app-name` (optional)
   - Click "Deploy!"

## Post-deployment Verification

- [ ] App loads without errors
- [ ] Sample data displays correctly
- [ ] All tabs and features work
- [ ] File upload functionality works
- [ ] Export features work (Excel/PDF)

## Troubleshooting

If deployment fails:
1. Check the logs in Streamlit Cloud interface
2. Verify all dependencies are in `requirements.txt`
3. Ensure `app.py` is the main file
4. Check that sample data file exists in `data/` directory

## Customization

To use your own data:
1. Replace `data/sample_data.xlsx` with your Excel file
2. Update `DEFAULT_DATA_PATH` in `data_processing.py` if needed
3. Commit and push changes
4. Redeploy on Streamlit Cloud
