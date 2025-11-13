@echo off
REM ============================================================================
REM Broadband Price Elasticity Analysis - Project Structure Setup
REM Author: Samir Orujov
REM Date: November 13, 2025
REM ============================================================================

echo.
echo ============================================================================
echo Creating Broadband Elasticity Project Structure
echo ============================================================================
echo.

REM Create main project directory
mkdir broadband_elasticity_project
cd broadband_elasticity_project

REM Create main folders
echo Creating main folders...
mkdir data
mkdir code
mkdir results
mkdir docs
mkdir figures
mkdir logs

REM Create data subfolders
echo Creating data subfolders...
mkdir data\raw
mkdir data\processed
mkdir data\interim

REM Create code subfolders
echo Creating code subfolders...
mkdir code\data_collection
mkdir code\data_preparation
mkdir code\analysis
mkdir code\visualization
mkdir code\utils

REM Create results subfolders
echo Creating results subfolders...
mkdir results\tables
mkdir results\regression_output
mkdir results\robustness

REM Create docs subfolders
echo Creating docs subfolders...
mkdir docs\manuscript
mkdir docs\methodology
mkdir docs\literature

REM Create figures subfolders
echo Creating figures subfolders...
mkdir figures\descriptive
mkdir figures\regression
mkdir figures\maps

REM Create README files
echo Creating README files...

(
echo # Broadband Price Elasticity Analysis
echo.
echo ## Project Structure
echo.
echo - **data/**: All data files
echo   - raw/: Original downloaded data
echo   - interim/: Intermediate processing steps
echo   - processed/: Final analysis-ready datasets
echo - **code/**: All analysis scripts
echo   - data_collection/: Download scripts
echo   - data_preparation/: Cleaning and preparation
echo   - analysis/: Econometric models
echo   - visualization/: Plots and figures
echo - **results/**: Analysis outputs
echo - **docs/**: Documentation and manuscript
echo - **figures/**: All generated figures
echo - **logs/**: Execution logs
) > README.md

echo # Raw Data > data\raw\README.md
echo # Processed Data > data\processed\README.md
echo # Code Documentation > code\README.md

REM Create .gitignore
echo Creating .gitignore...
(
echo # Data files
echo data/raw/*.csv
echo data/raw/*.xlsx
echo *.zip
echo.
echo # Python
echo __pycache__/
echo *.pyc
echo .ipynb_checkpoints/
echo venv/
echo .env
echo.
echo # R
echo .Rhistory
echo .RData
echo .Rproj.user/
echo.
echo # System files
echo .DS_Store
echo Thumbs.db
) > .gitignore

REM Create requirements.txt
echo Creating requirements.txt...
(
echo # Python Dependencies for Broadband Elasticity Analysis
echo # Data manipulation
echo pandas^>=2.0.0
echo numpy^>=1.24.0
echo.
echo # Data collection
echo requests^>=2.31.0
echo wbgapi^>=1.0.12
echo beautifulsoup4^>=4.12.0
echo.
echo # Econometric analysis
echo linearmodels^>=5.3
echo statsmodels^>=0.14.0
echo scikit-learn^>=1.3.0
echo scipy^>=1.11.0
echo.
echo # Visualization
echo matplotlib^>=3.7.0
echo seaborn^>=0.12.0
echo plotly^>=5.17.0
echo.
echo # Jupyter
echo jupyter^>=1.0.0
echo ipykernel^>=6.25.0
echo.
echo # Utilities
echo openpyxl^>=3.1.0
echo xlrd^>=2.0.1
) > requirements.txt

echo.
echo ============================================================================
echo Project structure created successfully!
echo ============================================================================
echo.
echo Next steps:
echo 1. cd broadband_elasticity_project
echo 2. Create a virtual environment: python -m venv venv
echo 3. Activate it: venv\Scripts\activate
echo 4. Install dependencies: pip install -r requirements.txt
echo 5. Run data collection script
echo.
echo Project location: %cd%\broadband_elasticity_project
echo.
pause
