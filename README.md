# Tree Ensemble Analysis Tool 🌳⚙️ (v3.3)

A Streamlit application built for Data Engineers and Data Scientists to quickly train and evaluate Random Forest and Extra Trees models on tabular datasets. This version includes options for loading data from a directory or via direct file upload, along with performance and UI refinements.

## ✨ Features

* **Algorithm Selection:** Choose between Random Forest and Extra Trees classifiers/regressors.
* **Flexible Dataset Input:**
    * Load CSV datasets from a specified local directory.
    * Upload a CSV file directly via the browser.
* **Interactive UI:** Configure analysis parameters via a user-friendly Streamlit interface.
    * Select Target and Feature columns.
    * Improved auto-detection for Problem Type (Classification/Regression) with warnings for potential mismatches.
    * Adjust common Hyperparameters within an expandable section.
    * Configure Train/Test split ratio.
* **Automated Preprocessing:**
    * Handles missing numerical values (median imputation).
    * Handles missing categorical values ('\_MISSING\_' placeholder).
    * Applies One-Hot Encoding for categorical features.
    * Applies Label Encoding for non-numeric/float classification targets.
* **Model Training & Evaluation:**
    * Trains the selected model using Scikit-learn.
    * Performs stratified train/test split for classification (with robust fallback for small classes).
    * Displays relevant evaluation metrics and visualizations in dedicated **Tabs** (Metrics, Plots, Data Info).
    * Includes Accuracy, Classification Report, Confusion Matrix (Classification); R², MSE, RMSE, Residuals Plot (Regression); Feature Importances (Both).
* **Performance & Caching:**
    * Uses Streamlit caching (`@st.cache_data`, `@st.cache_resource`) to speed up data loading, preprocessing, and model training reruns.
    * *Note on Caching:* Caching is based on inputs like file path/content hash, selected columns, parameters, etc. If the *content* of a file in the directory changes *without the filename changing*, the cache might not update automatically on a simple rerun; restarting the app or re-uploading the file may be needed.
* **Reproducibility:** Uses a fixed `random_state` for consistent splitting and model initialization.
* **Modern UI:** Uses tabs, columns, expanders, metrics, and themed plots for a clean presentation.
* **Production-Ready Aspects:** Modular code structure, clear configuration constants, enhanced error handling (including data split edge cases), type hinting, dependency management.

## 📂 Project Structure

random_forest_streamlit/
├── app.py                 # Main Streamlit application code
├── requirements.txt       # Python dependencies
└── data/                  # << DEFAULT DIRECTORY FOR CSV FILES >>
└── sample_data.csv    # Example dataset (optional)

## 🛠️ Setup Instructions

1.  **Prerequisites:**
    * Python (>= 3.8 recommended)
    * `pip` (Python package installer)

2.  **Clone/Download:** Get the project files onto your local machine.

3.  **Navigate to Directory:** Open your terminal or command prompt and change into the project's root directory (`random_forest_streamlit/`).

4.  **Create Virtual Environment (Recommended):**
    ```bash
    python -m venv .venv
    # Activate the environment:
    # Windows:
    # .venv\Scripts\activate
    # macOS/Linux:
    # source .venv/bin/activate
    ```

5.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

6.  **Prepare Data (If using Directory Mode):**
    * Ensure the `data` subdirectory exists (or the directory you specify in the UI). Create it if needed (`mkdir data`).
    * Place your CSV dataset files inside that directory.

## ▶️ Running the App

1.  Make sure your virtual environment is activated (if you created one).
2.  Navigate to the project's root directory in your terminal.
3.  Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```
4.  The application should automatically open in your default web browser.

## 🖱️ How to Use

1.  **Load Data:**
    * Choose your input method in the sidebar: "Select from Directory" or "Upload CSV File".
    * If "Directory": Enter the path (defaults to `./data`) and select a file from the dropdown.
    * If "Upload": Click "Browse files" and select a CSV from your computer.
2.  **Dataset Preview:** Once data is loaded, an expander on the main page shows a preview and basic column info.
3.  **Column Setup (Sidebar):**
    * Select your **Target Column**.
    * Select one or more **Feature Columns**.
4.  **Analysis Setup (Sidebar):**
    * Verify or choose the **Problem Type**. Check for warnings if selecting Regression for non-numeric targets.
    * Select the **Algorithm**.
    * Optionally expand "Tune Hyperparameters" to adjust model settings.
    * Choose the **Test Set Size**.
5.  **Run Analysis:** Click the "🚀 Run \[Algorithm Name] Analysis" button.
6.  **View Results:** The main panel will show results organized into tabs:
    * **📊 Metrics:** Displays key performance scores and the Classification Report.
    * **📈 Plots:** Shows visualizations like Confusion Matrix, Residuals Plot, and Feature Importances.
    * **ℹ️ Data Info:** Provides summary statistics of the original data and details about the preprocessing steps performed.

## 📜 Dependencies

The required Python libraries are listed in `requirements.txt`:

```txt
streamlit==1.32.2
pandas==2.2.1
scikit-learn==1.4.1.post1
matplotlib==3.8.3
seaborn==0.13.2
numpy==1.26.4
