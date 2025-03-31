import streamlit as st
import pandas as pd
import os
import logging
from typing import List, Tuple, Optional, Dict, Any, Union, IO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import io  # For handling uploaded files
import numpy as np

# --- Constants & Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
DEFAULT_DATA_DIR = "data"
SUPPORTED_ALGORITHMS = ["Random Forest", "Extra Trees"]
PLOT_PALETTE = "viridis"
RANDOM_STATE = 42
APP_VERSION = "3.3"  # Updated version


# --- Helper Functions ---

@st.cache_data(show_spinner=False)
def load_data(data_source: Union[str, IO[bytes]]) -> Optional[pd.DataFrame]:
    """Loads data from a file path or an uploaded file object."""
    source_name = data_source if isinstance(data_source, str) else getattr(data_source, 'name', 'uploaded file')
    logging.info(f"Attempting to load data from: {source_name}")
    try:
        if isinstance(data_source, str) and not os.path.exists(data_source):
            st.error(f"❌ Error: File not found at '{data_source}'.")
            return None

        # Explicitly set low_memory=False for potentially mixed-type columns due to messy CSVs
        df = pd.read_csv(data_source, low_memory=False)
        logging.info(f"Successfully loaded and parsed data from {source_name}")
        if df.empty:
            st.error(f"❌ Error: The file '{source_name}' is empty.")
            return None
        df = df.infer_objects()
        return df
    except FileNotFoundError:
        st.error(f"❌ Error: File not found at {source_name}")
        return None
    except pd.errors.EmptyDataError:
        st.error(f"❌ Error: No data found in file '{source_name}'.")
        return None
    except pd.errors.ParserError as pe:
        st.error(f"❌ Error: Could not parse file '{source_name}'. Ensure it's a valid CSV. Details: {pe}")
        return None
    except Exception as e:
        st.error(f"❌ An unexpected error occurred while loading data: {e}")
        logging.error(f"Unexpected error loading {source_name}: {e}", exc_info=True)
        return None


def list_csv_files(directory: str) -> List[str]:
    """Lists all CSV files in the specified directory, handling errors."""
    try:
        if not os.path.isdir(directory):
            try:
                os.makedirs(directory)
                st.info(f"Directory '{directory}' not found. Created it for you. Please add CSV files there.")
                return []
            except OSError as e:
                st.error(f"❌ Error: Directory '{directory}' not found and could not be created: {e}")
                return []

        files = [f for f in os.listdir(directory) if
                 f.lower().endswith('.csv') and os.path.isfile(os.path.join(directory, f))]
        if not files:
            st.warning(f"No CSV files found in '{directory}'. Please add data.")
        return sorted(files)
    except Exception as e:
        st.error(f"❌ Error listing files in directory '{directory}': {e}")
        logging.error(f"Error listing files in {directory}: {e}", exc_info=True)
        return []


@st.cache_data(show_spinner="⚙️ Preprocessing data...",
               hash_funcs={pd.DataFrame: lambda x: pd.util.hash_pandas_object(x, index=True).sum()})
def preprocess_data(df: pd.DataFrame, target_column: str, features_tuple: Tuple, problem_type: str) -> Optional[Tuple]:
    """Preprocesses data: handles missing values and encodes categoricals. Cached."""
    features = list(features_tuple)
    st.write("---")
    st.write("**Preprocessing Steps:**")
    try:
        required_cols = features + [target_column]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"❌ Error: Selected columns missing from dataset: {', '.join(missing_cols)}")
            return None

        df_processed = df[required_cols].copy()

        # --- Handle Missing Target Values FIRST ---
        initial_rows = len(df_processed)
        if df_processed[target_column].isnull().any():
            df_processed.dropna(subset=[target_column], inplace=True)
            rows_dropped = initial_rows - len(df_processed)
            st.write(
                f"- Dropped **{rows_dropped} rows** due to missing values in the target column ('{target_column}').")
            logging.info(f"Dropped {rows_dropped} rows due to missing target values.")
            if df_processed.empty:
                st.error("❌ Error: No data remaining after removing rows with missing target values.")
                return None
        else:
            st.write("- No missing values found in the target column.")

        # Re-evaluate features after potential row drops (in case a feature column becomes all NaN)
        df_processed.dropna(axis=1, how='all', inplace=True)  # Drop columns that are entirely NaN
        features = [f for f in features if f in df_processed.columns]  # Update feature list
        if not features:
            st.error("❌ Error: No valid feature columns remaining after handling missing values.")
            return None

        numeric_features = df_processed[features].select_dtypes(include=np.number).columns.tolist()
        categorical_features = df_processed[features].select_dtypes(exclude=np.number).columns.tolist()

        # Imputation (Features)
        numeric_missing_count = df_processed[numeric_features].isnull().sum().sum()
        categorical_missing_count = df_processed[categorical_features].isnull().sum().sum()
        imputation_applied = False
        if numeric_missing_count > 0:
            st.write(f"- Handling {numeric_missing_count} missing numeric feature values using **median imputation**.")
            num_imputer = SimpleImputer(strategy='median')
            df_processed.loc[:, numeric_features] = num_imputer.fit_transform(df_processed[numeric_features])
            imputation_applied = True

        if categorical_missing_count > 0:
            st.write(
                f"- Handling {categorical_missing_count} missing categorical feature values using **'_MISSING_' placeholder**.")
            cat_imputer = SimpleImputer(strategy='constant', fill_value='_MISSING_')
            df_processed.loc[:, categorical_features] = cat_imputer.fit_transform(df_processed[categorical_features])
            imputation_applied = True

        if not imputation_applied:
            st.write("- No missing values found in features.")

        # Encoding (Features)
        encoded_features = features  # Start with original (potentially updated) features
        if categorical_features:
            initial_feature_count = len(features)
            st.write(f"- Applying **One-Hot Encoding** to {len(categorical_features)} categorical feature(s).")
            # Ensure only valid categorical columns are processed
            valid_categorical_features = [f for f in categorical_features if f in df_processed.columns]
            if valid_categorical_features:
                df_processed = pd.get_dummies(df_processed, columns=valid_categorical_features, drop_first=True,
                                              dtype=int)
                encoded_features = [col for col in df_processed.columns if
                                    col != target_column]  # Recalculate encoded features
                st.write(f"- Features changed from {initial_feature_count} to {len(encoded_features)} after encoding.")
            else:
                st.write("- No valid categorical features remained for encoding.")
                encoded_features = numeric_features  # Only numeric features left
        else:
            st.write("- No categorical features found for encoding.")
            encoded_features = features  # Remains the same (numeric only or originally no categoricals)

        # Target Encoding (Only for Classification)
        target_encoder = None
        if problem_type == 'Classification':
            # Convert target to string first to handle potential mixed types before encoding
            df_processed.loc[:, target_column] = df_processed[target_column].astype(str)
            st.write("- Applying **Label Encoding** to the target variable.")
            target_encoder = LabelEncoder()
            df_processed.loc[:, target_column] = target_encoder.fit_transform(df_processed[target_column])
            # Ensure target is integer type AFTER encoding
            df_processed.loc[:, target_column] = df_processed[target_column].astype(int)
            st.write(f"- Target variable classes detected: {df_processed[target_column].nunique()}")

        # Final check for Regression: Ensure target is numeric
        elif not pd.api.types.is_numeric_dtype(df_processed[target_column]):
            st.error(
                f"❌ Error: Target column '{target_column}' (dtype: {df_processed[target_column].dtype}) is non-numeric, but 'Regression' was selected. Please choose 'Classification'.")
            return None

        # Re-select X and y AFTER potential row drops and all processing
        # Ensure encoded_features only contains columns present in the final df_processed
        final_encoded_features = [f for f in encoded_features if f in df_processed.columns]
        if not final_encoded_features:
            st.error("❌ Error: No feature columns remaining after preprocessing.")
            return None

        X = df_processed[final_encoded_features].copy()
        y = df_processed[target_column].copy()

        st.success("✅ Preprocessing complete.")
        st.write(f"Shape after preprocessing: Features={X.shape[1]}, Target={y.shape[0]}")
        st.write("---")
        # Return tuple for features to be hashable for caching
        return X, y, target_encoder, tuple(final_encoded_features)

    except KeyError as e:
        st.error(f"❌ Preprocessing Error: Column '{e}' not found. Check target/feature selections.")
        logging.error(f"Preprocessing KeyError: {e}", exc_info=True)
        return None
    except Exception as e:
        st.error(f"❌ An unexpected error occurred during preprocessing: {e}")
        logging.error(f"Preprocessing error: {e}", exc_info=True)
        return None


# --- Evaluation Display Functions --- (No changes needed)

def display_feature_importances(model, feature_names_tuple: Tuple):
    """Calculates and displays feature importances."""
    feature_names = list(feature_names_tuple)
    try:
        if hasattr(model, 'feature_importances_'):
            importances = pd.Series(model.feature_importances_, index=feature_names)
            importances = importances.sort_values(ascending=False).head(20)

            fig_imp, ax_imp = plt.subplots(figsize=(10, 8))
            sns.barplot(x=importances.values, y=importances.index, ax=ax_imp, palette=PLOT_PALETTE)
            ax_imp.set_title(f'Top {len(importances)} Feature Importances')
            ax_imp.set_xlabel('Importance Score')
            ax_imp.set_ylabel('Features')
            plt.tight_layout()
            st.pyplot(fig_imp)

            with st.expander("View All Feature Importances (Table)"):
                all_importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(
                    ascending=False)
                st.dataframe(all_importances.reset_index().rename(columns={'index': 'Feature', 0: 'Importance'}))
        else:
            st.info("Feature importances are not available for this model type.")
    except Exception as e:
        st.warning(f"Could not display feature importances: {e}")
        logging.warning(f"Feature importance plotting error: {e}", exc_info=True)


def display_classification_results(y_test: pd.Series, y_pred: np.ndarray, model, feature_names_tuple: Tuple,
                                   target_encoder=None):
    """Displays evaluation metrics and plots for classification."""
    accuracy = accuracy_score(y_test, y_pred)
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{accuracy:.4f}")

    st.text("Classification Report:")
    try:
        target_names_report = None
        unique_labels_in_data = sorted(list(set(y_test) | set(y_pred)))
        if target_encoder:
            try:
                target_names_report = [str(cls) for cls in target_encoder.inverse_transform(unique_labels_in_data)]
            except ValueError:
                logging.warning("Could not map all predicted labels back to original names for report.")
                target_names_report = [f"Class_{i}" for i in unique_labels_in_data]

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0,
                                       target_names=target_names_report, labels=unique_labels_in_data)
        df_report = pd.DataFrame(report).transpose()
        st.dataframe(df_report.style.format("{:.4f}").highlight_max(axis=0, subset=pd.IndexSlice[:,
                                                                                   ['precision', 'recall',
                                                                                    'f1-score']]))
    except Exception as e:
        st.warning(f"Could not generate detailed classification report: {e}")
        report_simple = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report_simple).transpose().style.format("{:.4f}"))

    st.text("Confusion Matrix:")
    try:
        cm = confusion_matrix(y_test, y_pred, labels=unique_labels_in_data)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=target_names_report or unique_labels_in_data,
                    yticklabels=target_names_report or unique_labels_in_data)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Could not display confusion matrix: {e}")
        logging.warning(f"Confusion matrix plotting error: {e}", exc_info=True)


def display_regression_results(y_test: pd.Series, y_pred: np.ndarray, model, feature_names_tuple: Tuple):
    """Displays evaluation metrics and plots for regression."""
    try:
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, y_pred)

        col1, col2, col3 = st.columns(3)
        col1.metric("R-squared (R²)", f"{r2:.4f}")
        col2.metric("Mean Squared Error (MSE)", f"{mse:.4f}")
        col3.metric("Root Mean Squared Error (RMSE)", f"{rmse:.4f}")
    except Exception as e:
        st.error(f"❌ Could not calculate regression metrics: {e}")
        logging.error(f"Regression metric calculation error: {e}", exc_info=True)
        return

    # Residuals Plot
    st.subheader("📉 Residuals Plot")
    try:
        residuals = y_test - y_pred
        fig_res, ax_res = plt.subplots(figsize=(8, 4))
        sns.scatterplot(x=y_pred, y=residuals, ax=ax_res, alpha=0.6, color='royalblue')
        ax_res.axhline(0, color='red', linestyle='--')
        ax_res.set_xlabel("Predicted Values")
        ax_res.set_ylabel("Residuals (Actual - Predicted)")
        ax_res.set_title("Residuals vs. Predicted Values")
        plt.tight_layout()
        st.pyplot(fig_res)
    except Exception as e:
        st.warning(f"Could not display residuals plot: {e}")
        logging.warning(f"Residual plot error: {e}", exc_info=True)


# --- Cached Model Training ---

@st.cache_resource(show_spinner=False)
def train_model(_alg: str, _xtrain_shape_tpl: Tuple, _ytrain_shape_tpl: Tuple, _prob_type: str, _params_fset: frozenset,
                X_train: pd.DataFrame, y_train: pd.Series) -> Optional[Any]:
    """Instantiates and trains the selected algorithm. Cached."""
    params_dict = dict(_params_fset)
    model = None
    logging.info(f"Cache miss or config changed. Training {_alg} model...")
    try:
        if _alg == "Random Forest":
            model_class = RandomForestClassifier if _prob_type == 'Classification' else RandomForestRegressor
        elif _alg == "Extra Trees":
            model_class = ExtraTreesClassifier if _prob_type == 'Classification' else ExtraTreesRegressor
        else:
            st.error(f"Algorithm '{_alg}' not supported internally.")
            logging.error(f"Unsupported algorithm requested: {_alg}")
            return None

        model = model_class(**params_dict, random_state=RANDOM_STATE, n_jobs=-1)

        # Prepare y_train based on problem type
        if _prob_type == 'Regression':
            if not pd.api.types.is_numeric_dtype(y_train):
                st.error(
                    f"❌ Training Error: Target variable for Regression must be numeric, but found {y_train.dtype}.")
                logging.error(f"Regression target type error: {y_train.dtype}")
                return None
            y_train_final = y_train.astype(float)
        else:  # Classification
            # Ensure it's integer type before fitting classifier
            if not pd.api.types.is_integer_dtype(y_train):
                st.warning(f"Classification target y_train was not integer ({y_train.dtype}), attempting conversion.")
                try:
                    y_train_final = y_train.astype(int)
                except ValueError as e:
                    st.error(f"❌ Could not convert classification target to integer: {e}")
                    logging.error(f"Classification target int conversion error: {e}", exc_info=True)
                    return None
            else:
                y_train_final = y_train

        # Fit the model
        model.fit(X_train, y_train_final)
        logging.info(f"{_alg} model trained successfully.")
        return model

    except ValueError as ve:
        # More specific handling for common fit errors
        if "Unknown label type" in str(ve) or "could not convert string to float" in str(ve):
            st.error(
                f"❌ Training Error: Mismatch between data type and problem type. {ve}. Ensure target column is numeric for Regression or categorical/integer for Classification.")
        else:
            st.error(f"❌ ValueError during model training for {_alg}: {ve}.")
        logging.error(f"{_alg} Training ValueError: {ve}", exc_info=True)
        return None
    except Exception as e:
        st.error(f"❌ An unexpected error occurred during model training for {_alg}: {e}")
        logging.error(f"{_alg} Training error: {e}", exc_info=True)
        return None


# --- Main Analysis Orchestration ---

def run_analysis(algorithm: str, df_input: pd.DataFrame, target_column: str, features_tuple: Tuple, problem_type: str,
                 model_params: Dict, test_size: float):
    """Orchestrates the preprocessing, splitting, training, and evaluation."""

    st.markdown("---")
    st.markdown(f"### 🔬 Running Analysis: {algorithm}")

    # --- 1. Preprocess Data ---
    preprocess_result = preprocess_data(df_input, target_column, features_tuple, problem_type)

    if preprocess_result is None:
        st.error("❌ Preprocessing failed. Cannot proceed.")
        logging.error("Preprocessing function returned None.")
        return
    X, y, target_encoder, encoded_features_tuple = preprocess_result

    # --- 2. Split Data ---
    st.write("**Splitting Data:**")
    X_train, X_test, y_train, y_test = None, None, None, None
    try:
        # Ensure y is writeable for stratify by making a copy
        y_stratify = y.copy() if problem_type == 'Classification' else None

        # Final check: Ensure y for classification is integer before split
        if problem_type == 'Classification' and not pd.api.types.is_integer_dtype(y):
            st.warning(f"Target variable 'y' for classification is {y.dtype}. Forcing to integer before split.")
            y = y.astype(int)
            if y_stratify is not None:
                y_stratify = y.copy()  # Update the copy as well

        # st.write(f"Debug y before split: dtype={y.dtype}, head={y.head().to_list()}") # Optional debug line

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y_stratify
        )
        st.write(f"Data split: Training set = {X_train.shape[0]} samples, Test set = {X_test.shape[0]} samples.")
        logging.info(f"Data split complete. Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    except ValueError as e:
        if y_stratify is not None and 'least populated class' in str(e):
            st.warning(f"⚠️ Could not stratify data split (Reason: {e}). Falling back to non-stratified splitting.")
            logging.warning(f"Stratification failed: {e}. Falling back.")
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=RANDOM_STATE
                )
                st.write(
                    f"Data split (unstratified): Training set = {X_train.shape[0]} samples, Test set = {X_test.shape[0]} samples.")
                logging.info(
                    f"Unstratified data split complete. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
            except Exception as inner_e:
                st.error(f"❌ Error during fallback data splitting: {inner_e}")
                logging.error(f"Fallback data splitting error: {inner_e}", exc_info=True)
                return
        else:
            st.error(f"❌ Error during data splitting: {e}")
            logging.error(f"Data splitting ValueError: {e}", exc_info=True)
            return
    except Exception as e:
        st.error(f"❌ An unexpected error occurred during data splitting: {e}")
        logging.error(f"Unexpected data splitting error: {e}", exc_info=True)
        return

    if X_train is None or y_train is None:
        st.error("❌ Data splitting failed. Cannot proceed with training.")
        logging.error("X_train or y_train is None after split block.")
        return

    st.write("---")

    # --- 3. Train Model (Cached) ---
    with st.spinner(f"⏳ Training {algorithm} model... (This may take a moment)"):
        frozen_params = frozenset(model_params.items())
        trained_model = train_model(algorithm, X_train.shape, y_train.shape, problem_type, frozen_params, X_train,
                                    y_train)

    # --- 4. Evaluate Model & Display Results ---
    if trained_model:
        st.success(f"✅ Model training complete for {algorithm}.")
        st.markdown(f"### 🏁 Evaluation Results: {algorithm}")

        tab_metrics, tab_plots, tab_data_info = st.tabs(["📊 Metrics", "📈 Plots", "ℹ️ Data Info"])

        with st.spinner("⚙️ Evaluating model on test set..."):
            try:
                y_pred = trained_model.predict(X_test)

                # --- Metrics Tab ---
                with tab_metrics:
                    if problem_type == 'Classification':
                        display_classification_results(y_test, y_pred, trained_model, encoded_features_tuple,
                                                       target_encoder)
                    else:  # Regression
                        display_regression_results(y_test, y_pred, trained_model, encoded_features_tuple)

                # --- Plots Tab ---
                with tab_plots:
                    st.info("Plots related to model evaluation are generated here.")
                    if problem_type == 'Classification':
                        # Explicitly call plots here if needed outside display_results
                        st.write("*(Confusion Matrix shown in Metrics tab)*")
                        display_feature_importances(trained_model, encoded_features_tuple)
                    else:  # Regression
                        st.write("*(Residuals Plot shown in Metrics tab)*")
                        display_feature_importances(trained_model, encoded_features_tuple)

                # --- Data Info Tab ---
                with tab_data_info:
                    st.subheader("Dataset Summary Statistics (Original)")
                    st.dataframe(df_input.describe(include='all'))
                    st.subheader("Analysis Setup Summary")
                    st.write(f"**Algorithm:** {algorithm}")
                    st.write(f"**Problem Type:** {problem_type}")
                    st.write(f"**Target Column:** {target_column}")
                    st.write(f"**Original Features Selected:** {len(features_tuple)}")
                    st.write(f"**Features After Encoding:** {len(encoded_features_tuple)}")
                    st.write(f"**Test Set Size:** {test_size:.2f}")
                    st.write("**Model Hyperparameters:**")
                    st.json(model_params)

                st.balloons()
                logging.info(f"Analysis and evaluation for {algorithm} completed successfully.")

            except Exception as e:
                st.error(f"❌ An error occurred during model prediction or evaluation display: {e}")
                logging.error(f"Prediction/Evaluation error: {e}", exc_info=True)
    else:
        st.error(f"❌ Model training failed for {algorithm}. Cannot evaluate.")
        logging.error(f"Model training function returned None for {algorithm}.")


# --- Streamlit App UI --- (No changes from v3.2)

def main_ui():
    st.set_page_config(page_title="Tree Ensemble Runner", layout="wide", initial_sidebar_state="expanded")

    st.sidebar.title("⚙️ Configuration")
    st.sidebar.markdown(f"`App Version: {APP_VERSION}`")

    st.sidebar.subheader("1. Load Data")
    input_method = st.sidebar.radio(
        "Choose data source:",
        ("Select from Directory", "Upload CSV File"),
        horizontal=True,
        help="Load data from a local directory or upload a file directly."
    )

    df_loaded = None
    data_source_name = None

    if input_method == "Select from Directory":
        data_dir = st.sidebar.text_input("Data Directory Path:", value=DEFAULT_DATA_DIR,
                                         help="Path to the folder containing your CSV files.")
        csv_files = list_csv_files(data_dir)
        if not csv_files:
            st.info(f"💡 Tip: Place your CSV files inside the '{data_dir}' folder.")
            st.stop()
        selected_file = st.sidebar.selectbox("Select Dataset:", csv_files, help="Choose the CSV file to analyze.")
        if selected_file:
            file_path = os.path.join(data_dir, selected_file)
            with st.spinner(f"Loading data from '{selected_file}'..."):
                df_loaded = load_data(file_path)
            data_source_name = selected_file

    else:
        uploaded_file = st.sidebar.file_uploader("Upload your CSV file:", type="csv",
                                                 help="Upload a CSV file from your computer.")
        if uploaded_file is not None:
            with st.spinner(f"Loading data from uploaded file '{uploaded_file.name}'..."):
                df_loaded = load_data(uploaded_file)
            data_source_name = uploaded_file.name

    if data_source_name:
        st.title(f"🌳 Tree Ensemble Analysis: `{data_source_name}`")
    else:
        st.title(f"🌳 Tree Ensemble Analysis Tool")

    st.markdown("---")

    if df_loaded is not None:
        with st.expander("📂 Dataset Preview & Info", expanded=False):
            st.dataframe(df_loaded.head())
            st.write(f"**Shape:** {df_loaded.shape[0]} rows, {df_loaded.shape[1]} columns")
            cols_info = pd.DataFrame({
                'Column': df_loaded.columns,
                'DataType': df_loaded.dtypes.astype(str),
                'Non-Null Count': df_loaded.notna().sum(),
                'Null Count': df_loaded.isna().sum()
            }).reset_index(drop=True)
            st.dataframe(cols_info)

        st.sidebar.divider()
        st.sidebar.subheader("2. Analysis Setup")
        columns = df_loaded.columns.tolist()

        potential_targets = [col for col in columns if
                             col.lower().replace(" ", "_") in ['target', 'label', 'class', 'anomaly', 'suspicious',
                                                               'output', 'result']]
        default_target = potential_targets[0] if potential_targets else columns[-1]
        try:
            default_target_index = columns.index(default_target)
        except ValueError:
            default_target_index = len(columns) - 1

        target_column = st.sidebar.selectbox("🎯 Select Target Column:", columns, index=default_target_index,
                                             help="The column you want to predict.")

        default_features = [col for col in columns if col != target_column]
        features_to_use = st.sidebar.multiselect("✨ Select Feature Columns:", columns, default=default_features,
                                                 help="Columns used to predict the target.")

        if not features_to_use:
            st.sidebar.error("Please select at least one feature.")
            st.stop()
        if target_column in features_to_use:
            st.sidebar.error("Target column cannot also be a feature.")
            st.stop()

        problem_type_options = ['Classification', 'Regression']
        target_dtype_kind = df_loaded[target_column].dtype.kind
        unique_target_values = df_loaded[target_column].nunique()

        if target_dtype_kind in ['O', 'b', 'S']:
            detected_type = 'Classification'
        elif pd.api.types.is_numeric_dtype(df_loaded[target_column]):
            if pd.api.types.is_float_dtype(df_loaded[target_column]) or unique_target_values > min(25,
                                                                                                   len(df_loaded) * 0.05):
                detected_type = 'Regression'
            else:
                detected_type = 'Classification'
                st.sidebar.info(
                    f"Target '{target_column}' is numeric with few unique values ({unique_target_values}). Assuming Classification.")
        else:
            detected_type = 'Classification'
            logging.warning(
                f"Could not reliably detect type for target '{target_column}' (dtype: {df_loaded[target_column].dtype}). Defaulting to Classification.")

        problem_type = st.sidebar.selectbox("Problem Type:", problem_type_options,
                                            index=problem_type_options.index(detected_type),
                                            help="Select 'Classification' for discrete categories or 'Regression' for continuous values.")

        if problem_type == 'Regression' and target_dtype_kind in ['O', 'b', 'S']:
            st.sidebar.warning(
                f"⚠️ Target column '{target_column}' appears non-numeric. Selecting 'Regression' may cause errors.")

        st.sidebar.divider()
        st.sidebar.subheader("3. Model Configuration")
        selected_algorithm = st.sidebar.selectbox("Select Algorithm:", SUPPORTED_ALGORITHMS)

        with st.sidebar.expander(f"Tune {selected_algorithm} Hyperparameters", expanded=True):
            h_col1, h_col2 = st.columns(2)
            n_estimators = h_col1.slider("Num. Trees:", 10, 1000, 100, 10,
                                         help="Number of trees in the forest (n_estimators).")
            max_depth_val = h_col2.slider("Max Depth:", 3, 50, 10, 1,
                                          help="Maximum depth of the trees. Set to 50 for 'None' (unlimited depth).")
            max_depth = None if max_depth_val >= 50 else max_depth_val

            min_samples_split = h_col1.slider("Min Samples Split:", 2, 20, 2, 1,
                                              help="Minimum number of samples required to split an internal node.")
            min_samples_leaf = h_col2.slider("Min Samples Leaf:", 1, 20, 1, 1,
                                             help="Minimum number of samples required to be at a leaf node.")

        model_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf
        }

        st.sidebar.divider()
        st.sidebar.subheader("4. Data Split")
        test_size = st.sidebar.slider("Test Set Size:", 0.1, 0.5, 0.25, 0.05, format="%.2f",
                                      help="Proportion of the dataset to hold out for testing.")

        st.sidebar.divider()
        if st.sidebar.button(f"🚀 Run {selected_algorithm} Analysis", use_container_width=True, type="primary"):
            logging.info(f"Starting analysis run for algorithm: {selected_algorithm}")
            # Pass the loaded DataFrame directly
            run_analysis(selected_algorithm, df_loaded, target_column, tuple(features_to_use), problem_type,
                         model_params, test_size)

    else:
        st.info("👈 Please select or upload a CSV dataset using the sidebar to begin.")

    st.sidebar.divider()
    st.sidebar.markdown("---")
    show_code=""
    #show_code = st.sidebar.checkbox("Show App Source Code")
    if show_code:
        st.subheader("🐍 Application Source Code (`app.py`)")
        try:
            script_path = os.path.realpath(__file__)
            with open(script_path, 'r') as f:
                st.code(f.read(), language='python')
        except Exception as e:
            st.warning(f"Could not display source code: {e}")


if __name__ == "__main__":
    main_ui()
