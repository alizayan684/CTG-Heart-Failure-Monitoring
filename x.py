import sys
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QTextEdit

class CTGApp(QWidget):
    def __init__(self):
        super().__init__()

        # Load the trained Random Forest model
        self.model = joblib.load('random_forest_model.pkl')
        
        # Initialize GUI components
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("CTG Analysis - Heart Failure Detection")
        self.setGeometry(100, 100, 600, 400)

        layout = QVBoxLayout()

        # Button to upload data
        self.upload_button = QPushButton("Upload CTG Data (CSV)", self)
        self.upload_button.clicked.connect(self.upload_data)
        layout.addWidget(self.upload_button)

        # Button to process data
        self.process_button = QPushButton("Process Data", self)
        self.process_button.clicked.connect(self.process_data)
        layout.addWidget(self.process_button)

        # Button to diagnose heart failure
        self.diagnose_button = QPushButton("Diagnose Heart Failure", self)
        self.diagnose_button.clicked.connect(self.diagnose_heart_failure)
        layout.addWidget(self.diagnose_button)

        # Button to make a report
        self.report_button = QPushButton("Make Report", self)
        self.report_button.clicked.connect(self.make_report)
        layout.addWidget(self.report_button)

        # Display area for data and results
        self.data_display = QTextEdit(self)
        layout.addWidget(self.data_display)

        self.setLayout(layout)

    def upload_data(self):
        # File dialog to upload CSV data
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open CTG CSV Data", "", "CSV Files (*.csv)", options=options)

        if file_path:
            self.data = pd.read_csv(file_path)
            self.data_display.setPlainText(str(self.data.head()))  # Show first few rows of the data

    def process_data(self):
        # Preprocess data: Standardize, Clean, and Filter relevant columns
        if hasattr(self, 'data'):
            # Example: Assuming 'fetal_health' is the target and rest are features
            X = self.data.drop("fetal_health", axis=1)
            y = self.data["fetal_health"]

            # Standard scaling of features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Clean and filter out outliers or missing data if necessary (this is a placeholder)
            self.cleaned_data = pd.DataFrame(X_scaled, columns=X.columns)

            # Display the processed data
            self.data_display.setPlainText(f"Processed Data:\n{self.cleaned_data.describe().T}")
        else:
            self.data_display.setPlainText("Please upload the data first.")

    def diagnose_heart_failure(self):
        if hasattr(self, 'data'):
            # Preprocess data (same as process_data)
            X = self.data.drop("fetal_health", axis=1)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            col_names = list(X.columns)
            X_df = pd.DataFrame(X_scaled, columns=col_names) 
            # Predict with the loaded model
            predictions = self.model.predict(X_df)
            accuracy = accuracy_score(self.data["fetal_health"], predictions)

            # Display prediction results
            self.data_display.setPlainText(f"Heart Failure Diagnosis:\nAccuracy: {accuracy * 100:.2f}%\nPredictions: {predictions[:10]}...")

        else:
            self.data_display.setPlainText("Please upload the data first.")

    def make_report(self):
        if hasattr(self, 'data'):
            # Generate a simple report (customize as needed)
            X = self.data.drop("fetal_health", axis=1)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Predict with the model
            predictions = self.model.predict(X_scaled)

            # Get biological insights (here we use basic info like mean values of features)
            report = "Patient Heart Condition Report\n"
            report += "------------------------------\n"
            report += f"Number of observations: {len(self.data)}\n"
            report += f"Predictions (indicating possible heart failure): {np.sum(predictions)}\n"
            report += f"Biological Information (Feature Means):\n"
            report += f"{self.data.describe().T[['mean']]}"

            # Display the report
            self.data_display.setPlainText(report)
        else:
            self.data_display.setPlainText("Please upload the data first.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ctg_app = CTGApp()
    ctg_app.show()
    sys.exit(app.exec_())
