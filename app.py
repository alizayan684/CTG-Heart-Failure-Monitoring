from design import UI
from PyQt5.QtWidgets import QFileDialog
import pandas as pd
from PyQt5.QtWidgets import QTableWidgetItem
import numpy as np
from sklearn.preprocessing import StandardScaler
from math import floor
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report
import joblib
class App(UI):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.show()
        self.cleaned_data = None
        self.upload.clicked.connect(self.upload_data)
        self.analyse.clicked.connect(self.preprocess_data)
        self.diagnose.clicked.connect(self.diagnose_data)
        self.save.clicked.connect(self.save_data)
        self.model = joblib.load('random_forest_model.pkl')
        

    def save_data(self):
        if self.cleaned_data is not None:
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Cleaned Data", "", "CSV Files (*.csv)", options=options)
            if file_path:
                self.cleaned_data.to_csv(file_path, index=False, header=True)

    def upload_data(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open CTG CSV Data", "", "CSV Files (*.csv)", options=options)

        if file_path:
            self.data = pd.read_csv(file_path)
            self.actual = floor(self.data['fetal_health'].mean())
            self.data.drop(columns=['fetal_health'], inplace=True)
            data_summary = self.data['FHR'].describe()
            print(data_summary)
            self.table_widget.setRowCount(len(data_summary))
            self.table_widget.setColumnCount(1)
            self.table_widget.setHorizontalHeaderLabels(['value'])
            self.table_widget.setVerticalHeaderLabels(data_summary.index)
            for i , (index, value) in enumerate(data_summary.items()):
                self.table_widget.setItem(i, 0, QTableWidgetItem(str(value)))
            self.plot_data()
    def plot_data(self):
        self.draw_widget.figure.clear()
        ax = self.draw_widget.figure.add_subplot(111)
        ax.plot(self.data['second'], self.data['FHR'])
        self.draw_widget.draw()
    
    def plot_spec_data(self, data):
        self.draw_widget.figure.clear()
        ax = self.draw_widget.figure.add_subplot(111)
        ax.plot(self.data['second'], data)
        self.draw_widget.draw()
    
    def preprocess_data(self):
        fhr = self.data['FHR']
        fhr = self.clean_fhr(fhr)
        self.cleaned_data = self.data.copy()
        self.cleaned_data.insert(0, 'baseline value', fhr)
        # update the table
        data_summary = fhr.describe()
        self.table_widget.setRowCount(len(data_summary))
        self.table_widget.setColumnCount(1)
        self.table_widget.setHorizontalHeaderLabels(['value'])
        self.table_widget.setVerticalHeaderLabels(data_summary.index)
        # set the labels size to fit the content
        self.table_widget.resizeColumnsToContents()
        for i , (index, value) in enumerate(data_summary.items()):
                self.table_widget.setItem(i, 0, QTableWidgetItem(str(value)))
        
        self.plot_spec_data(fhr)

    def clean_fhr(self,fhr, show_figure=False):
        '''
        Cleans fetal heart rate (FHR) signal.
        Inputs:
        fhr - series, the "FHR" column from one of the csv files
        show_figure - boolean, whether to display the FHR before and after cleaning
        Outputs:
        fhr - series, clean version of the FHR column
        '''
        if show_figure:
            # Show original trace
            self.plot_spec_data(fhr)

        # Replace 0 with NaN
        fhr.replace(0, np.nan, inplace=True)

        # Remove NaN if they occured for more than 15 seconds consecutively
        na = fhr.isnull()
        fhr = fhr[~(fhr.groupby(na.ne(na.shift()).cumsum().values).transform('size').ge(61) & na)].reset_index(drop=True)

        # Set outliers to NaN
        fhr[fhr < 50] = np.nan
        fhr[fhr > 200] = np.nan

        # Replace missing values using linear interpolation
        fhr = fhr.interpolate(method='linear')

        # Find how each value has changed from the prior value
        diff = fhr - fhr.shift()

        # Where difference is more than +- 25, set as NaN
        fhr[(diff > 25) | (diff < -25)] = np.nan

        # Replace missing values using linear interpolation
        fhr = fhr.interpolate(method='linear')

        if show_figure:
            # Show clean trace
            self.plot_spec_data(fhr)
        return(fhr)
    
    def diagnose_data(self):
        if not hasattr(self, 'cleaned_data'):
            return
        # Extract features
        X = self.cleaned_data.drop(columns= ["second", "minute", "FHR"] , axis=1)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        col_names = list(X.columns)
        X_df = pd.DataFrame(X_scaled, columns=col_names)
        length = X_df.shape[0]
        
        for col in X_df.columns:
             X_df[col] = [X_df[col][5]]*length
        

        predictions = self.model.predict(X_df)
        print(predictions)
        self.predictions = predictions.mean()
        self.table_widget.setVerticalHeaderLabels(["predictions"])
        self.table_widget.setHorizontalHeaderLabels(['value'])
        if self.predictions == 1:
                self.table_widget.setItem(0, 0, QTableWidgetItem("Normal"))
        elif self.predictions == 2:
                self.table_widget.setItem(0, 0, QTableWidgetItem("Suspect"))
        else:
                self.table_widget.setItem(0, 0, QTableWidgetItem("Pathological"))

    
    def plot_results(self):
        self.draw_widget.figure.clear()
        ax = self.draw_widget.figure.add_subplot(111)
        y_true = self.actual

        # Ensure y_true and y_prediction are arrays
        if not isinstance(y_true, np.ndarray):
            y_true = np.array([y_true])
        if not isinstance(self.predictions, np.ndarray):
            self.predictions = np.array([self.predictions])

        # Binarize the output labels
        y_true_binarized = label_binarize(y_true, classes=[1, 2, 3])
        y_pred_binarized = label_binarize(self.predictions, classes=[1, 2, 3])

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(y_true_binarized.shape[1]):
            fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_pred_binarized[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot ROC curve
        for i in range(y_true_binarized.shape[1]):
            if i + 1 == 1:
                cl = "Normal"
            elif i + 1 == 2:
                cl = "Suspect"
            else:
                cl = "Pathological"
            ax.plot(fpr[i], tpr[i], label=f'ROC curve of class {i+1} : {cl} (area = {roc_auc[i]:.2f})')

        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc='lower right')
        self.draw_widget.draw()




if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    ctg_app = App()
    sys.exit(app.exec_())