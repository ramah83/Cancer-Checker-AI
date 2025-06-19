import os
import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import logging
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'colorectal_cancer_prediction_key'

models = {}
scaler = None
feature_names = []
X_train_data = None
X_test_data = None
y_train_data = None
y_test_data = None
le = LabelEncoder()

def load_and_process_data():
    global scaler, feature_names, X_train_data, X_test_data, y_train_data, y_test_data
    
    try:
        logger.info("Loading and processing data")
        
        data_path = r"C:\Users\Rammah\OneDrive\Desktop\AI_Lecs\Real project\Colorectal Cancer Patient Data.csv"
        
        if not os.path.exists(data_path):
            logger.warning(f"Data file not found at {data_path}. Using embedded sample data.")
            df = create_sample_data()
        else:
            df = pd.read_csv(data_path)
        
        df.dropna(inplace=True)
        
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = le.fit_transform(df[col].astype(str).str.strip().str.lower())
        
        drop_cols = [col for col in ["DFS event", "Unnamed: 0", "ID_REF"] if col in df.columns]
        X = df.drop(drop_cols, axis=1)
        feature_names = X.columns.tolist()
        logger.info(f"Features: {feature_names}")
        
        y = df["DFS event"]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        X_train_data = X_train_scaled
        X_test_data = X_test_scaled
        y_train_data = y_train
        y_test_data = y_test
        
        logger.info("Data processing completed successfully")
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    except Exception as e:
        logger.error(f"Error in load_and_process_data: {str(e)}")
        return create_and_process_sample_data()

def create_sample_data():
    """Create a sample dataset based on the original structure"""
    logger.info("Creating sample data")
    
    sample_df = pd.DataFrame({
        'Gender': ['male', 'female', 'male', 'female'] * 25,
        'Age': np.random.randint(30, 80, 100),
        'BMI': np.random.uniform(18.5, 35, 100),
        'T_Stage': np.random.randint(1, 5, 100),
        'N_Stage': np.random.randint(0, 3, 100),
        'M_Stage': np.random.randint(0, 2, 100),
        'DFS event': np.random.randint(0, 2, 100),
        'Unnamed: 0': range(100),
        'ID_REF': [f'P{i}' for i in range(100)]
    })
    
    return sample_df

def create_and_process_sample_data():
    global scaler, feature_names, X_train_data, X_test_data, y_train_data, y_test_data
    
    try:
        df = create_sample_data()
        
        df.dropna(inplace=True)
        
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = le.fit_transform(df[col].astype(str).str.strip().str.lower())
        
        drop_cols = [col for col in ["DFS event", "Unnamed: 0", "ID_REF"] if col in df.columns]
        X = df.drop(drop_cols, axis=1)
        feature_names = X.columns.tolist()
        
        y = df["DFS event"]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        X_train_data = X_train_scaled
        X_test_data = X_test_scaled
        y_train_data = y_train
        y_test_data = y_test
        
        logger.info("Sample data processing completed successfully")
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    except Exception as e:
        logger.error(f"Error in create_and_process_sample_data: {str(e)}")
        return None, None, None, None

def train_models(X_train, y_train):
    """Train KNN and Naive Bayes models"""
    global models
    
    try:
        logger.info("Training models")
        
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train, y_train)
        models['knn'] = knn
        
        nb = GaussianNB()
        nb.fit(X_train, y_train)
        models['nb'] = nb
        
        logger.info("Models trained successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error in train_models: {str(e)}")
        return False

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model and return metrics and plots"""
    try:
        logger.info(f"Evaluating {model_name} model")
        
        y_pred = model.predict(X_test)
        
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        
        plt.figure(figsize=(6, 4))
        cmap = "Blues" if model_name == "KNN" else "Greens"
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=cmap)
        plt.title(f"{model_name} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()
        
        encoded = base64.b64encode(image_png).decode('utf-8')
        plot_data = f"data:image/png;base64,{encoded}"
        
        logger.info(f"Model evaluation completed with accuracy: {acc}")
        return {
            'report': report,
            'confusion_matrix': conf_matrix.tolist(),
            'accuracy': acc,
            'plot': plot_data
        }
    
    except Exception as e:
        logger.error(f"Error in evaluate_model: {str(e)}")
        return {
            'report': {},
            'confusion_matrix': [[0, 0], [0, 0]],
            'accuracy': 0,
            'plot': ''
        }

@app.route('/')
def index():
    """Render the home page"""
    return render_template('index.html', features=feature_names)

@app.route('/results')
def results():
    """Display model results"""
    try:
        if not models:
            flash('Models not available. Please restart the application.')
            return redirect(url_for('index'))

        X_test = X_test_data if X_test_data is not None else None
        y_test = y_test_data if y_test_data is not None else None

        if X_test is None or y_test is None:
            flash('Test data not available. Please restart the application.')
            return redirect(url_for('index'))

        knn_results = evaluate_model(models['knn'], X_test, y_test, "KNN")
        nb_results = evaluate_model(models['nb'], X_test, y_test, "Naive Bayes")

        return render_template(
            'results.html',
            knn_results=knn_results,
            nb_results=nb_results
        )

    except Exception as e:
        logger.error(f"Error in results route: {str(e)}")
        flash(f'An error occurred: {str(e)}')
        return redirect(url_for('index'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Handle prediction for new data"""
    try:
        if request.method == 'GET':
            return render_template('predict.html', features=feature_names)
        
        if request.method == 'POST':
            if not models or 'knn' not in models or 'nb' not in models:
                flash('Models not available. Please restart the application.')
                return redirect(url_for('index'))
            
            feature_values = []
            for feature in feature_names:
                value = request.form.get(feature)
                try:
                    feature_values.append(float(value))
                except:
                    flash(f'Invalid value for {feature}')
                    return redirect(url_for('predict'))
            
            input_data = np.array(feature_values).reshape(1, -1)
            
            if scaler:
                input_data = scaler.transform(input_data)
            
            knn_pred = models['knn'].predict(input_data)[0]
            knn_prob = models['knn'].predict_proba(input_data)[0]
            
            nb_pred = models['nb'].predict(input_data)[0]
            nb_prob = models['nb'].predict_proba(input_data)[0]
            
            results = {
                'knn': {
                    'prediction': int(knn_pred),
                    'probability': float(knn_prob[int(knn_pred)]) * 100
                },
                'nb': {
                    'prediction': int(nb_pred),
                    'probability': float(nb_prob[int(nb_pred)]) * 100
                }
            }
            
            return render_template('prediction_result.html', results=results)
    
    except Exception as e:
        logger.error(f"Error in predict route: {str(e)}")
        flash(f'An error occurred: {str(e)}')
        return redirect(url_for('index'))

if __name__ == '__main__':
    os.makedirs('static/img', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)

    X_train, X_test, y_train, y_test = load_and_process_data()

    if X_train is not None and y_train is not None:
        train_models(X_train, y_train)

        X_train_data = X_train
        X_test_data = X_test
        y_train_data = y_train
        y_test_data = y_test

    app.run(debug=True)