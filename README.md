# 🌾 AgriSense - Crop Yield Prediction System

An AI-driven precision agriculture platform designed to predict crop yields with high accuracy by integrating machine learning with deep **agronomic domain knowledge**. Built for farmers, agronomists, and researchers to optimize land productivity.

---

## 🚀 Key Features

- **Multi-Crop Support**: Tailored predictions for Rice, Tea, Rubber, Sugarcane, and Cinnamon.
- **Agronomic Intelligence**: Incorporates "Good Zone" indicators, heat stress, drought risk, and nutrient balance (NPK) analysis.
- **Real-time Analytics**: Interactive dashboards showing model performance, feature importance, and historical trends.
- **Secure Data Storage**: Integration with Supabase for persistent storage of prediction history.
- **Responsive UI**: A premium, mobile-first frontend built with Next.js and Tailwind CSS.
- **ML Interpretability**: Visualizes the top factors affecting every single prediction.

---

## 🛠️ Technical Stack

### **Frontend**
- **Framework**: Next.js 14 (React)
- **Styling**: Tailwind CSS
- **Animations**: Framer Motion
- **Charts**: Chart.js & React-Chartjs-2
- **Icons**: Lucide React

### **Backend**
- **API**: Flask (Python)
- **Database**: Supabase (PostgreSQL)
- **Environment**: Python Dotenv

### **Machine Learning**
- **Core Engine**: XGBoost (Extreme Gradient Boosting)
- **Analysis**: Scikit-learn, Pandas, NumPy
- **Feature Engineering**: Custom agronomic scaling and stress-indicator logic.
- **Persistence**: Joblib & JSON-based feature scaling.

---

## 📂 Project Architecture

```text
crop-yield-prediction/
├── backend/               # Flask API, Business Logic & Database Integration
├── frontend/              # Next.js Application & Interactive Dashboards
├── data/                  # Raw and Raw Agronomic Datasets
├── models/                # Trained .pkl models, Feature Scalers, and Metadata
├── scripts/               # Training, Data Cleaning, and Diagnostic Scripts
│   ├── train_xgboost.py   # Primary model training pipeline
│   └── analyze_data.py    # Statistical analysis tools
├── reports/               # Agronomic documentation and diagnostic logs
└── .gitignore             # Standardized git exclusion rules
```

---

## 🧠 Agronomic Domain Logic

Unlike standard regressions, this system applies agricultural constraints to prevent unrealistic predictions:
- **Optimization Zones**: Models are sensitized to optimal ranges (e.g., Tea performs best between 18-26°C).
- **Stress Penalties**: Automatically identifies and accounts for drought stress (low rain + high temp) and nutrient imbalances.
- **Non-Negative Constraints**: Post-processing layer ensures yields never drop below 0, regardless of extreme input data.

---

## 🚥 Getting Started

### **1. Clone the Repository**
```bash
git clone https://github.com/devthuva27/crop-yield-prediction.git
cd crop-yield-prediction
```

### **2. Backend Setup**
```bash
cd backend
python -m venv venv
source venv/bin/scripts/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

### **3. Frontend Setup**
```bash
cd ../frontend
npm install
npm run dev
```

---

## 📊 Database & Environment
Create a `.env` file in the `backend/` directory with the following:
```env
# Server
PORT=5000
HOST=localhost

# Supabase
SUPABASE_URL=your_project_url
SUPABASE_KEY=your_api_key
```

---

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

---

**Developed for Precision Agriculture & Sustainable Farming.**
