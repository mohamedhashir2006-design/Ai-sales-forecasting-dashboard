# 📊 AI Sales Forecasting & Recommendation Dashboard

An interactive AI-powered dashboard for **sales forecasting, trend analysis, anomaly detection, and business insights** using Machine Learning models like **Prophet and LSTM**.

---

## 🚀 Features

* 📈 **Sales Forecasting (Prophet)**
* 🤖 **LSTM Deep Learning Model**
* 📊 **Trend & Seasonality Analysis**
* 🚨 **Anomaly Detection (Spikes & Drops)**
* 📅 **Weekly Sales Insights**
* 🥧 **Sales Channel Distribution**
* 🧠 **AI Business Insights (Ollama / LLM)**
* 🔁 **Model Comparison (Prophet vs LSTM)**

---

## 🖥️ Dashboard Preview

* Forecast visualization
* Trend analysis
* Weekly seasonality
* Model comparison
* AI-generated insights

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **Visualization:** Plotly
* **Data Processing:** Pandas, NumPy
* **Forecasting:** Prophet
* **Deep Learning:** TensorFlow (LSTM)
* **AI Insights:** Ollama (LLM)

---

## 📂 Project Structure

```
sales_dashboard/
│
├── app.py
├── requirements.txt
├── data/
│   ├── sales_history.csv
│   └── interaction_history.csv
│
├── modules/
│   ├── forecasting/
│   │   ├── prophet_model.py
│   │   ├── lstm_model.py
│   │   └── preprocess.py
│   │
│   ├── recommendation/
│   │   └── model.py
│   │
│   ├── decision_engine.py
│   └── chatbot.py
│
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/mohamedhashir2006-design/Ai-sales-forecasting-dashboard.git
cd Ai-sales-forecasting-dashboard
```

---

### 2️⃣ Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Mac/Linux
```

---

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Run the app

```bash
streamlit run app.py
```

---

## 🤖 AI Insights (Optional)

Make sure Ollama is running:

```bash
ollama run gemma:2b
```

---

## 📊 Key Functionalities Explained

### 🔹 Forecasting

Uses **Facebook Prophet** to predict future sales trends.

### 🔹 LSTM Model

Deep learning model capturing complex patterns in time series.

### 🔹 Anomaly Detection

Detects unusual spikes or drops in sales using percentage change.

### 🔹 Model Comparison

Compares Prophet vs LSTM performance visually.

### 🔹 AI Insights

Generates business insights like:

* Best/worst sales day
* Top-performing channels
* Recommendations

---

## 📌 Future Improvements

* 🔮 Real-time data integration
* 📉 Advanced anomaly detection (ML-based)
* 🌐 Cloud deployment
* 📊 More KPIs & dashboards

---

## 👨‍💻 Author

**Mohamed Hashir**

---


---
