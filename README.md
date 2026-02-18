# bankfraudML — Transaction Intelligence Platform

A high-performance fraud detection system leveraging **Random Forest** and **K-Medoids Clustering** to analyze transaction patterns. Built with FastAPI and a Cyberpunk-inspired terminal UI.



## 📂 Project Structure

```text
bankfraudML/
├── .venv/               # Virtual environment
├── app/
│   ├── models/          # Trained .pkl model files
│   ├── main.py          # FastAPI application & routes
│   └── modelProcess.py  # ML logic & Feature engineering
├── static/
│   └── css/             # im lazy so I didn't use it
├── templates/
│   └── index.html       # Terminal UI
├── bank_data.db         # SQLite database
├── dataPre.py           # Data initialization script
└── requirements.txt     # Project dependencies
```



## 🛠 Prerequisites

* **Python:** 3.11.x (Strictly preferred)
* **OS:** Windows (Command usage optimized for PowerShell)

## 🚀 Setup & Execution

### 1. Environment Activation
Activate the virtual environment to ensure all dependencies are isolated.

```powershell
# In Windows PowerShell
& ./.venv/Scripts/Activate.ps1
```

### 2. Dependency Installation
Install the necessary ML and Web frameworks:

```powershell
pip install -r requirements.txt
```

### 3. Data Preprocessing
You **must** run the preprocessing script first to initialize the SQLite database and prepare the lookup tables for the models.

```powershell
python dataPre.py
```

### 4. Launch the API
Start the FastAPI server. The entry point is located in the `app` directory.

```powershell
uvicorn app.main:app --reload
```



## 🧪 API Usage

Once the server is live at `http://127.0.0.1:8000`:

1. **Terminal UI:** Access the interactive dashboard via your browser.
2. **REST Endpoint:** Send `POST` requests to `/predict` with the following JSON shape:

```json
{
  "step": 179,
  "customer": "C1234567",
  "age": 3,
  "gender": "M",
  "merchant": "M1823072687",
  "category": "food",
  "amount": 150.00
}
```


## 🧑🏿‍💻 Other notes:
This is just my own archived version of my works in the unit COS30049 - Innovation Tech Project at Swinburne Vietnam.
This is one of the earliest ML product in my IT career. Hope I can get far in this journey:)