import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import re
from itertools import cycle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
from imblearn.over_sampling import SMOTE


# 1. Create models folder
os.makedirs("models", exist_ok=True)


# 2. Load CICIDS dataset files
csv_files = [
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
]

data_list = []

for file in csv_files:
    if not os.path.exists(file):
        raise FileNotFoundError(f"{file} not found!")
    
    df = pd.read_csv(file, encoding="utf-8-sig", low_memory=False)
    df.columns = df.columns.str.strip()  # Clean column names
    df = df.dropna(subset=["Label"])     # Drop rows without labels
    df["Label"] = df["Label"].str.strip()
    
    data_list.append(df)

data = pd.concat(data_list, ignore_index=True)
print("Combined dataset shape:", data.shape)
print("Unique labels in dataset:", data["Label"].unique())


# 3. Clean labels
# Robust label cleaning
def clean_label(label):
    if not isinstance(label, str):
        return label
    label = label.lower()
    # Replace any type of dash or weird character with standard "-"
    label = re.sub(r"[–—�]", "-", label)
    # Remove any other non-alphanumeric characters except dash and space
    label = re.sub(r"[^a-z0-9\- ]", "", label)
    # Collapse multiple spaces
    label = re.sub(r"\s+", " ", label)
    label = label.strip()
    return label

data["Label"] = data["Label"].apply(clean_label)

# 4. Map to unified attack classes
label_map = {
    "benign": "Normal",
    "bot": "Bot",
    "ddos": "DDoS",
    "dos hulk": "DDoS",
    "portscan": "PortScan",
    "ftp-patator": "BruteForce",
    "ssh-patator": "BruteForce",
    "web attack - brute force": "WebAttack",
    "web attack - xss": "WebAttack",
    "web attack - sql injection": "WebAttack",
    "web attack - file inclusion": "WebAttack",
    "web attack - directory traversal": "WebAttack",
}

data["Label"] = data["Label"].map(label_map)
data = data.dropna(subset=["Label"])

print("Labels after mapping:", data["Label"].unique())


# 5. Encode labels
le = LabelEncoder()
data['Label'] = le.fit_transform(data['Label'])
print("All attack classes:", le.classes_)


# 6. Prepare features & target
# Define the columns that are non-numeric or metadata
cols_to_drop = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp', 'Label', 'Binary_Label']

# Drop them safely (only if they exist in the dataframe)
X = data.drop(columns=[c for c in cols_to_drop if c in data.columns])
y = data["Label"]

print(f"Features used for training: {X.columns.tolist()[:5]}... (Total: {len(X.columns)})")

# 7️. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# 8️. Scaling & handle NaNs
X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# 9️. Handle class imbalance
sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)


# 10. Train only Random Forest (fast & stable)
best_model = RandomForestClassifier(
    n_estimators=50,   # faster than 150
    max_depth=20,      #  limits tree size
    max_features='sqrt',  # each split uses sqrt(n_features)
    random_state=42,
    n_jobs=-1
)
    
print("\nTraining Random Forest...")
best_model.fit(X_train, y_train)
best_score = best_model.score(X_test, y_test)
best_name = "Random Forest"
print(f"\n Training completed! Best Model: {best_name} with Accuracy: {best_score:.4f}")


# 1️0.1  Confusion Matrix
all_classes = le.classes_
cm = confusion_matrix(y_test, best_model.predict(X_test), labels=np.arange(len(all_classes)))

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=all_classes, yticklabels=all_classes)
plt.title(f"Confusion Matrix - {best_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# 1️0.2️ Classification Report & Metrics
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=all_classes))

precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
print("Overall Precision:", precision)
print("Overall Recall:", recall)
print("Overall F1 Score:", f1)

plt.figure(figsize=(6,4))
metrics = ["Precision", "Recall", "F1"]
values = [precision, recall, f1]
plt.bar(metrics, values, color=["blue","green","orange"])
plt.ylim(0,1)
plt.title("Overall Performance Metrics")
plt.show()


# 1️0.3️ ROC Curve per class
y_test_bin = label_binarize(y_test, classes=np.arange(len(all_classes)))
y_score = best_model.predict_proba(X_test)

plt.figure(figsize=(10,8))
colors = cycle(['blue','red','green','orange','purple','brown','pink'])
for i, color in zip(range(len(all_classes)), colors):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=2, label=f"{all_classes[i]} (AUC={roc_auc:.2f})")

plt.plot([0,1], [0,1], 'k--', lw=1)
plt.title(f"Multi-Class ROC Curve - {best_name}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


# 1️0.4️ Precision-Recall Curve per class
plt.figure(figsize=(10,8))
for i, color in zip(range(len(all_classes)), colors):
    precision_vals, recall_vals, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
    ap = average_precision_score(y_test_bin[:, i], y_score[:, i])
    plt.plot(recall_vals, precision_vals, color=color, lw=2,
             label=f"{all_classes[i]} (AP={ap:.2f})")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"Precision-Recall Curve - {best_name}")
plt.legend(loc="lower left")
plt.grid(True)
plt.show()


# 1️0.5️ Feature Importance (Random Forest only)
importances = best_model.feature_importances_
features = X.columns
feat_imp = pd.Series(importances, index=features)
top_features = feat_imp.nlargest(10)
plt.figure(figsize=(8,6))
top_features.sort_values().plot(kind='barh', color='green')
plt.title("Top 10 Important Features")
plt.xlabel("Importance Score")
plt.show()



# 1️0.6️ Save model, scaler, label encoder

with open("models/best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)


print("\n Model training complete. Files saved in 'models/' folder.")

<------------------------DEPLOYMENT------------------------------->

# Routes 
from flask import Flask, render_template, request, jsonify,session, redirect
import os
import pandas as pd
import numpy as np
import pickle
import sqlite3
import datetime
import joblib

#  Flask App 
app = Flask(__name__)
app.secret_key = "super_secret_key"

#  Config 
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB


@app.errorhandler(413)
def request_entity_too_large(error):
    return "File too large", 413

MODEL_PATH = "models/best_model.pkl"
SCALER_PATH = "models/scaler.pkl"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"
DB_FILE = "user_predictions.db"

# Default in case loading fails
model = scaler = le = None
FEATURE_COUNT = 20

try:
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)      # or pickle.load(open(SCALER_PATH,"rb"))
        FEATURE_COUNT = scaler.n_features_in_
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    if os.path.exists(LABEL_ENCODER_PATH):
        le = joblib.load(LABEL_ENCODER_PATH)
except Exception as e:
    print("Error loading ML objects:", e)
    model = scaler = le = None

    
  
#  Database 
def get_db_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

with get_db_connection() as conn:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    """)
    conn.commit()

with get_db_connection() as conn:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            user TEXT,
            ip TEXT,
            attack_type TEXT,
            confidence REAL,
            risk INTEGER
        )
    """)
    conn.commit()

#  Utility 
def is_logged_in():
    return "user" in session


#  register 


from flask import render_template, request, redirect, session, jsonify

@app.route("/register", methods=["GET", "POST"])
def register():
    # 1. Redirect if already authenticated
    if "user" in session:
        return redirect("/dashboard")

    # 2. Serve the registration page
    if request.method == "GET":
        return render_template("register.html")

    # 3. Handle AJAX POST request
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "message": "Invalid request format."}), 400

        # Extract fields from your frontend payload
        username = data.get("username", "").strip()
        password = data.get("password", "").strip()
        fullname = data.get("fullname", "").strip()
        email = data.get("email", "").strip()

        # Basic Validation
        if not username or not password:
            return jsonify({"success": False, "message": "Username and password are required fields."}), 400

        # Database Insertion
        with get_db_connection() as conn:
            # Check if your table has fullname/email columns; 
            # if not, use: "INSERT INTO users (username, password) VALUES (?, ?)"
            conn.execute(
                "INSERT INTO users (username, password) VALUES (?, ?)",
                (username, password)
            )
            conn.commit()
            
        return jsonify({
            "success": True, 
            "message": "Access provisioned successfully! Redirecting to login..."
        })

    except sqlite3.IntegrityError:
        # This triggers if 'username' is a UNIQUE column in your DB
        return jsonify({"success": False, "message": "Operator ID (Username) already exists in the registry."}), 409
    
    except Exception as e:
        print(f"Registration Error: {e}")
        return jsonify({"success": False, "message": "Internal system error during provisioning."}), 500


#  Login 
@app.route("/login", methods=["GET", "POST"])
def login():
    if "user" in session:
        return redirect("/login")

    if request.method == "GET":
        return render_template("login.html")

    data = request.get_json()
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()

    with get_db_connection() as conn:
        user = conn.execute(
            "SELECT * FROM users WHERE username = ? AND password = ?",
            (username, password)
        ).fetchone()

    if user:
        session["user"] = username
        return jsonify({"success": True})
    else:
        return jsonify({"success": False, "message": "Invalid username or password"})
    
#  Logout 
@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect("/login")

#  -----------------Routes ---------------------
def is_logged_in():
    return "user" in session

@app.route("/")
def home():
    if "user" not in session:
        return render_template("login.html")
    return render_template("cyberindex.html")  # render the template directly

@app.route("/detection")
def prediction():
    if not is_logged_in():
        return redirect("/login")
    return render_template("detection.html")

@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect("/login")
    return render_template("cyberdashboard.html")

# ------------------- File Validation -------------------
ALLOWED_EXTENSIONS = {"csv", "zip"}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ------------------- Upload Route -------------------
from flask import Flask, request, jsonify, redirect, render_template, send_file
import math

# Upload Route (GET + POST)

UPLOAD_FOLDER = "uploads"

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if not is_logged_in():
        return jsonify({"error": "Unauthorized"}), 401

    if request.method == "GET":
        return render_template("upload.html")

    try:
        if "datasetFile" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["datasetFile"]
        if not file.filename.strip():
            return jsonify({"error": "Empty filename"}), 400

        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Read CSV
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            return jsonify({"error": f"Failed to read CSV: {str(e)}"}), 400

        # Detect label column or predict
        label_col = next((c for c in df.columns if c.lower() == "label"), None)
        if label_col:
            df["predicted_label"] = df[label_col]
        else:
            df_features = df.select_dtypes(include=[np.number]).fillna(0)
            if not (scaler and model and le):
                return jsonify({"error": "Model, scaler, or label encoder not initialized"}), 500
            X_scaled = scaler.transform(df_features)
            preds = model.predict(X_scaled)
            df["predicted_label"] = le.inverse_transform(preds)

        # Attack counts
        attack_counts = df["predicted_label"].value_counts().to_dict()
        attack_counts_safe = {str(k): int(v) for k, v in attack_counts.items()}  # ensure strings & ints

        # Top attack
        top_attack = max(attack_counts_safe.items(), key=lambda x: x[1])[0] if attack_counts_safe else "None"
        normal_traffic = attack_counts_safe.get("Normal", 0)

        # Save CSV result
        result_path = os.path.join(UPLOAD_FOLDER, "last_result.csv")
        df.to_csv(result_path, index=False)

        # Pagination
        page = int(request.args.get("page", 1))
        per_page = 50
        total_rows = len(df)
        total_pages = math.ceil(total_rows / per_page)
        start = (page - 1) * per_page
        end = start + per_page
        paginated_data = df.iloc[start:end].to_dict(orient="records")

        # Convert NaN/inf to None
        for row in paginated_data:
            for k, v in row.items():
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    row[k] = None

        return jsonify({
            "columns": list(df.columns),
            "data": paginated_data,
            "total_rows": total_rows,
            "page": page,
            "per_page": per_page,
            "total_pages": total_pages,
            "attack_counts": attack_counts_safe,
            "download_url": "/download_report",
            "top_attack": top_attack,
            "normal_traffic": normal_traffic
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Server error: {str(e)}"}), 500


@app.route("/download_report")
def download_report():
    path = os.path.join(UPLOAD_FOLDER, "last_result.csv")
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    return jsonify({"error": "No report available"}), 404

# ------------------- Prediction -------------------
# Solutions Mapping (YOUR TEXT)
solutions = {
    "DDoS": "Block suspicious IP addresses and enable rate limiting to mitigate attack.",
    "PortScan": "Update firewall rules and continuously monitor network ports.",
    "BruteForce": "Lock the affected account and enable multi-factor authentication (MFA).",
    "Infiltration": "Isolate the affected host and run a full antivirus scan.",
    "Bot": "Monitor bot activity, block suspicious IPs, and implement CAPTCHA where needed.",
    "WebAttack": "Patch web vulnerabilities, validate input, and monitor web traffic for anomalies.",
    "Normal": "No immediate action required. Continue monitoring system activity."
}


# Predict API
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        symptoms = data.get("symptoms", {})
        user = data.get("user", "Anonymous")
        ip = request.remote_addr

        
        # Convert Yes/No → 1/0
        slow = 1 if symptoms.get("slow_system") == "Yes" else 0
        failed = 1 if symptoms.get("failed_login") == "Yes" else 0
        conn = 1 if symptoms.get("unexpected_conn") == "Yes" else 0
        file_mod = 1 if symptoms.get("file_mod") == "Yes" else 0
        port = 1 if symptoms.get("port_scan") == "Yes" else 0
        bot_activity = 1 if symptoms.get("bot_activity") == "Yes" else 0
        web_attack = 1 if symptoms.get("web_attack") == "Yes" else 0

        
        # Weighted Score
        score = (
            slow * 2 +
            failed * 3 +
            conn * 3 +
            file_mod * 2 +
            port * 3
        )

        # Attack Logic
        if score >= 8:
            attack_type = "DDoS"
            confidence = 0.95

        elif failed == 1 and port == 0:
            attack_type = "BruteForce"
            confidence = 0.85

        elif port == 1 and slow == 0:
            attack_type = "PortScan"
            confidence = 0.80

        elif bot_activity == 1:
            attack_type = "Bot"
            confidence = 0.82


        elif web_attack == 1: 
            attack_type = "WebAttack"
            confidence = 0.80

        elif conn == 1 or file_mod == 1:
            attack_type = "Infiltration"
            confidence = 0.75

        else:
            attack_type = "Normal"
            confidence = 0.60
        

        # Risk Score
        risk_score = int(confidence * 100)


        # Severity Levels
        if risk_score >= 80:
            severity = "Critical"
        elif risk_score >= 60:
            severity = "Medium"
        else:
            severity = "Low"

        # USE YOUR CUSTOM SOLUTION TEXT
        action = solutions.get(attack_type, "No action available")


        # Save to DB
        with get_db_connection() as conn_db:
            conn_db.execute(
                "INSERT INTO predictions (timestamp, user, ip, attack_type, confidence, risk) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    datetime.datetime.now().isoformat(),
                    user,
                    ip,
                    attack_type,
                    confidence,
                    risk_score
                )
            )
            conn_db.commit()

 
        # Response
        return jsonify({
            "attack_type": attack_type,
            "confidence": round(confidence * 100, 2),
            "risk": risk_score,
            "severity": severity,
            "action": action   
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------------- Logs -------------------
@app.route("/user_logs")
def user_logs():
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT 500"
        ).fetchall()

        logs = []
        for r in rows:
            logs.append({
                "timestamp": r["timestamp"],
                "user": r["user"],
                "ip": r["ip"],
                "attack_type": r["attack_type"],
                "confidence": round(r["confidence"] * 100, 2),
                "risk": r["risk"]
            })
        return jsonify(logs)
    
#-------------Summary----------#
@app.route("/attack_summary")
def attack_summary():
    with get_db_connection() as conn:
        rows = conn.execute("""
            SELECT attack_type, COUNT(*) as count
            FROM predictions
            GROUP BY attack_type
        """).fetchall()

    summary = {row["attack_type"]: row["count"] for row in rows}
    # Ensure all attack types appear (fill 0 if missing)
    all_types = ["DDoS", "PortScan", "Bot", "BruteForce", "WebAttack", "Infiltration", "Normal"]
    for t in all_types:
        summary.setdefault(t, 0)

    return jsonify(summary)

@app.route("/dashboard_stats")
def dashboard_stats():
    with get_db_connection() as conn:
        # 1. Total Detections
        total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        
        # 2. High Risk (Assume Risk Score > 80 is High Risk)
        high_risk = conn.execute("SELECT COUNT(*) FROM predictions WHERE risk >= 80").fetchone()[0]
        
        # 3. Mitigated (In a real app, these are resolved threats. For now, we'll use Total - High Risk)
        mitigated = total - high_risk
        
        # 4. Network Load (Calculated or Static for UI)
        net_load = f"{np.random.randint(30, 65)}%" 

    return jsonify({
        "total": total,
        "high_risk": high_risk,
        "mitigated": mitigated,
        "net_load": net_load
    })

# ------------------- Run Flask -------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
    
