import numpy as np
import joblib
import os
import json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import xgboost as xgb
import torch
import torch.nn as nn
from dotenv import load_dotenv

load_dotenv()

# paths
LSTM_PATH= os.getenv("LSTM_PATH")
SCALER_PATH= os.getenv("SCALER_PATH")
XGB_PATH= os.getenv("XGB_PATH")
CENTRAL_DATA_PATH= os.getenv("CENTRAL_DATA_PATH")

# load lstm model class
class SuspiciousLSTM(nn.Module):
    def __init__(self, input_dim=3, hidden=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden,1)
        self.act = nn.Sigmoid()
    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.act(self.fc(last)).squeeze(1)

def compute_suspicious_with_lstm(lstm_model, history):
    if lstm_model is None:
        # fallback heuristic
        pps = history[-1,0]; syn = history[-1,2]
        s = min(1.0, (pps/2000)*0.6 + syn*0.6)
        return s
    arr = history.astype("float32")[None,...]
    with torch.no_grad():
        t = torch.from_numpy(arr)
        out = lstm_model(t).numpy()
        return float(out[0])

# generate labeled dataset (three classes)
def gen_snapshot(lstm_model=None):
    # generate a short sequence
    seq_len=10
    base_pps = np.random.randint(10, 300)
    base_syn = np.random.random()*0.2
    base_ips = np.random.randint(1, 40)
    seq = []
    attack=False
    for t in range(seq_len):
        if np.random.random() < 0.06 and t>3:
            attack=True
            pps = base_pps + np.random.randint(800,2500)
            syn = min(1.0, base_syn + np.random.random()*0.8 + 0.2)
            uniq = base_ips + np.random.randint(5,50)
        else:
            pps = max(1, int(base_pps + np.random.randint(-20,40)))
            syn = min(1.0, base_syn + np.random.random()*0.15)
            uniq = max(1, base_ips + np.random.randint(-2,6))
        seq.append([pps, uniq, syn])
    hist = np.array(seq)
    suspicious = compute_suspicious_with_lstm(lstm_model, hist)
    current = hist[-1]
    # label heuristic (we'll convert to classes)
    pps_val = current[0]
    if pps_val > 1500 or suspicious>0.8 or current[2]>0.7:
        label = 2
    elif pps_val > 800 or suspicious>0.5:
        label = 1
    else:
        label = 0
    # features: pps, uniq, syn, suspicious
    feat = np.concatenate([current, np.array([suspicious])])
    return feat, label

def load_real_data(lstm_model):
    """Load data from central dataset for continuous learning"""
    if not os.path.exists(CENTRAL_DATA_PATH):
        return None, None
    
    try:
        with open(CENTRAL_DATA_PATH, 'r') as f:
            data = json.load(f)
        
        if not data or len(data) == 0:
            return None, None
        
        print(f"✓ Loaded {len(data)} samples from central dataset")
        
        X = []
        y = []
        
        for sample in data:
            history = np.array(sample['history'], dtype=np.float32)
            current = np.array(sample['current'], dtype=np.float32)
            
            # Compute suspicious score using LSTM
            suspicious = compute_suspicious_with_lstm(lstm_model, history)
            
            # Features: [pps, unique_ips, syn_ratio, suspicious]
            features = np.concatenate([current, [suspicious]])
            X.append(features)
            y.append(sample.get('action', sample.get('actual_action', 0)))
        
        return np.array(X), np.array(y)
    
    except Exception as e:
        print(f"✗ Failed to load central dataset: {e}")
        return None, None

def main(n=8000):
    # load lstm if exists
    lstm = None
    if os.path.exists(LSTM_PATH):
        lstm = SuspiciousLSTM()
        lstm.load_state_dict(torch.load(LSTM_PATH, map_location="cpu"))
        lstm.eval()
        print(f"✓ Loaded LSTM for suspicious scoring from {LSTM_PATH}")
    else:
        print(f"✗ No LSTM found at {LSTM_PATH}; using heuristic for suspicious score")

    # Try to load real data first
    X_real, y_real = load_real_data(lstm)
    
    if X_real is not None and len(X_real) >= 100:
        # Use real data from continuous learning
        X = X_real
        y = y_real
        print(f"Mode: Continuous Learning with {len(X)} real samples")
        use_real_data = True
    else:
        # Generate synthetic data for initial training
        print(f"Mode: Initial Training - generating {n} synthetic samples")
        X = []
        y = []
        for i in range(n):
            feat, lab = gen_snapshot(lstm)
            X.append(feat)
            y.append(lab)
        X = np.array(X)
        y = np.array(y)
        use_real_data = False
    
    # Load existing scaler if it exists (for continuous learning)
    if os.path.exists(SCALER_PATH) and use_real_data:
        try:
            scaler = joblib.load(SCALER_PATH)
            print(f"✓ Loaded existing scaler from {SCALER_PATH}")
        except:
            scaler = StandardScaler()
            print("Creating new scaler")
    else:
        scaler = StandardScaler()
        print("Creating new scaler")
    
    # Scale features
    Xs = scaler.fit_transform(X)
    joblib.dump(scaler, SCALER_PATH)
    print(f"✓ Saved scaler: {SCALER_PATH}")
    
    # Load existing XGBoost model if it exists (for continuous learning)
    existing_model = None
    if os.path.exists(XGB_PATH) and use_real_data:
        try:
            existing_model = joblib.load(XGB_PATH)
            print(f"✓ Loaded existing XGBoost model from {XGB_PATH}")
        except:
            print("Creating new XGBoost model")
    
    # Train xgboost classifier
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.18, random_state=42, stratify=y)
    
    if existing_model and use_real_data:
        # Continue training from existing model
        print("Continuing training from existing model...")
        clf = xgb.XGBClassifier(
            n_estimators=50,  # Fewer trees for incremental training
            max_depth=5,
            use_label_encoder=False,
            eval_metric="mlogloss"
        )
        # Use xgb_model parameter to continue training
        clf.fit(Xtr, ytr, xgb_model=existing_model.get_booster())
    else:
        # Train fresh model
        print("Training fresh XGBoost model...")
        clf = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            use_label_encoder=False,
            eval_metric="mlogloss"
        )
        clf.fit(Xtr, ytr)
    
    preds = clf.predict(Xte)
    print("\nXGBoost classification report:")
    print(classification_report(yte, preds, target_names=['Allow', 'Rate Limit', 'Block']))
    
    joblib.dump(clf, XGB_PATH)
    print(f"✓ Saved XGBoost: {XGB_PATH}")

if __name__ == "__main__":
    main(8000)