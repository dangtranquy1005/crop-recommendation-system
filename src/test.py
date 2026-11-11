import pandas as pd
from joblib import load

from config import BASE_DIR, FEATURES

# Tai mo hinh da huan luyen
model = load(BASE_DIR / "models" / "lda_model.joblib")

# Mot mau du lieu moi
sample = pd.DataFrame([{
    "N": 90,
    "P": 42,
    "K": 43,
    "temperature": 25,
    "humidity": 80,
    "ph": 6.5,
    "rainfall": 200
}])[FEATURES]

# Du doan loai cay
prediction = model.predict(sample)

print("Cay trong du doan:", prediction[0])
