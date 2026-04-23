import pandas as pd
import json
from sklearn.model_selection import train_test_split

df = pd.read_csv("unsw_nb15.csv")
print(f"Loaded CSV with {len(df)} total rows.")

# Sample 10,000 random rows for a rigorous training run
df_sample = df.sample(n=min(10000, len(df)), random_state=42)

samples = []
for _, row in df_sample.iterrows():
    input_text = (
        f"Network traffic using {row['proto']} protocol, "
        f"service {row['service']}, state {row['state']}, "
        f"source bytes {row['sbytes']}, destination bytes {row['dbytes']}, "
        f"rate {row['rate']}."
    )
    
    attack_type = "Normal" if row['label'] == 0 else row['attack_cat']
    output = {
        "attack_type": attack_type,
        "severity": "HIGH" if row['label'] == 1 else "LOW"
    }
    
    samples.append({
        "input": input_text,
        "output": json.dumps(output)
    })

# Split 80% Train, 20% Test
train_samples, test_samples = train_test_split(samples, test_size=0.2, random_state=42)

with open("train.jsonl", "w") as f:
    for s in train_samples:
        f.write(json.dumps(s) + "\n")

with open("test.jsonl", "w") as f:
    for s in test_samples:
        f.write(json.dumps(s) + "\n")

print(f"✅ Created train.jsonl with {len(train_samples)} rows")
print(f"✅ Created test.jsonl with {len(test_samples)} rows")