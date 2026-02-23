import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path

# =========================
# 설정
# =========================
DAY = "2026-01-06"
INPUT_PATH = f"out/common/sessions_packed_{DAY}.parquet"
OUTPUT_PATH = f"out/trackB/sessions_scored_{DAY}.parquet"

ALPHA_B1 = 0.5
ALPHA_B2 = 1.0
EPS = 1e-9

# =========================
# Load
# =========================
df = pd.read_parquet(INPUT_PATH)

# route_group null 방지
df["route_group"] = df["route_group"].apply(
    lambda x: x if isinstance(x, list) else []
)

# =========================
# Partition loop
# =========================
results = []

for (client, day), part in df.groupby(["client_name", "day"]):

    # ---------------------
    # vocab + transitions
    # ---------------------
    vocab = set()
    transitions = Counter()
    total_transitions = 0

    for seq in part["route_group"]:
        if len(seq) >= 2:
            vocab.update(seq)
            for a, b in zip(seq[:-1], seq[1:]):
                transitions[(a, b)] += 1
                total_transitions += 1

    vocab = list(vocab)
    V = len(vocab) if len(vocab) > 0 else 1

    # transition denominator per state
    outgoing = Counter()
    for (a, b), c in transitions.items():
        outgoing[a] += c

    # ---------------------
    # 세션별 점수 계산
    # ---------------------
    for idx, row in part.iterrows():

        seq = row["route_group"]
        L = len(seq)

        # 기본값
        b1_mnll = 0.0
        b2_score = 0.0
        entropy = 0.0

        # ----- B1 -----
        if L >= 2 and total_transitions > 0:

            log_probs = []

            for a, b in zip(seq[:-1], seq[1:]):
                c_ab = transitions.get((a, b), 0)
                c_a = outgoing.get(a, 0)

                prob = (c_ab + ALPHA_B1) / (
                    c_a + ALPHA_B1 * V
                )

                log_probs.append(np.log(prob + EPS))

            b1_mnll = -np.mean(log_probs)

        # ----- B2 -----
        if L >= 2 and total_transitions > 0:

            log_probs = []

            for a, b in zip(seq[:-1], seq[1:]):
                c_ab = transitions.get((a, b), 0)

                prob = (c_ab + ALPHA_B2) / (
                    total_transitions + ALPHA_B2 * V * V
                )

                log_probs.append(np.log(prob + EPS))

            b2_score = -np.mean(log_probs)

        # ----- B3 entropy -----
        if L > 0:
            counts = Counter(seq)
            probs = np.array(list(counts.values())) / L
            entropy = -np.sum(probs * np.log(probs + EPS))

        results.append({
            "client_name": client,
            "day": day,
            "user_id": row["user_id"],
            "session_key": row["session_key"],
            "b1_mnll": b1_mnll,
            "b2_score": b2_score,
            "b3_entropy": entropy
        })

# =========================
# DataFrame 변환
# =========================
score_df = pd.DataFrame(results)

# =========================
# Robust Z 함수
# =========================
def robust_z(series):
    median = series.median()
    mad = np.median(np.abs(series - median))
    return (series - median) / (1.4826 * (mad + EPS))

# partition 단위 z 계산
z_cols = []

for (client, day), part in score_df.groupby(["client_name", "day"]):
    idx = part.index

    score_df.loc[idx, "b1_z"] = robust_z(part["b1_mnll"])
    score_df.loc[idx, "b2_z"] = robust_z(part["b2_score"])
    score_df.loc[idx, "b3_z"] = robust_z(part["b3_entropy"]).abs()

# composite
score_df["B_risk"] = (
    score_df["b1_z"].clip(lower=0) +
    score_df["b2_z"].clip(lower=0) +
    score_df["b3_z"]
)

# =========================
# 저장
# =========================
Path("out/trackB").mkdir(parents=True, exist_ok=True)
score_df.to_parquet(OUTPUT_PATH, index=False)

# =========================
# Top 50 출력
# =========================
top50 = score_df.sort_values("B_risk", ascending=False).head(50)

print("\n=== Track B Top 50 ===")
print(top50[[
    "client_name",
    "user_id",
    "session_key",
    "b1_z",
    "b2_z",
    "b3_z",
    "B_risk"
]])