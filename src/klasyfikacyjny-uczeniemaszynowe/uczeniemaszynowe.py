"""
Predykcja opoznien lotow - Uczenie Maszynowe (Czesc 2 projektu)
Dataset: Airlines Dataset to Predict a Delay
Zrodlo: https://www.kaggle.com/datasets/jimschacko/airlines-dataset-to-predict-a-delay

Problem klasyfikacyjny: przewidywanie czy lot bedzie opozniony (0/1).

Testowane metody (4):
  1. Las Losowy (Random Forest) - zespol drzew decyzyjnych z glosowaniem wiekszosciowym;
     zmniejsza wariancje dzieki bootstrap + losowemu doborowi cech.
  2. Drzewo Decyzyjne - hierarchiczny model dzielacy przestrzen cech;
     kazdy podzial wybierany wg kryterium (gini/entropy).
  3. k-Najblizszych Sasiadow (k-NN) - klasyfikacja na podstawie glosowania
     k najblizszych probek w przestrzeni cech (wedlug wybranej metryki).
  4. Regresja Logistyczna - model liniowy z funkcja sigmoid, zwracajacy
     prawdopodobienstwo przynaleznosci do klasy; regularyzacja L2.

Testowane parametry (8, kazdy z 4 wartosciami - wymagane min. 3 parametry):
  RF:     n_estimators, max_depth
  DT:     max_depth, min_samples_leaf
  k-NN:   n_neighbors, metric
  LogReg: C, solver
"""

import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

np.random.seed(42)
RESULTS_ML = "wyniki_ml_klasyfikacja"
N_SAMPLE = 10000
os.makedirs(RESULTS_ML, exist_ok=True)
C_TRAIN, C_TEST = "#1976D2", "#D32F2F"

# ================================================================
# 1. DANE
# ================================================================
print("Wczytywanie i przygotowywanie danych...")
df = pd.read_csv("klasyfikacyjny-uczeniemaszynowe/Airlines.csv").dropna(subset=["Delay", "DayOfWeek", "Time", "Length"])
le = LabelEncoder()
for col in ["Airline", "AirportFrom", "AirportTo"]:
    df[col] = le.fit_transform(df[col].astype(str))

X_all = df[["Airline", "AirportFrom", "AirportTo", "DayOfWeek", "Time", "Length"]].values.astype(float)
y_all = df["Delay"].values.astype(int)

idx = np.random.choice(len(X_all), N_SAMPLE, replace=False)
X_all, y_all = X_all[idx], y_all[idx]

X_all = (X_all - X_all.min(0)) / (X_all.max(0) - X_all.min(0) + 1e-8)


def podziel(X, y, test_size=0.2):
    idx = np.random.permutation(len(X))
    s = int((1 - test_size) * len(X))
    return X[idx[:s]], X[idx[s:]], y[idx[:s]], y[idx[s:]]


X_train, X_test, y_train, y_test = podziel(X_all, y_all)
print(f"X_train={X_train.shape}  X_test={X_test.shape}\n")

# ================================================================
# 2. FUNKCJA ANALIZUJACA WPLYW PARAMETRU
# ================================================================
def analizuj(nazwa_modelu, nazwa_param, wartosci, buduj_model_fn):
    print(f"[{nazwa_modelu}]  Parametr: {nazwa_param}")
    wyniki = []
    for war in wartosci:
        model = buduj_model_fn(war)
        model.fit(X_train, y_train)
        yp_tr = model.predict(X_train)
        yp_te = model.predict(X_test)

        acc_tr = accuracy_score(y_train, yp_tr)
        acc_te = accuracy_score(y_test, yp_te)
        prec = precision_score(y_test, yp_te, zero_division=0)
        rec = recall_score(y_test, yp_te, zero_division=0)
        f1 = f1_score(y_test, yp_te, zero_division=0)

        print(f"   {nazwa_param}={war:<10}  tr={acc_tr:.4f}  te={acc_te:.4f}  "
              f"P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}")
        wyniki.append({
            "wartosc": str(war),
            "train_acc": acc_tr, "test_acc": acc_te,
            "precision": prec, "recall": rec, "f1": f1,
        })
    print()
    return wyniki


# ================================================================
# 3. TESTOWANIE 4 METOD x 2 PARAMETRY (kazdy min. 4 wartosci)
# ================================================================
# --- LAS LOSOWY ---
rf_n = analizuj(
    "Las Losowy", "n_estimators", [10, 50, 100, 200],
    lambda v: RandomForestClassifier(n_estimators=v, random_state=42, n_jobs=-1),
)
rf_d = analizuj(
    "Las Losowy", "max_depth", [3, 5, 10, 20],
    lambda v: RandomForestClassifier(n_estimators=100, max_depth=v, random_state=42, n_jobs=-1),
)

# --- DRZEWO DECYZYJNE ---
dt_d = analizuj(
    "Drzewo Decyzyjne", "max_depth", [3, 5, 10, 20],
    lambda v: DecisionTreeClassifier(max_depth=v, random_state=42),
)
dt_l = analizuj(
    "Drzewo Decyzyjne", "min_samples_leaf", [1, 5, 10, 20],
    lambda v: DecisionTreeClassifier(min_samples_leaf=v, random_state=42),
)

# --- k-NN ---
knn_k = analizuj(
    "k-NN", "n_neighbors", [3, 5, 9, 15],
    lambda v: KNeighborsClassifier(n_neighbors=v, n_jobs=-1),
)
knn_m = analizuj(
    "k-NN", "metric", ["euclidean", "manhattan", "chebyshev", "minkowski"],
    lambda v: KNeighborsClassifier(n_neighbors=5, metric=v, n_jobs=-1),
)

# --- REGRESJA LOGISTYCZNA ---
lr_c = analizuj(
    "Regresja Logistyczna", "C", [0.01, 0.1, 1.0, 10.0],
    lambda v: LogisticRegression(C=v, max_iter=2000, random_state=42),
)
lr_s = analizuj(
    "Regresja Logistyczna", "solver", ["lbfgs", "liblinear", "newton-cg", "saga"],
    lambda v: LogisticRegression(solver=v, max_iter=2000, random_state=42),
)

# ================================================================
# 4. WYKRESY
# ================================================================
def wykres(wyniki, nazwa_modelu, nazwa_param, plik):
    vals = [r["wartosc"] for r in wyniki]
    tr_m = [r["train_acc"] for r in wyniki]
    te_m = [r["test_acc"] for r in wyniki]

    x, w = np.arange(len(vals)), 0.35
    fig, ax = plt.subplots(figsize=(8, 5))

    btr = ax.bar(x - w / 2, tr_m, w, label="Zbior uczacy", color=C_TRAIN, alpha=0.85)
    bte = ax.bar(x + w / 2, te_m, w, label="Zbior testowy", color=C_TEST, alpha=0.85)

    for b in list(btr) + list(bte):
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 0.003, f"{h:.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylim(max(0, min(tr_m + te_m) - 0.05), 1.02)
    ax.set_xticks(x)
    ax.set_xticklabels(vals, fontsize=10)
    ax.set_xlabel(nazwa_param, fontsize=11, fontweight="bold")
    ax.set_ylabel("Dokladnosc (Accuracy)", fontsize=11)
    ax.set_title(f"{nazwa_modelu} - wplyw parametru: {nazwa_param}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_ML, plik), dpi=150, bbox_inches="tight")
    plt.close()


print("Generowanie wykresow...")
wykres(rf_n,  "Las Losowy",           "n_estimators",     "rf_n_estimators.png")
wykres(rf_d,  "Las Losowy",           "max_depth",        "rf_max_depth.png")
wykres(dt_d,  "Drzewo Decyzyjne",     "max_depth",        "dt_max_depth.png")
wykres(dt_l,  "Drzewo Decyzyjne",     "min_samples_leaf", "dt_min_samples_leaf.png")
wykres(knn_k, "k-NN",                 "n_neighbors",      "knn_n_neighbors.png")
wykres(knn_m, "k-NN",                 "metric",           "knn_metric.png")
wykres(lr_c,  "Regresja Logistyczna", "C",                "lr_C.png")
wykres(lr_s,  "Regresja Logistyczna", "solver",           "lr_solver.png")

# ================================================================
# 5. PODSUMOWANIE TABELARYCZNE (do sprawozdania)
# ================================================================
def wiersze(nazwa, param, wyniki):
    return [
        {"model": nazwa, "parametr": param, "wartosc": r["wartosc"],
         "train_acc": round(r["train_acc"], 4), "test_acc": round(r["test_acc"], 4),
         "precision": round(r["precision"], 4), "recall": round(r["recall"], 4),
         "f1": round(r["f1"], 4)}
        for r in wyniki
    ]


rows = []
rows += wiersze("Las Losowy", "n_estimators", rf_n)
rows += wiersze("Las Losowy", "max_depth", rf_d)
rows += wiersze("Drzewo Decyzyjne", "max_depth", dt_d)
rows += wiersze("Drzewo Decyzyjne", "min_samples_leaf", dt_l)
rows += wiersze("k-NN", "n_neighbors", knn_k)
rows += wiersze("k-NN", "metric", knn_m)
rows += wiersze("Regresja Logistyczna", "C", lr_c)
rows += wiersze("Regresja Logistyczna", "solver", lr_s)

podsumowanie = pd.DataFrame(rows)
csv_path = os.path.join(RESULTS_ML, "podsumowanie.csv")
podsumowanie.to_csv(csv_path, index=False, encoding="utf-8")

print("\n=== PODSUMOWANIE ===")
print(podsumowanie.to_string(index=False))
print(f"\nZapisano tabele: {csv_path}")
print(f"Wyniki zapisano w folderze: {RESULTS_ML}")
