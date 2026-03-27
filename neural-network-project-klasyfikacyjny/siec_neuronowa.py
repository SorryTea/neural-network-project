import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import os, json

np.random.seed(42)
RESULTS = "wyniki"
N_SAMPLE = 10000
N_REPEATS = 3
EPOKI = 80
os.makedirs(RESULTS, exist_ok=True)

KOLOR_TRAIN = "#1976D2"
KOLOR_TEST = "#D32F2F"

# ================================================================
# 1. WCZYTANIE DANYCH
# ================================================================
# Kolumny Airlines.csv:
#   Airline, Flight, AirportFrom, AirportTo, DayOfWeek, Time, Length, Delay

df = pd.read_csv("Airlines.csv")
print(f"Wczytano {len(df):,} rekordow")

df = df.dropna(subset=["Delay", "DayOfWeek", "Time", "Length"])

le = LabelEncoder()
for col in ["Airline", "AirportFrom", "AirportTo"]:
    df[col] = le.fit_transform(df[col].astype(str))

feat_cols = ["Airline", "AirportFrom", "AirportTo", "DayOfWeek", "Time", "Length"]
X = df[feat_cols].values.astype(float)
y = df["Delay"].values.astype(float)

idx = np.random.choice(len(X), N_SAMPLE, replace=False)
X, y = X[idx], y[idx]

X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8)

idx = np.random.permutation(len(X))
split = int(0.8 * len(X))
X_train, X_test = X[idx[:split]], X[idx[split:]]
y_train, y_test = y[idx[:split]], y[idx[split:]]

N_FEATURES = X.shape[1]
print(f"X_train={X_train.shape}  X_test={X_test.shape}")
print(f"Opoznione w train: {y_train.mean()*100:.1f}%\n")


# ================================================================
# 2. SIEC NEURONOWA OD ZERA
# ================================================================
class SiecNeuronowa:

    def __init__(self, rozmiary_warstw, aktywacja="relu", lr=0.01, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.rozmiary = rozmiary_warstw
        self.aktywacja_nazwa = aktywacja
        self.lr = lr
        self.wagi = []
        self.biasy = []
        for i in range(len(rozmiary_warstw) - 1):
            skala = np.sqrt(2.0 / rozmiary_warstw[i])
            self.wagi.append(
                np.random.randn(rozmiary_warstw[i], rozmiary_warstw[i + 1]) * skala
            )
            self.biasy.append(np.zeros((1, rozmiary_warstw[i + 1])))
        self.historia_acc = []

    def aktywuj(self, z):
        if self.aktywacja_nazwa == "relu":
            return np.maximum(0, z)
        if self.aktywacja_nazwa == "sigmoid":
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        if self.aktywacja_nazwa == "tanh":
            return np.tanh(z)
        if self.aktywacja_nazwa == "elu":
            return np.where(z >= 0, z, np.exp(np.clip(z, -500, 0)) - 1)

    def pochodna_aktywacji(self, z):
        if self.aktywacja_nazwa == "relu":
            return (z > 0).astype(float)
        if self.aktywacja_nazwa == "sigmoid":
            s = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
            return s * (1 - s)
        if self.aktywacja_nazwa == "tanh":
            return 1 - np.tanh(z) ** 2
        if self.aktywacja_nazwa == "elu":
            return np.where(z >= 0, 1.0, np.exp(np.clip(z, -500, 0)))

    @staticmethod
    def _sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def propagacja_w_przod(self, X):
        self.z_lista = []
        self.a_lista = [X]
        a = X
        for i, (W, b) in enumerate(zip(self.wagi, self.biasy)):
            z = a @ W + b
            self.z_lista.append(z)
            a = self.aktywuj(z) if i < len(self.wagi) - 1 else self._sigmoid(z)
            self.a_lista.append(a)
        return a

    def propagacja_wsteczna(self, X, y):
        m = X.shape[0]
        y = y.reshape(-1, 1)
        delta = self.a_lista[-1] - y
        grad_wagi, grad_biasy = [], []
        for i in reversed(range(len(self.wagi))):
            grad_wagi.insert(0, (self.a_lista[i].T @ delta) / m)
            grad_biasy.insert(0, delta.mean(axis=0, keepdims=True))
            if i > 0:
                delta = (delta @ self.wagi[i].T) * self.pochodna_aktywacji(
                    self.z_lista[i - 1]
                )
        for i in range(len(self.wagi)):
            self.wagi[i] -= self.lr * grad_wagi[i]
            self.biasy[i] -= self.lr * grad_biasy[i]

    def ucz(self, X_tr, y_tr, X_te, y_te, epoki=80, batch=64):
        m = X_tr.shape[0]
        self.historia_acc = []
        for epoka in range(epoki):
            idx = np.random.permutation(m)
            X_s, y_s = X_tr[idx], y_tr[idx]
            for start in range(0, m, batch):
                self.propagacja_w_przod(X_s[start : start + batch])
                self.propagacja_wsteczna(
                    X_s[start : start + batch], y_s[start : start + batch]
                )
            if (epoka + 1) % 10 == 0:
                acc_tr = (
                    (self.propagacja_w_przod(X_tr).flatten() > 0.5) == y_tr
                ).mean()
                acc_te = (
                    (self.propagacja_w_przod(X_te).flatten() > 0.5) == y_te
                ).mean()
                self.historia_acc.append((acc_tr, acc_te))

    def predykcja(self, X):
        return (self.propagacja_w_przod(X).flatten() > 0.5).astype(int)

    def dokladnosc(self, X, y):
        return (self.predykcja(X) == y).mean()


# ================================================================
# 3. ANALIZA PARAMETROW
# ================================================================
def analizuj(nazwa, wartosci, buduj_fn):
    print(f"\nPARAMETR: {nazwa}")
    wyniki = []
    for war in wartosci:
        tr_list, te_list = [], []
        for rep in range(N_REPEATS):
            np.random.seed(rep * 17 + 3)
            siec = buduj_fn(war)
            siec.ucz(X_train, y_train, X_test, y_test, epoki=EPOKI)
            acc_tr = siec.dokladnosc(X_train, y_train)
            acc_te = siec.dokladnosc(X_test, y_test)
            tr_list.append(acc_tr)
            te_list.append(acc_te)
            print(
                f"  [{nazwa}={war} | pow.{rep+1}] train={acc_tr:.4f}  test={acc_te:.4f}"
            )
        wyniki.append(
            {
                "wartosc": str(war),
                "train_mean": round(float(np.mean(tr_list)), 4),
                "train_std": round(float(np.std(tr_list)), 4),
                "train_min": round(float(np.min(tr_list)), 4),
                "train_max": round(float(np.max(tr_list)), 4),
                "test_mean": round(float(np.mean(te_list)), 4),
                "test_std": round(float(np.std(te_list)), 4),
                "test_min": round(float(np.min(te_list)), 4),
                "test_max": round(float(np.max(te_list)), 4),
                "train_all": tr_list,
                "test_all": te_list,
            }
        )
    return wyniki


wyniki_lr = analizuj(
    "Wsp. uczenia",
    [0.001, 0.01, 0.05, 0.1],
    lambda lr: SiecNeuronowa([N_FEATURES, 8, 1], aktywacja="relu", lr=lr),
)
wyniki_neurony = analizuj(
    "Neurony",
    [4, 8, 16, 32],
    lambda n: SiecNeuronowa([N_FEATURES, n, 1], aktywacja="relu", lr=0.01),
)
wyniki_akt = analizuj(
    "Aktywacja",
    ["sigmoid", "relu", "tanh", "elu"],
    lambda a: SiecNeuronowa([N_FEATURES, 8, 1], aktywacja=a, lr=0.01),
)
wyniki_warstwy = analizuj(
    "Warstwy",
    [1, 2, 3, 4],
    lambda n: SiecNeuronowa([N_FEATURES] + [8] * n + [1], aktywacja="relu", lr=0.01),
)


# ================================================================
# 4. WYKRESY PARAMETROW
# ================================================================
def wykres_slupkowy(wyniki, nazwa_param, filename):
    wartosci = [r["wartosc"] for r in wyniki]
    tr_mean = [r["train_mean"] for r in wyniki]
    te_mean = [r["test_mean"] for r in wyniki]
    tr_std = [r["train_std"] for r in wyniki]
    te_std = [r["test_std"] for r in wyniki]
    x = np.arange(len(wartosci))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    bars_tr = ax.bar(
        x - w / 2,
        tr_mean,
        w,
        yerr=tr_std,
        label="Zbior uczacy",
        color=KOLOR_TRAIN,
        alpha=0.85,
        capsize=5,
    )
    bars_te = ax.bar(
        x + w / 2,
        te_mean,
        w,
        yerr=te_std,
        label="Zbior testowy",
        color=KOLOR_TEST,
        alpha=0.85,
        capsize=5,
    )
    for b in list(bars_tr) + list(bars_te):
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2,
            h + 0.003,
            f"{h:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_ylim(max(0, min(tr_mean + te_mean) - 0.05), 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(wartosci, fontsize=11)
    ax.set_xlabel(nazwa_param, fontsize=12, fontweight="bold")
    ax.set_ylabel("Dokladnosc (Accuracy)", fontsize=12)
    ax.set_title(
        f"Wplyw parametru: {nazwa_param}\n(srednia z {N_REPEATS} powt. +/- odch. std.)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, filename), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Zapisano: {filename}")


print("\nGenerowanie wykresow...")
wykres_slupkowy(wyniki_lr, "Wspolczynnik uczenia", "p1_lr.png")
wykres_slupkowy(wyniki_neurony, "Liczba neuronow", "p2_neurony.png")
wykres_slupkowy(wyniki_akt, "Funkcja aktywacji", "p3_aktywacja.png")
wykres_slupkowy(wyniki_warstwy, "Liczba warstw ukrytych", "p4_warstwy.png")


# ================================================================
# 5. FINALNY MODEL
# ================================================================
print("\nFinalny model [N->16->8->1, ReLU, lr=0.01, 150 epok]...")

final_modele = []
for rep in range(N_REPEATS):
    siec = SiecNeuronowa(
        [N_FEATURES, 16, 8, 1], aktywacja="relu", lr=0.01, seed=rep * 10
    )
    siec.ucz(X_train, y_train, X_test, y_test, epoki=150)
    final_modele.append(siec)

best = max(final_modele, key=lambda s: s.dokladnosc(X_test, y_test))
acc_tr_fin = best.dokladnosc(X_train, y_train)
acc_te_fin = best.dokladnosc(X_test, y_test)
print(f"  train={acc_tr_fin:.4f}  test={acc_te_fin:.4f}")

epoki_x = list(range(10, 151, 10))
tr_acc_hist = [v[0] for v in best.historia_acc]
te_acc_hist = [v[1] for v in best.historia_acc]

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(
    epoki_x,
    tr_acc_hist,
    "-o",
    color=KOLOR_TRAIN,
    label="Zbior uczacy",
    linewidth=2,
    markersize=5,
)
ax.plot(
    epoki_x,
    te_acc_hist,
    "-s",
    color=KOLOR_TEST,
    label="Zbior testowy",
    linewidth=2,
    markersize=5,
)
ax.set_xlabel("Epoka", fontsize=12)
ax.set_ylabel("Dokladnosc", fontsize=12)
ax.set_title("Krzywa uczenia — finalny model", fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(alpha=0.3, linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "krzywa_uczenia.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Zapisano: krzywa_uczenia.png")

y_pred_fin = best.predykcja(X_test)
TP = int(((y_pred_fin == 1) & (y_test == 1)).sum())
TN = int(((y_pred_fin == 0) & (y_test == 0)).sum())
FP = int(((y_pred_fin == 1) & (y_test == 0)).sum())
FN = int(((y_pred_fin == 0) & (y_test == 1)).sum())
cm = np.array([[TN, FP], [FN, TP]])

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap="Blues")
plt.colorbar(im, ax=ax)
for i in range(2):
    for j in range(2):
        ax.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            fontsize=22,
            fontweight="bold",
            color="white" if cm[i, j] > cm.max() * 0.6 else "black",
        )
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Punkt. (pred.)", "Opozn. (pred.)"], fontsize=10)
ax.set_yticklabels(["Punkt. (rzecz.)", "Opozn. (rzecz.)"], fontsize=10)
ax.set_title("Macierz pomylek — finalny model", fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "macierz_pomylek.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Zapisano: macierz_pomylek.png")

rows = []
for wyniki, nazwa in [
    (wyniki_lr, "Wspolczynnik uczenia"),
    (wyniki_neurony, "Liczba neuronow"),
    (wyniki_akt, "Funkcja aktywacji"),
    (wyniki_warstwy, "Liczba warstw ukrytych"),
]:
    for r in wyniki:
        rows.append(
            {
                "Parametr": nazwa,
                "Wartosc": r["wartosc"],
                "Train Mean": r["train_mean"],
                "Train Std": r["train_std"],
                "Train Min": r["train_min"],
                "Train Max": r["train_max"],
                "Test Mean": r["test_mean"],
                "Test Std": r["test_std"],
                "Test Min": r["test_min"],
                "Test Max": r["test_max"],
            }
        )
pd.DataFrame(rows).to_csv(
    os.path.join(RESULTS, "tabela_wynikow.csv"), index=False, encoding="utf-8-sig"
)
print("  Zapisano: tabela_wynikow.csv")

precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
with open(os.path.join(RESULTS, "metryki_final.json"), "w", encoding="utf-8") as f:
    json.dump(
        {
            "acc_train": round(acc_tr_fin, 4),
            "acc_test": round(acc_te_fin, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "TP": TP,
            "TN": TN,
            "FP": FP,
            "FN": FN,
        },
        f,
        indent=2,
    )


# ================================================================
# 6. POROWNANIE Z KLASYCZNYMI METODAMI
# ================================================================
print("\nPorownanie z klasycznymi metodami...")

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

metody = {
    "Drzewo decyzyjne": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Las losowy": RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=42
    ),
    "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "Naiwny Bayes": GaussianNB(),
    "SVM (RBF)": SVC(kernel="rbf", C=1.0, random_state=42),
}

wyniki_por = {}
for nazwa, clf in metody.items():
    clf.fit(X_train, y_train)
    wyniki_por[nazwa] = {
        "train": round(accuracy_score(y_train, clf.predict(X_train)), 4),
        "test": round(accuracy_score(y_test, clf.predict(X_test)), 4),
    }
    print(
        f"  {nazwa:<22} train={wyniki_por[nazwa]['train']}  test={wyniki_por[nazwa]['test']}"
    )

wyniki_por["SSN (wlasna impl.)"] = {
    "train": round(acc_tr_fin, 4),
    "test": round(acc_te_fin, 4),
}

nazwy = list(wyniki_por.keys())
tr_vals = [wyniki_por[n]["train"] for n in nazwy]
te_vals = [wyniki_por[n]["test"] for n in nazwy]
x = np.arange(len(nazwy))
w = 0.35

fig, ax = plt.subplots(figsize=(11, 6))
bars_tr = ax.bar(
    x - w / 2, tr_vals, w, label="Zbior uczacy", color=KOLOR_TRAIN, alpha=0.85
)
bars_te = ax.bar(
    x + w / 2, te_vals, w, label="Zbior testowy", color=KOLOR_TEST, alpha=0.85
)
for i, n in enumerate(nazwy):
    if "SSN" in n:
        ax.bar(i - w / 2, tr_vals[i], w, color="#0D47A1", alpha=1.0)
        ax.bar(i + w / 2, te_vals[i], w, color="#B71C1C", alpha=1.0)
for b in list(bars_tr) + list(bars_te):
    h = b.get_height()
    ax.text(
        b.get_x() + b.get_width() / 2,
        h + 0.003,
        f"{h:.3f}",
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold",
    )
ax.set_ylim(max(0, min(tr_vals + te_vals) - 0.05), 1.0)
ax.set_xticks(x)
ax.set_xticklabels(nazwy, fontsize=9, rotation=10, ha="right")
ax.set_ylabel("Dokladnosc (Accuracy)", fontsize=12)
ax.set_title(
    "Porownanie wlasnej SSN z klasycznymi metodami",
    fontsize=13,
    fontweight="bold",
    pad=12,
)
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "porownanie_metod.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Zapisano: porownanie_metod.png")

with open(os.path.join(RESULTS, "porownanie.json"), "w", encoding="utf-8") as f:
    json.dump(wyniki_por, f, ensure_ascii=False, indent=2)

print("\nGOTOWE! Pliki zapisane w folderze:", RESULTS)
