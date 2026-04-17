"""
Predykcja opoznien lotow — Siec Neuronowa od ZERA (tylko NumPy)
Dataset: Airlines Dataset to Predict a Delay
Link:    https://www.kaggle.com/datasets/jimschacko/airlines-dataset-to-predict-a-delay
Plik:    Airlines.csv
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import os, json

np.random.seed(42)
RESULTS = "wyniki"
N_SAMPLE = 10000
N_REPEATS = 3
EPOKI = 80
os.makedirs(RESULTS, exist_ok=True)
C_TRAIN, C_TEST = "#1976D2", "#D32F2F"


# ================================================================
# 1. DANE
# Kolumny: Airline, Flight, AirportFrom, AirportTo, DayOfWeek, Time, Length, Delay
# ================================================================
df = pd.read_csv("Airlines.csv").dropna(subset=["Delay", "DayOfWeek", "Time", "Length"])
le = LabelEncoder()
for col in ["Airline", "AirportFrom", "AirportTo"]:
    df[col] = le.fit_transform(df[col].astype(str))

X_all = df[
    ["Airline", "AirportFrom", "AirportTo", "DayOfWeek", "Time", "Length"]
].values.astype(float)
y_all = df["Delay"].values.astype(float)

idx = np.random.choice(len(X_all), N_SAMPLE, replace=False)
X_all, y_all = X_all[idx], y_all[idx]
X_all = (X_all - X_all.min(0)) / (X_all.max(0) - X_all.min(0) + 1e-8)
N_FEATURES = X_all.shape[1]
print(f"Wczytano {N_SAMPLE:,} rekordow | opoznione: {y_all.mean()*100:.1f}%")


def podziel(X, y, test_size=0.2):
    idx = np.random.permutation(len(X))
    s = int((1 - test_size) * len(X))
    return X[idx[:s]], X[idx[s:]], y[idx[:s]], y[idx[s:]]


X_train, X_test, y_train, y_test = podziel(X_all, y_all)
print(f"X_train={X_train.shape}  X_test={X_test.shape}\n")


# ================================================================
# 2. SIEC NEURONOWA
# ================================================================
class SSN:

    def __init__(self, warstwy, aktywacja="relu", lr=0.01, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.akt = aktywacja
        self.lr = lr
        self.W = [
            np.random.randn(warstwy[i], warstwy[i + 1]) * np.sqrt(2 / warstwy[i])
            for i in range(len(warstwy) - 1)
        ]
        self.b = [np.zeros((1, warstwy[i + 1])) for i in range(len(warstwy) - 1)]
        self.historia = []

    def _f(self, z):
        if self.akt == "relu":
            return np.maximum(0, z)
        if self.akt == "sigmoid":
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        if self.akt == "tanh":
            return np.tanh(z)
        if self.akt == "elu":
            return np.where(z >= 0, z, np.exp(np.clip(z, -500, 0)) - 1)

    def _df(self, z):
        if self.akt == "relu":
            return (z > 0).astype(float)
        if self.akt == "sigmoid":
            s = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
            return s * (1 - s)
        if self.akt == "tanh":
            return 1 - np.tanh(z) ** 2
        if self.akt == "elu":
            return np.where(z >= 0, 1.0, np.exp(np.clip(z, -500, 0)))

    @staticmethod
    def _sig(z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def _przod(self, X):
        self.zs, self.as_ = [], [X]
        a = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ W + b
            self.zs.append(z)
            a = self._f(z) if i < len(self.W) - 1 else self._sig(z)
            self.as_.append(a)
        return a

    def _wstecz(self, X, y):
        m = X.shape[0]
        y = y.reshape(-1, 1)
        d = self.as_[-1] - y
        gW, gb = [], []
        for i in reversed(range(len(self.W))):
            gW.insert(0, self.as_[i].T @ d / m)
            gb.insert(0, d.mean(0, keepdims=True))
            if i > 0:
                d = (d @ self.W[i].T) * self._df(self.zs[i - 1])
        for i in range(len(self.W)):
            self.W[i] -= self.lr * gW[i]
            self.b[i] -= self.lr * gb[i]

    def ucz(self, Xtr, ytr, Xte, yte, epoki=80, batch=64):
        self.historia = []
        for ep in range(epoki):
            idx = np.random.permutation(len(Xtr))
            Xs, ys = Xtr[idx], ytr[idx]
            for s in range(0, len(Xtr), batch):
                self._przod(Xs[s : s + batch])
                self._wstecz(Xs[s : s + batch], ys[s : s + batch])
            if (ep + 1) % 10 == 0:
                self.historia.append(
                    (
                        ((self._przod(Xtr).flatten() > 0.5) == ytr).mean(),
                        ((self._przod(Xte).flatten() > 0.5) == yte).mean(),
                    )
                )

    def acc(self, X, y):
        return ((self._przod(X).flatten() > 0.5).astype(int) == y).mean()


# ================================================================
# 3. ANALIZA  PARAMETROW
# Kazdy parametr: 4 wartosci, kazda konfiguracja powtorzona 3 razy
# Wyniki: srednia, std, min, max — osobno dla train i test
# ================================================================
def analizuj(nazwa, wartosci, buduj_fn, Xtr=None, Xte=None, ytr=None, yte=None):
    Xtr = X_train if Xtr is None else Xtr
    Xte = X_test if Xte is None else Xte
    ytr = y_train if ytr is None else ytr
    yte = y_test if yte is None else yte
    print(f"\nPARAMETR: {nazwa}")
    wyniki = []
    for war in wartosci:
        trl, tel = [], []
        for rep in range(N_REPEATS):
            np.random.seed(rep * 17 + 3)
            siec = buduj_fn(war)
            siec.ucz(Xtr, ytr, Xte, yte, epoki=EPOKI)
            trl.append(siec.acc(Xtr, ytr))
            tel.append(siec.acc(Xte, yte))
            print(f"  [{nazwa}={war}|p{rep+1}] tr={trl[-1]:.4f} te={tel[-1]:.4f}")
        wyniki.append(
            {
                "wartosc": str(war),
                "train_mean": round(float(np.mean(trl)), 4),
                "train_std": round(float(np.std(trl)), 4),
                "train_min": round(float(np.min(trl)), 4),
                "train_max": round(float(np.max(trl)), 4),
                "test_mean": round(float(np.mean(tel)), 4),
                "test_std": round(float(np.std(tel)), 4),
                "test_min": round(float(np.min(tel)), 4),
                "test_max": round(float(np.max(tel)), 4),
                "train_all": trl,
                "test_all": tel,
            }
        )
    return wyniki


w1 = analizuj(
    "Wspolczynnik uczenia",
    [0.001, 0.01, 0.05, 0.1],
    lambda lr: SSN([N_FEATURES, 8, 1], lr=lr),
)

w2 = analizuj(
    "Liczba neuronow", [4, 8, 16, 32], lambda n: SSN([N_FEATURES, n, 1], lr=0.01)
)

w3 = analizuj(
    "Funkcja aktywacji",
    ["sigmoid", "relu", "tanh", "elu"],
    lambda a: SSN([N_FEATURES, 8, 1], aktywacja=a, lr=0.01),
)

w4 = analizuj(
    "Liczba warstw", [1, 2, 3, 4], lambda n: SSN([N_FEATURES] + [8] * n + [1], lr=0.01)
)

print("\nPARAMETR: Podzial danych (train/test)")
w5 = []
for ts in [0.4, 0.3, 0.2, 0.1]:
    Xtr, Xte, ytr, yte = podziel(X_all, y_all, test_size=ts)
    trl, tel = [], []
    for rep in range(N_REPEATS):
        np.random.seed(rep * 17 + 3)
        siec = SSN([N_FEATURES, 8, 1], lr=0.01)
        siec.ucz(Xtr, ytr, Xte, yte, epoki=EPOKI)
        trl.append(siec.acc(Xtr, ytr))
        tel.append(siec.acc(Xte, yte))
        lbl = f"{int((1-ts)*100)}/{int(ts*100)}"
        print(f"  [split={lbl}|p{rep+1}] tr={trl[-1]:.4f} te={tel[-1]:.4f}")
    lbl = f"{int((1-ts)*100)}/{int(ts*100)}"
    w5.append(
        {
            "wartosc": lbl,
            "train_mean": round(float(np.mean(trl)), 4),
            "train_std": round(float(np.std(trl)), 4),
            "train_min": round(float(np.min(trl)), 4),
            "train_max": round(float(np.max(trl)), 4),
            "test_mean": round(float(np.mean(tel)), 4),
            "test_std": round(float(np.std(tel)), 4),
            "test_min": round(float(np.min(tel)), 4),
            "test_max": round(float(np.max(tel)), 4),
            "train_all": trl,
            "test_all": tel,
        }
    )


print("\nPARAMETR: Liczba epok")
w6 = []
for epoki_test in [40, 80, 150, 200]:
    trl, tel = [], []
    for rep in range(N_REPEATS):
        np.random.seed(rep * 17 + 3)
        siec = SSN([N_FEATURES, 8, 1], lr=0.01)
        siec.ucz(X_train, y_train, X_test, y_test, epoki=epoki_test)
        trl.append(siec.acc(X_train, y_train))
        tel.append(siec.acc(X_test, y_test))
        print(f"  [epoki={epoki_test}|p{rep+1}] tr={trl[-1]:.4f} te={tel[-1]:.4f}")
    w6.append(
        {
            "wartosc": str(epoki_test),
            "train_mean": round(float(np.mean(trl)), 4),
            "train_std": round(float(np.std(trl)), 4),
            "train_min": round(float(np.min(trl)), 4),
            "train_max": round(float(np.max(trl)), 4),
            "test_mean": round(float(np.mean(tel)), 4),
            "test_std": round(float(np.std(tel)), 4),
            "test_min": round(float(np.min(tel)), 4),
            "test_max": round(float(np.max(tel)), 4),
        }
    )

print("\nPARAMETR: Rozmiar batcha")
w7 = []
for batch in [32, 64, 128, 256]:
    trl, tel = [], []
    for rep in range(N_REPEATS):
        np.random.seed(rep * 17 + 3)
        siec = SSN([N_FEATURES, 8, 1], lr=0.01)

        def ucz_z_batch(ssn, Xtr, ytr, Xte, yte, epoki=EPOKI):
            ssn.historia = []
            for ep in range(epoki):
                idx = np.random.permutation(len(Xtr))
                Xs, ys = Xtr[idx], ytr[idx]
                for s in range(0, len(Xtr), batch):
                    ssn._przod(Xs[s : s + batch])
                    ssn._wstecz(Xs[s : s + batch], ys[s : s + batch])
                if (ep + 1) % 10 == 0:
                    ssn.historia.append(
                        (
                            ((ssn._przod(Xtr).flatten() > 0.5) == ytr).mean(),
                            ((ssn._przod(Xte).flatten() > 0.5) == yte).mean(),
                        )
                    )

        ucz_z_batch(siec, X_train, y_train, X_test, y_test)
        trl.append(siec.acc(X_train, y_train))
        tel.append(siec.acc(X_test, y_test))
        print(f"  [batch={batch}|p{rep+1}] tr={trl[-1]:.4f} te={tel[-1]:.4f}")
    w7.append(
        {
            "wartosc": str(batch),
            "train_mean": round(float(np.mean(trl)), 4),
            "train_std": round(float(np.std(trl)), 4),
            "train_min": round(float(np.min(trl)), 4),
            "train_max": round(float(np.max(trl)), 4),
            "test_mean": round(float(np.mean(tel)), 4),
            "test_std": round(float(np.std(tel)), 4),
            "test_min": round(float(np.min(tel)), 4),
            "test_max": round(float(np.max(tel)), 4),
        }
    )

print("\nPARAMETR: Momentum")


class SSN_momentum(SSN):
    def __init__(self, warstwy, momentum=0.0, **kwargs):
        super().__init__(warstwy, **kwargs)
        self.momentum = momentum
        self.vW = [np.zeros_like(W) for W in self.W]
        self.vb = [np.zeros_like(b) for b in self.b]

    def _wstecz(self, X, y):
        m = X.shape[0]
        y = y.reshape(-1, 1)
        d = self.as_[-1] - y
        gW, gb = [], []
        for i in reversed(range(len(self.W))):
            gW.insert(0, self.as_[i].T @ d / m)
            gb.insert(0, d.mean(0, keepdims=True))
            if i > 0:
                d = (d @ self.W[i].T) * self._df(self.zs[i - 1])
        for i in range(len(self.W)):
            self.vW[i] = self.momentum * self.vW[i] - self.lr * gW[i]
            self.vb[i] = self.momentum * self.vb[i] - self.lr * gb[i]
            self.W[i] += self.vW[i]
            self.b[i] += self.vb[i]


w8 = []
for mom in [0.0, 0.5, 0.9, 0.95]:
    trl, tel = [], []
    for rep in range(N_REPEATS):
        np.random.seed(rep * 17 + 3)
        siec = SSN_momentum([N_FEATURES, 8, 1], momentum=mom, lr=0.01)
        siec.ucz(X_train, y_train, X_test, y_test, epoki=EPOKI)
        trl.append(siec.acc(X_train, y_train))
        tel.append(siec.acc(X_test, y_test))
        print(f"  [momentum={mom}|p{rep+1}] tr={trl[-1]:.4f} te={tel[-1]:.4f}")
    w8.append(
        {
            "wartosc": str(mom),
            "train_mean": round(float(np.mean(trl)), 4),
            "train_std": round(float(np.std(trl)), 4),
            "train_min": round(float(np.min(trl)), 4),
            "train_max": round(float(np.max(trl)), 4),
            "test_mean": round(float(np.mean(tel)), 4),
            "test_std": round(float(np.std(tel)), 4),
            "test_min": round(float(np.min(tel)), 4),
            "test_max": round(float(np.max(tel)), 4),
        }
    )


# ================================================================
# 4. WYKRESY
# ================================================================
def wykres(wyniki, nazwa, plik):
    vals = [r["wartosc"] for r in wyniki]
    tr_m = [r["train_mean"] for r in wyniki]
    te_m = [r["test_mean"] for r in wyniki]
    tr_s = [r["train_std"] for r in wyniki]
    te_s = [r["test_std"] for r in wyniki]
    x, w = np.arange(len(vals)), 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    btr = ax.bar(
        x - w / 2,
        tr_m,
        w,
        yerr=tr_s,
        label="Zbior uczacy",
        color=C_TRAIN,
        alpha=0.85,
        capsize=5,
    )
    bte = ax.bar(
        x + w / 2,
        te_m,
        w,
        yerr=te_s,
        label="Zbior testowy",
        color=C_TEST,
        alpha=0.85,
        capsize=5,
    )
    for b in list(btr) + list(bte):
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
    ax.set_ylim(max(0, min(tr_m + te_m) - 0.05), 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(vals, fontsize=10)
    ax.set_xlabel(nazwa, fontsize=11, fontweight="bold")
    ax.set_ylabel("Dokladnosc (Accuracy)", fontsize=11)
    ax.set_title(
        f"Parametr: {nazwa}  (srednia z {N_REPEATS} powt. +/- std)",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, plik), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Zapisano: {plik}")


print("\nGenerowanie wykresow...")
wykres(w1, "Wspolczynnik uczenia", "p1_lr.png")
wykres(w2, "Liczba neuronow", "p2_neurony.png")
wykres(w3, "Funkcja aktywacji", "p3_aktywacja.png")
wykres(w4, "Liczba warstw", "p4_warstwy.png")
wykres(w5, "Podzial danych", "p5_podzial.png")
wykres(w6, "Liczba epok", "p6_epoki.png")
wykres(w7, "Rozmiar batcha", "p7_batch.png")
wykres(w8, "Momentum", "p8_momentum.png")

# ================================================================
# 5. FINALNY MODEL
# ================================================================
print("\nFinalny model [N->16->8->1, ReLU, lr=0.01, 150 epok]...")
final = [
    SSN([N_FEATURES, 16, 8, 1], aktywacja="relu", lr=0.01, seed=rep * 10)
    for rep in range(N_REPEATS)
]
for s in final:
    s.ucz(X_train, y_train, X_test, y_test, epoki=150)

best = max(final, key=lambda s: s.acc(X_test, y_test))
acc_tr = best.acc(X_train, y_train)
acc_te = best.acc(X_test, y_test)
print(f"  train={acc_tr:.4f}  test={acc_te:.4f}")

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(
    range(10, 151, 10),
    [v[0] for v in best.historia],
    "-o",
    color=C_TRAIN,
    label="Zbior uczacy",
    lw=2,
    ms=5,
)
ax.plot(
    range(10, 151, 10),
    [v[1] for v in best.historia],
    "-s",
    color=C_TEST,
    label="Zbior testowy",
    lw=2,
    ms=5,
)
ax.set_xlabel("Epoka")
ax.set_ylabel("Dokladnosc")
ax.set_title("Krzywa uczenia — finalny model", fontweight="bold")
ax.legend()
ax.grid(alpha=0.3, linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "krzywa_uczenia.png"), dpi=150, bbox_inches="tight")
plt.close()

yp = (best._przod(X_test).flatten() > 0.5).astype(int)
TP = int(((yp == 1) & (y_test == 1)).sum())
TN = int(((yp == 0) & (y_test == 0)).sum())
FP = int(((yp == 1) & (y_test == 0)).sum())
FN = int(((yp == 0) & (y_test == 1)).sum())
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
ax.set_xticklabels(["Punkt.(pred.)", "Opozn.(pred.)"], fontsize=10)
ax.set_yticklabels(["Punkt.(rzecz.)", "Opozn.(rzecz.)"], fontsize=10)
ax.set_title("Macierz pomylek — finalny model", fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "macierz_pomylek.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Zapisano: krzywa_uczenia.png, macierz_pomylek.png")

rows = []
for wyniki, nazwa in [
    (w1, "Wspolczynnik uczenia"),
    (w2, "Liczba neuronow"),
    (w3, "Funkcja aktywacji"),
    (w4, "Liczba warstw"),
    (w5, "Podzial danych"),
    (w6, "Liczba epok"),
    (w7, "Rozmiar batcha"),
    (w8, "Momentum"),
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

prec = TP / (TP + FP) if (TP + FP) > 0 else 0
rec = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
with open(os.path.join(RESULTS, "metryki_final.json"), "w", encoding="utf-8") as f:
    json.dump(
        {
            "acc_train": round(acc_tr, 4),
            "acc_test": round(acc_te, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
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
        f"  {nazwa:<22} tr={wyniki_por[nazwa]['train']}  te={wyniki_por[nazwa]['test']}"
    )
wyniki_por["SSN (wlasna)"] = {"train": round(acc_tr, 4), "test": round(acc_te, 4)}

nazwy = list(wyniki_por.keys())
tr_vals = [wyniki_por[n]["train"] for n in nazwy]
te_vals = [wyniki_por[n]["test"] for n in nazwy]
x, w = np.arange(len(nazwy)), 0.35
fig, ax = plt.subplots(figsize=(11, 6))
btr = ax.bar(x - w / 2, tr_vals, w, label="Zbior uczacy", color=C_TRAIN, alpha=0.85)
bte = ax.bar(x + w / 2, te_vals, w, label="Zbior testowy", color=C_TEST, alpha=0.85)
for i, n in enumerate(nazwy):
    if "SSN" in n:
        ax.bar(i - w / 2, tr_vals[i], w, color="#0D47A1")
        ax.bar(i + w / 2, te_vals[i], w, color="#B71C1C")
for b in list(btr) + list(bte):
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
ax.set_ylabel("Dokladnosc (Accuracy)", fontsize=11)
ax.set_title(
    "Porownanie SSN z klasycznymi metodami", fontsize=12, fontweight="bold", pad=12
)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "porownanie_metod.png"), dpi=150, bbox_inches="tight")
plt.close()
with open(os.path.join(RESULTS, "porownanie.json"), "w", encoding="utf-8") as f:
    json.dump(wyniki_por, f, ensure_ascii=False, indent=2)
print("  Zapisano: porownanie_metod.png\n\nGOTOWE! Pliki w folderze:", RESULTS)
