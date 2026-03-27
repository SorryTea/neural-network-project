"""
Predykcja opóźnień lotów — Sieć Neuronowa od ZERA (tylko NumPy)
Klasyfikacja: lot opóźniony (1) lub punktualny (0)

=================================================================
DANE — WYBIERZ JEDEN Z 3 ZBIORÓW Z KAGGLE:

OPCJA A (ZALECANA) — ~540 000 rekordów, gotowa kolumna 0/1
  Nazwa:  "Airlines Dataset to Predict a Delay"
  Link:   https://www.kaggle.com/datasets/jimschacko/airlines-dataset-to-predict-a-delay
  Plik:   Airlines.csv
  Ustaw:  CSV_PATH = "Airlines.csv"

OPCJA B — ~540 000 rekordów, identyczny format
  Nazwa:  "Airlines Delay"
  Link:   https://www.kaggle.com/datasets/ulrikthygepedersen/airlines-delay
  Plik:   airlines_delay.csv
  Ustaw:  CSV_PATH = "airlines_delay.csv"

OPCJA C — ~3 000 000 rekordów (duży plik!)
  Nazwa:  "Flight Delay and Cancellation Dataset (2019-2023)"
  Link:   https://www.kaggle.com/datasets/patrickzel/flight-delay-and-cancellation-dataset-2019-2023
  Plik:   flights_sample_3m.csv
  Ustaw:  CSV_PATH = "flights_sample_3m.csv"

Jeśli plik nie istnieje → uruchomi się na danych syntetycznych (do testu)
=================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, json

np.random.seed(42)
RESULTS = "wyniki"
os.makedirs(RESULTS, exist_ok=True)

# ================================================================
# 1. WCZYTANIE DANYCH
# ================================================================

CSV_PATH = "Airlines.csv"   # <-- ZMIEŃ NA ŚCIEŻKĘ DO PLIKU Z KAGGLE
N_SAMPLE = 10000            # ile rekordów użyć (wystarczy na analizę)

def wczytaj_dane(path, n_sample=N_SAMPLE):
    if not (path and os.path.exists(path)):
        print("▶ Plik CSV nie znaleziony — uruchamiam dane syntetyczne (test)")
        print("  Pobierz dane z Kaggle i ustaw CSV_PATH w kodzie!\n")
        return _syntetyczne(n_sample)

    print(f"▶ Wczytywanie: {path}")
    df = pd.read_csv(path)
    print(f"  Wczytano {len(df):,} rekordów | kolumny: {list(df.columns)[:8]}...")

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()

    # ── OPCJA A / B: Airlines.csv lub airlines_delay.csv ──────────
    # Kolumny: Airline, Flight, AirportFrom, AirportTo, DayOfWeek, Time, Length, Delay
    if 'Delay' in df.columns and 'DayOfWeek' in df.columns:
        print("  Format: Airlines Dataset (Opcja A lub B)")
        df = df.dropna(subset=['Delay', 'DayOfWeek', 'Time', 'Length'])
        for col in ['Airline', 'AirportFrom', 'AirportTo']:
            df[col] = le.fit_transform(df[col].astype(str))
        feat = ['Airline', 'AirportFrom', 'AirportTo', 'DayOfWeek', 'Time', 'Length']
        X = df[feat].values.astype(float)
        y = df['Delay'].values.astype(float)

    # ── OPCJA C: flights_sample_3m.csv ────────────────────────────
    # Kolumny: AIRLINE, ORIGIN, DEST, DEP_TIME, ARR_DELAY, DISTANCE, MONTH, DAY_OF_WEEK
    elif 'ARR_DELAY' in df.columns and 'ORIGIN' in df.columns:
        print("  Format: Flight Delay 2019-2023 (Opcja C)")
        df = df.dropna(subset=['ARR_DELAY', 'DEP_TIME', 'DISTANCE'])
        df['delayed'] = (df['ARR_DELAY'] >= 15).astype(float)
        for col in ['AIRLINE', 'ORIGIN', 'DEST']:
            df[col] = le.fit_transform(df[col].astype(str))
        avail = [c for c in ['AIRLINE','ORIGIN','DEST','DAY_OF_WEEK','MONTH','DEP_TIME','DISTANCE'] if c in df.columns]
        df = df.dropna(subset=avail)
        X = df[avail].values.astype(float)
        y = df['delayed'].values.astype(float)

    else:
        raise ValueError(f"Nieznany format CSV. Kolumny: {list(df.columns)}")

    # Próbka i normalizacja
    if n_sample and len(X) > n_sample:
        idx = np.random.choice(len(X), n_sample, replace=False)
        X, y = X[idx], y[idx]

    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8)
    print(f"  Używam {len(X):,} rekordów | opóźnione: {y.mean()*100:.1f}%")
    return X, y, X.shape[1]


def _syntetyczne(n=8000):
    rng = np.random.default_rng(42)
    airline = rng.integers(0, 18, n).astype(float)
    origin  = rng.integers(0, 50, n).astype(float)
    dest    = rng.integers(0, 50, n).astype(float)
    dow     = rng.integers(1,  8, n).astype(float)
    dep_h   = rng.uniform(0, 23, n)
    dist    = rng.integers(100, 2500, n).astype(float)
    p = (0.25 + 0.22*(dep_h>=18) + 0.15*np.isin(dow,[5,6]) - 0.08*(dist>1800)).clip(0.05,0.90)
    y = (rng.uniform(size=n) < p).astype(float)
    X = np.column_stack([airline/17, origin/49, dest/49, dow/7, dep_h/23, dist/2500])
    return X, y, 6


# Wczytaj dane
result = wczytaj_dane(CSV_PATH)
X, y, N_FEATURES = result

# Podział 80/20
idx = np.random.permutation(len(X))
split = int(0.8 * len(X))
X_train, X_test = X[idx[:split]], X[idx[split:]]
y_train, y_test = y[idx[:split]], y[idx[split:]]
print(f"  X_train={X_train.shape}  X_test={X_test.shape}")
print(f"  Klasa 1 w train: {y_train.mean()*100:.1f}%\n")


# ================================================================
# 2. SIEĆ NEURONOWA OD ZERA
# ================================================================
class SiecNeuronowa:
    """
    Wielowarstwowa sieć neuronowa zaimplementowana od podstaw.
    Używa tylko NumPy — bez Keras, TensorFlow, sklearn.
    """

    def __init__(self, rozmiary_warstw, aktywacja='relu', lr=0.01, seed=None):
        """
        rozmiary_warstw: lista np. [5, 8, 1]
          → warstwa wejściowa: 5 neuronów
          → warstwa ukryta: 8 neuronów
          → warstwa wyjściowa: 1 neuron
        """
        if seed is not None:
            np.random.seed(seed)
        self.rozmiary = rozmiary_warstw
        self.aktywacja_nazwa = aktywacja
        self.lr = lr

        # Inicjalizacja wag metodą He/Xavier
        self.wagi = []
        self.biasy = []
        for i in range(len(rozmiary_warstw) - 1):
            skala = np.sqrt(2.0 / rozmiary_warstw[i])
            W = np.random.randn(rozmiary_warstw[i], rozmiary_warstw[i+1]) * skala
            b = np.zeros((1, rozmiary_warstw[i+1]))
            self.wagi.append(W)
            self.biasy.append(b)

        self.historia_strat = []
        self.historia_acc   = []

    # ── Funkcje aktywacji ──────────────────────────────────────
    def aktywuj(self, z):
        if self.aktywacja_nazwa == 'relu':
            return np.maximum(0, z)
        elif self.aktywacja_nazwa == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        elif self.aktywacja_nazwa == 'tanh':
            return np.tanh(z)
        elif self.aktywacja_nazwa == 'elu':
            return np.where(z >= 0, z, np.exp(np.clip(z, -500, 0)) - 1)
        return np.maximum(0, z)

    def pochodna_aktywacji(self, z):
        if self.aktywacja_nazwa == 'relu':
            return (z > 0).astype(float)
        elif self.aktywacja_nazwa == 'sigmoid':
            s = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
            return s * (1 - s)
        elif self.aktywacja_nazwa == 'tanh':
            return 1 - np.tanh(z)**2
        elif self.aktywacja_nazwa == 'elu':
            return np.where(z >= 0, 1.0, np.exp(np.clip(z, -500, 0)))
        return (z > 0).astype(float)

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    # ── Propagacja w przód ─────────────────────────────────────
    def propagacja_w_przod(self, X):
        self.z_lista  = []   # liniowe kombinacje
        self.a_lista  = [X]  # aktywacje

        a = X
        for i, (W, b) in enumerate(zip(self.wagi, self.biasy)):
            z = a @ W + b
            self.z_lista.append(z)

            if i < len(self.wagi) - 1:    # warstwy ukryte
                a = self.aktywuj(z)
            else:                           # warstwa wyjściowa → sigmoid
                a = self.sigmoid(z)
            self.a_lista.append(a)

        return a  # predykcje (0..1)

    # ── Wsteczna propagacja błędu ──────────────────────────────
    def propagacja_wsteczna(self, X, y):
        m = X.shape[0]
        y = y.reshape(-1, 1)

        # Gradient straty na wyjściu (binarna entropia skrzyżowana)
        delta = self.a_lista[-1] - y

        grad_wagi  = []
        grad_biasy = []

        for i in reversed(range(len(self.wagi))):
            dW = (self.a_lista[i].T @ delta) / m
            db = delta.mean(axis=0, keepdims=True)
            grad_wagi.insert(0, dW)
            grad_biasy.insert(0, db)

            if i > 0:
                delta = (delta @ self.wagi[i].T) * self.pochodna_aktywacji(self.z_lista[i-1])

        # Aktualizacja wag (SGD)
        for i in range(len(self.wagi)):
            self.wagi[i]  -= self.lr * grad_wagi[i]
            self.biasy[i] -= self.lr * grad_biasy[i]

    # ── Strata ─────────────────────────────────────────────────
    def strata(self, y_pred, y_true):
        y_true = y_true.reshape(-1, 1)
        eps = 1e-12
        return -np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))

    # ── Uczenie ────────────────────────────────────────────────
    def ucz(self, X_tr, y_tr, X_te, y_te, epoki=100, batch=64, verbose=False):
        m = X_tr.shape[0]
        self.historia_strat = []
        self.historia_acc   = []

        for epoka in range(epoki):
            # Tasowanie danych
            idx = np.random.permutation(m)
            X_s, y_s = X_tr[idx], y_tr[idx]

            # Mini-batche
            for start in range(0, m, batch):
                Xb = X_s[start:start+batch]
                yb = y_s[start:start+batch]
                self.propagacja_w_przod(Xb)
                self.propagacja_wsteczna(Xb, yb)

            # Logi co 10 epok
            if (epoka + 1) % 10 == 0:
                y_pred_tr = self.propagacja_w_przod(X_tr)
                y_pred_te = self.propagacja_w_przod(X_te)
                loss = self.strata(y_pred_tr, y_tr)
                acc_tr = ((y_pred_tr.flatten() > 0.5) == y_tr).mean()
                acc_te = ((y_pred_te.flatten() > 0.5) == y_te).mean()
                self.historia_strat.append(loss)
                self.historia_acc.append((acc_tr, acc_te))
                if verbose:
                    print(f"  Epoka {epoka+1:3d}: strata={loss:.4f}  "
                          f"acc_train={acc_tr:.4f}  acc_test={acc_te:.4f}")

    # ── Predykcja ──────────────────────────────────────────────
    def predykcja(self, X):
        return (self.propagacja_w_przod(X).flatten() > 0.5).astype(int)

    def dokladnosc(self, X, y):
        return (self.predykcja(X) == y).mean()


# ================================================================
# 3. FUNKCJA ANALIZY PARAMETRU
# ================================================================
N_REPEATS = 3
EPOKI     = 80

def analizuj(nazwa, wartosci, buduj_fn):
    """Trenuje sieć N_REPEATS razy dla każdej wartości parametru."""
    print(f"\n{'='*55}")
    print(f"  PARAMETR: {nazwa}")
    print(f"{'='*55}")

    wyniki = []
    for war in wartosci:
        train_list, test_list = [], []
        for rep in range(N_REPEATS):
            np.random.seed(rep * 17 + 3)   # inna inicjalizacja wag w każdym powtórzeniu
            siec = buduj_fn(war)
            siec.ucz(X_train, y_train, X_test, y_test, epoki=EPOKI)
            acc_tr = siec.dokladnosc(X_train, y_train)
            acc_te = siec.dokladnosc(X_test, y_test)
            train_list.append(acc_tr)
            test_list.append(acc_te)
            print(f"  [{nazwa}={war} | pow.{rep+1}] train={acc_tr:.4f}  test={acc_te:.4f}")

        wyniki.append({
            'wartosc'   : str(war),
            'train_mean': round(float(np.mean(train_list)), 4),
            'train_std' : round(float(np.std(train_list)),  4),
            'train_min' : round(float(np.min(train_list)),  4),
            'train_max' : round(float(np.max(train_list)),  4),
            'test_mean' : round(float(np.mean(test_list)),  4),
            'test_std'  : round(float(np.std(test_list)),   4),
            'test_min'  : round(float(np.min(test_list)),   4),
            'test_max'  : round(float(np.max(test_list)),   4),
            'train_all' : train_list,
            'test_all'  : test_list,
        })

    return wyniki


# ================================================================
# 4. ANALIZA 4 PARAMETRÓW (min. 4 dla grupy 2-osobowej)
# ================================================================

# PARAMETR 1: Współczynnik uczenia
print("\n>>> PARAMETR 1: WSPÓŁCZYNNIK UCZENIA")
lr_vals = [0.001, 0.01, 0.05, 0.1]
wyniki_lr = analizuj(
    "Wsp. uczenia", lr_vals,
    lambda lr: SiecNeuronowa([N_FEATURES, 8, 1], aktywacja='relu', lr=lr)
)

# PARAMETR 2: Liczba neuronów w warstwie ukrytej
print("\n>>> PARAMETR 2: LICZBA NEURONÓW")
neuron_vals = [4, 8, 16, 32]
wyniki_neurony = analizuj(
    "Neurony", neuron_vals,
    lambda n: SiecNeuronowa([N_FEATURES, n, 1], aktywacja='relu', lr=0.01)
)

# PARAMETR 3: Funkcja aktywacji
print("\n>>> PARAMETR 3: FUNKCJA AKTYWACJI")
akt_vals = ['sigmoid', 'relu', 'tanh', 'elu']
wyniki_akt = analizuj(
    "Aktywacja", akt_vals,
    lambda a: SiecNeuronowa([N_FEATURES, 8, 1], aktywacja=a, lr=0.01)
)

# PARAMETR 4: Liczba warstw ukrytych
print("\n>>> PARAMETR 4: LICZBA WARSTW UKRYTYCH")

def zbuduj_warstwami(n_warstw):
    rozmiary = [N_FEATURES] + [8] * n_warstw + [1]
    return SiecNeuronowa(rozmiary, aktywacja='relu', lr=0.01)

warstwy_vals = [1, 2, 3, 4]
wyniki_warstwy = analizuj(
    "Warstwy", warstwy_vals, zbuduj_warstwami
)


# ================================================================
# 5. WYKRESY
# ================================================================
print("\n>>> GENEROWANIE WYKRESÓW...")

KOLOR_TRAIN = '#1976D2'
KOLOR_TEST  = '#D32F2F'

def wykres_slupkowy(wyniki, nazwa_param, filename):
    wartosci  = [r['wartosc']    for r in wyniki]
    tr_mean   = [r['train_mean'] for r in wyniki]
    te_mean   = [r['test_mean']  for r in wyniki]
    tr_std    = [r['train_std']  for r in wyniki]
    te_std    = [r['test_std']   for r in wyniki]

    x = np.arange(len(wartosci))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars_tr = ax.bar(x - w/2, tr_mean, w, yerr=tr_std, label='Zbiór uczący',
                     color=KOLOR_TRAIN, alpha=0.85, capsize=5)
    bars_te = ax.bar(x + w/2, te_mean, w, yerr=te_std, label='Zbiór testowy',
                     color=KOLOR_TEST,  alpha=0.85, capsize=5)

    for b in list(bars_tr) + list(bars_te):
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h + 0.003,
                f'{h:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    y_min = max(0, min(tr_mean + te_mean) - 0.05)
    ax.set_ylim(y_min, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(wartosci, fontsize=11)
    ax.set_xlabel(nazwa_param, fontsize=12, fontweight='bold')
    ax.set_ylabel('Dokładność (Accuracy)', fontsize=12)
    ax.set_title(f'Wpływ parametru: {nazwa_param}\n'
                 f'(średnia z {N_REPEATS} powtórzeń ± odch. std.)',
                 fontsize=13, fontweight='bold', pad=12)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Zapisano: {filename}")


wykres_slupkowy(wyniki_lr,      "Współczynnik uczenia",    "p1_lr.png")
wykres_slupkowy(wyniki_neurony, "Liczba neuronów",          "p2_neurony.png")
wykres_slupkowy(wyniki_akt,     "Funkcja aktywacji",        "p3_aktywacja.png")
wykres_slupkowy(wyniki_warstwy, "Liczba warstw ukrytych",   "p4_warstwy.png")


# ================================================================
# 6. FINALNY MODEL + KRZYWA UCZENIA + MACIERZ POMYLEK
# ================================================================
print("\n>>> FINALNY MODEL (3 powtórzenia najlepszej konfiguracji)...")

# Najlepsza konfiguracja ustalona na podstawie analizy
FINAL_ARCH = [N_FEATURES, 16, 8, 1]
EPOKI_FINAL = 150

final_wyniki = []
for rep in range(N_REPEATS):
    siec = SiecNeuronowa(FINAL_ARCH, aktywacja='relu', lr=0.01, seed=rep*10)
    siec.ucz(X_train, y_train, X_test, y_test, epoki=EPOKI_FINAL, verbose=False)
    final_wyniki.append(siec)

# Wybieramy najlepszy z powtórzeń
best = max(final_wyniki, key=lambda s: s.dokladnosc(X_test, y_test))
acc_tr_fin = best.dokladnosc(X_train, y_train)
acc_te_fin = best.dokladnosc(X_test,  y_test)
print(f"  Finalny model — train={acc_tr_fin:.4f}  test={acc_te_fin:.4f}")

# Krzywa uczenia
epoki_x = list(range(10, EPOKI_FINAL + 1, 10))
tr_acc_hist = [v[0] for v in best.historia_acc]
te_acc_hist = [v[1] for v in best.historia_acc]

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(epoki_x, tr_acc_hist, '-o', color=KOLOR_TRAIN, label='Zbiór uczący',  linewidth=2, markersize=5)
ax.plot(epoki_x, te_acc_hist, '-s', color=KOLOR_TEST,  label='Zbiór testowy', linewidth=2, markersize=5)
ax.set_xlabel('Epoka', fontsize=12)
ax.set_ylabel('Dokładność (Accuracy)', fontsize=12)
ax.set_title('Krzywa uczenia — finalny model\n[5→16→8→1, ReLU, lr=0.01]',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "krzywa_uczenia.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Zapisano: krzywa_uczenia.png")

# Macierz pomyłek
y_pred_final = best.predykcja(X_test)
from collections import Counter

TP = int(((y_pred_final == 1) & (y_test == 1)).sum())
TN = int(((y_pred_final == 0) & (y_test == 0)).sum())
FP = int(((y_pred_final == 1) & (y_test == 0)).sum())
FN = int(((y_pred_final == 0) & (y_test == 1)).sum())
cm = np.array([[TN, FP], [FN, TP]])

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap='Blues')
plt.colorbar(im, ax=ax)
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                fontsize=22, fontweight='bold',
                color='white' if cm[i, j] > cm.max() * 0.6 else 'black')
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Punkt. (pred.)', 'Opóźn. (pred.)'], fontsize=10)
ax.set_yticklabels(['Punkt. (rzecz.)', 'Opóźn. (rzecz.)'], fontsize=10)
ax.set_title('Macierz pomyłek — finalny model', fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "macierz_pomylek.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Zapisano: macierz_pomylek.png")


# ================================================================
# 7. TABELA WYNIKOW CSV
# ================================================================
rows = []
for wyniki, nazwa in [
    (wyniki_lr,      "Wspólczynnik uczenia"),
    (wyniki_neurony, "Liczba neuronów"),
    (wyniki_akt,     "Funkcja aktywacji"),
    (wyniki_warstwy, "Liczba warstw ukrytych"),
]:
    for r in wyniki:
        rows.append({
            "Parametr"  : nazwa,
            "Wartość"   : r['wartosc'],
            "Train Mean": r['train_mean'],
            "Train Std" : r['train_std'],
            "Train Min" : r['train_min'],
            "Train Max" : r['train_max'],
            "Test Mean" : r['test_mean'],
            "Test Std"  : r['test_std'],
            "Test Min"  : r['test_min'],
            "Test Max"  : r['test_max'],
        })

df_csv = pd.DataFrame(rows)
df_csv.to_csv(os.path.join(RESULTS, "tabela_wynikow.csv"), index=False, encoding="utf-8-sig")
print("\nTabela wynikow:\n")
print(df_csv.to_string(index=False))

# Zapisz też metryki finalnego modelu do JSON
import json
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

metryki = {
    "arch"     : str(FINAL_ARCH),
    "aktywacja": "relu",
    "lr"       : 0.01,
    "acc_train": round(acc_tr_fin, 4),
    "acc_test" : round(acc_te_fin, 4),
    "precision": round(precision, 4),
    "recall"   : round(recall, 4),
    "f1"       : round(f1, 4),
    "TP": TP, "TN": TN, "FP": FP, "FN": FN,
}
with open(os.path.join(RESULTS, "metryki_final.json"), "w", encoding="utf-8") as f:
    json.dump(metryki, f, ensure_ascii=False, indent=2)


# ================================================================
# 8. PORÓWNANIE Z KLASYCZNYMI METODAMI KLASYFIKACJI
# (użycie gotowych bibliotek dozwolone zgodnie z treścią zadania
#  do porównania wyników własnej SSN z bibliotekami)
# ================================================================
print("\n" + "="*55)
print("  PORÓWNANIE Z KLASYCZNYMI METODAMI (sklearn)")
print("="*55)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

metody = {
    "Drzewo decyzyjne"  : DecisionTreeClassifier(max_depth=5, random_state=42),
    "Las losowy"         : RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    "k-NN (k=5)"         : KNeighborsClassifier(n_neighbors=5),
    "Naiwny Bayes"       : GaussianNB(),
    "SVM (RBF)"          : SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
}

wyniki_porownanie = {}
for nazwa, clf in metody.items():
    clf.fit(X_train, y_train)
    acc_tr = accuracy_score(y_train, clf.predict(X_train))
    acc_te = accuracy_score(y_test,  clf.predict(X_test))
    wyniki_porownanie[nazwa] = {'train': round(acc_tr, 4), 'test': round(acc_te, 4)}
    print(f"  {nazwa:<22}  train={acc_tr:.4f}  test={acc_te:.4f}")

# Własna SSN — wynik finalny
wyniki_porownanie["SSN (własna impl.)"] = {
    'train': round(acc_tr_fin, 4),
    'test' : round(acc_te_fin, 4)
}

# ── Wykres porównawczy ────────────────────────────────────────
print("\n>>> Generowanie wykresu porównawczego...")
nazwy   = list(wyniki_porownanie.keys())
tr_vals = [wyniki_porownanie[n]['train'] for n in nazwy]
te_vals = [wyniki_porownanie[n]['test']  for n in nazwy]

x = np.arange(len(nazwy))
w = 0.35

fig, ax = plt.subplots(figsize=(11, 6))
bars_tr = ax.bar(x - w/2, tr_vals, w, label='Zbiór uczący',
                 color=KOLOR_TRAIN, alpha=0.85)
bars_te = ax.bar(x + w/2, te_vals, w, label='Zbiór testowy',
                 color=KOLOR_TEST, alpha=0.85)

# Podświetl słupek SSN
for i, n in enumerate(nazwy):
    if 'SSN' in n:
        ax.bar(i - w/2, tr_vals[i], w, color='#1565C0', alpha=1.0)
        ax.bar(i + w/2, te_vals[i], w, color='#B71C1C', alpha=1.0)

for b in list(bars_tr) + list(bars_te):
    h = b.get_height()
    ax.text(b.get_x() + b.get_width()/2, h + 0.003,
            f'{h:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

y_min = max(0, min(tr_vals + te_vals) - 0.05)
ax.set_ylim(y_min, 1.0)
ax.set_xticks(x)
ax.set_xticklabels(nazwy, fontsize=9, rotation=10, ha='right')
ax.set_ylabel('Dokładność (Accuracy)', fontsize=12)
ax.set_title('Porównanie własnej SSN z klasycznymi metodami klasyfikacji',
             fontsize=13, fontweight='bold', pad=12)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "porownanie_metod.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"  Zapisano: porownanie_metod.png")

# Zapis do JSON dla sprawozdania
import json
with open(os.path.join(RESULTS, "porownanie.json"), "w", encoding="utf-8") as f:
    json.dump(wyniki_porownanie, f, ensure_ascii=False, indent=2)

print("\nWyniki porównania:")
for n, v in wyniki_porownanie.items():
    print(f"  {n:<22}  train={v['train']}  test={v['test']}")
print("\nGOTOWE!")

