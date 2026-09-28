"""
Quale quota di mercato e' il miglior previsore 1X2, e le quote contengono
ancora informazione non sfruttata?

In produzione il previsore e' la media dei bookmaker in chiusura convertita
con Shin (peso del modello statistico: zero, vedi ROADMAP.md). Restano da
provare solo varianti del mercato stesso, tutte con colonne gia' nei file
grezzi di football-data:

- Pinnacle in chiusura (il book "sharp") con Shin, proporzionale e power;
- Betfair Exchange in chiusura (copertura parziale);
- ricalibrazione logistica delle probabilita' Pinnacle, piu' movimento
  apertura->chiusura, dispersione fra bookmaker (Max/Avg) ed effetto lega,
  allenata walk-forward sulle sole stagioni precedenti (test 2020-2025);
- apertura contro chiusura: quanto vale leggere la quota tardi.

Protocollo: 5 campionati, 2019-2025, verdetto su RPS con bootstrap appaiato
(protocollo.confronta).

    python valida_fonte_quote.py
"""
import os

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from sklearn.linear_model import LogisticRegression

import protocollo as pr
from modello import probabilita_shin

LEGHE = {"I1": "stagioni", "E0": "altre_leghe/E0", "SP1": "altre_leghe/SP1",
         "D1": "altre_leghe/D1", "F1": "altre_leghe/F1"}
TRIPLE = {k: [k + c for c in "HDA"] for k in ["AvgC", "PSC", "BFEC", "MaxC", "Avg", "PS"]}


def carica():
    df = pd.concat([pd.read_csv(f"{cartella}/{s}.txt", encoding="utf-8-sig").assign(Lega=lega, Stagione=int(s))
                    for lega, cartella in LEGHE.items() for s in pr.STAGIONI_TEST
                    if os.path.exists(f"{cartella}/{s}.txt")], ignore_index=True)
    quote = TRIPLE["AvgC"] + TRIPLE["PSC"] + TRIPLE["PS"] + TRIPLE["Avg"] + TRIPLE["MaxC"]
    df = df.dropna(subset=quote + ["FTR"])
    return df[(df[quote] > 1).all(axis=1)].reset_index(drop=True)


def shin(df, k):
    return np.array([probabilita_shin(q) for q in df[TRIPLE[k]].values])


def proporzionale(df, k):
    pi = 1 / df[TRIPLE[k]].values
    return pi / pi.sum(axis=1, keepdims=True)


def power(df, k):
    """p_i = pi_i^a, con a tale che la somma sia 1."""
    return np.array([pi ** brentq(lambda a: (pi ** a).sum() - 1, 0.5, 3) for pi in 1 / df[TRIPLE[k]].values])


def stampa(e):
    print(f"  {e.nome_a:28s} RPS {e.rps_a:.5f} vs {e.rps_b:.5f}  dRPS IC95% [{e.ic_rps[0]:+.5f}, {e.ic_rps[1]:+.5f}]"
          f"  {e.verdetto:15s} acc {e.acc_a:.2%} / {e.acc_b:.2%}  n={e.n}")


def walk_forward(x, y, stagioni, prima_test=2020):
    out = np.zeros((len(y), 3))
    for s in sorted(set(stagioni[stagioni >= prima_test])):
        tr, te = stagioni < s, stagioni == s
        clf = LogisticRegression(max_iter=2000).fit(x[tr], y[tr])
        out[te] = clf.predict_proba(x[te])[:, [list(clf.classes_).index(c) for c in pr.ORDINE_CLASSI]]
    return out


def main():
    df = carica()
    y = df["FTR"].map({"H": "1", "D": "X", "A": "2"}).values
    print(f"Partite: {len(df)} {df.groupby('Lega').size().to_dict()}")

    produzione = shin(df, "AvgC")
    psc = shin(df, "PSC")

    print("\nFonte e conversione (contro media in chiusura + Shin, produzione):")
    for nome, p in [("Pinnacle chiusura Shin", psc), ("Pinnacle chiusura proporz.", proporzionale(df, "PSC")),
                    ("Pinnacle chiusura power", power(df, "PSC"))]:
        stampa(pr.confronta(nome, p, "produzione", produzione, y))
    bfe = ((df[TRIPLE["BFEC"]] > 1).all(axis=1)).values
    print(f"  (Betfair Exchange in chiusura: copertura {bfe.mean():.0%})")
    stampa(pr.confronta("Betfair Exchange chiusura", proporzionale(df[bfe], "BFEC"),
                        "produzione", produzione[bfe], y[bfe]))

    print("\nInformazione residua nelle quote (contro Pinnacle chiusura Shin, test 2020-2025):")
    lp = np.log(psc)
    x = np.c_[lp[:, 0] - lp[:, 1], lp[:, 2] - lp[:, 1]]
    movimento = np.log(shin(df, "PS")) - lp
    dispersione = np.log(df[TRIPLE["MaxC"]].values / df[TRIPLE["AvgC"]].values)
    stagioni = df["Stagione"].values
    test = stagioni >= 2020
    for nome, xx in [("ricalibrazione", x), ("+ movimento", np.c_[x, movimento]),
                     ("+ movimento + dispersione", np.c_[x, movimento, dispersione]),
                     ("+ mov + disp + lega", np.c_[x, movimento, dispersione, pd.get_dummies(df["Lega"]).values])]:
        stampa(pr.confronta(nome, walk_forward(xx, y, stagioni)[test], "Pinnacle chiusura", psc[test], y[test]))

    print("\nTempismo (contro Pinnacle chiusura Shin):")
    stampa(pr.confronta("Pinnacle apertura", shin(df, "PS"), "Pinnacle chiusura", psc, y))
    stampa(pr.confronta("media apertura", shin(df, "Avg"), "Pinnacle chiusura", psc, y))

    print("\nPer campionato, Pinnacle contro media (chiusura, Shin):")
    for lega, g in df.groupby("Lega"):
        i = g.index.values
        stampa(pr.confronta(f"Pinnacle {lega}", psc[i], "media", produzione[i], y[i]))


if __name__ == "__main__":
    main()
