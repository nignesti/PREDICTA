"""
Valutazione bonus scommesse Serie A.

Nessuna API espone in modo affidabile le offerte bonus dei bookmaker (cambiano
spesso, e farne scraping dai loro siti e' un rischio legale/di ToS): i bonus
si inseriscono a mano in bonus_config.json, uno per bookmaker.

Il "valore netto" stimato qui NON e' un EV preciso da matched betting (che
richiederebbe coprire tutti gli esiti su piu' book): e' una stima
approssimata e trasparente del costo-opportunita' del bonus, utile solo per
ordinare le offerte fra loro, non per prevedere un guadagno esatto.

Formula, per tipo di bonus:
- "freebet" (puntata gratuita, si perde lo stake se si perde la giocata):
      valore_netto = importo * (1 - margine_stimato)
  si stima di recuperare l'importo al netto del solo margine del bookmaker.
- "deposito" (bonus accreditato con un requisito di scommessa/rollover):
      valore_netto = importo - importo * rollover * margine_stimato
  si stima la perdita attesa nel girare il fatturato richiesto (rollover
  volte l'importo) al margine stimato.
In entrambi i casi il risultato e' clampato a 0 (un bonus non vale mai meno
di niente in questo modello).
"""
import json

import pandas as pd

PERCORSO_DEFAULT = "predicta/serie_a/bonus/bonus_config.json"
MARGINE_DEFAULT = 0.05


def carica_bonus(percorso=PERCORSO_DEFAULT):
    with open(percorso, encoding="utf-8") as f:
        bonus = json.load(f)
    return pd.DataFrame(bonus)


def valore_netto(bonus):
    """bonus: dict con almeno 'importo' e 'tipo' ('freebet' o 'deposito').
    Opzionali: 'margine_stimato' (default 5%), 'rollover' (default 1, solo
    per 'deposito'). Restituisce il valore netto stimato, >= 0."""
    importo = float(bonus["importo"])
    margine = float(bonus.get("margine_stimato", MARGINE_DEFAULT))
    tipo = bonus["tipo"]
    if tipo == "freebet":
        valore = importo * (1 - margine)
    elif tipo == "deposito":
        rollover = float(bonus.get("rollover", 1))
        valore = importo - importo * rollover * margine
    else:
        raise ValueError(f"tipo bonus sconosciuto: {tipo!r} (atteso 'freebet' o 'deposito')")
    return max(valore, 0.0)


def classifica_bonus(df):
    """Aggiunge la colonna 'valore_netto' e ordina i bonus dal piu'
    conveniente al meno conveniente."""
    if df.empty:
        return df.assign(valore_netto=pd.Series(dtype=float))
    risultato = df.assign(valore_netto=[valore_netto(r) for r in df.to_dict("records")])
    return risultato.sort_values("valore_netto", ascending=False).reset_index(drop=True)
