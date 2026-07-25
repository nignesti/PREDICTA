"""
Costruzione e analisi di una schedina 1X2 a partire dalle quote.

Il modello di produzione assegna il 100% del peso al mercato (vedi ROADMAP.md):
la previsione per una singola partita e' quindi la probabilita' implicita nelle
quote, ripulita dal margine del bookmaker con la correzione di Shin. Questo
modulo aggiunge lo strato che serve per una schedina: quali partite conviene
includere, e che probabilita' ha davvero di uscire.

Due cose che il calcolo mostra e che non sono ovvie:

1. **La leva principale e' la SELEZIONE delle partite, non il modello.** Su
   12.459 partite (5 campionati, 2019-2025), giocare il favorito su tutte le
   partite da una schedina da 13 azzeccata nello 0,033% dei casi; giocarlo solo
   sulle partite con confidenza >= 70% porta la stessa schedina al 3,6%. Cento
   volte tanto, a parita' di modello.

2. **Probabilita' e moltiplicatore si muovono in direzioni opposte, e il loro
   prodotto resta sotto 1.** Selezionare partite piu' sicure alza la probabilita'
   ma abbassa le quote, e il ritorno atteso resta intorno al 78-82% qualunque
   soglia si scelga. E' il margine del bookmaker, e nessun modello lo sposta:
   servirebbe un vantaggio informativo sul mercato, che il progetto ha misurato
   non esserci.

Il ritorno atteso calcolato qui e' quindi sempre < 1. Il modulo serve a scegliere
consapevolmente il punto sulla curva rischio/premio e a sapere in anticipo quanto
e' improbabile una schedina piena, non a promettere un guadagno.
"""
import numpy as np

from modello import probabilita_shin

ESITI = ("1", "X", "2")


def analizza_partita(quota_1, quota_x, quota_2):
    """Probabilita' vere dei tre esiti e pronostico per una singola partita.

    Restituisce un dict con le probabilita' (gia' ripulite dal margine con
    Shin), l'esito piu' probabile, la sua probabilita' ("confidenza") e la
    quota corrispondente, piu' il margine del bookmaker su quella partita."""
    quote = [float(quota_1), float(quota_x), float(quota_2)]
    if any(q <= 1.0 for q in quote):
        raise ValueError("Le quote decimali devono essere maggiori di 1")

    probabilita = probabilita_shin(quote)
    i = int(np.argmax(probabilita))
    return {
        "prob_1": probabilita[0], "prob_X": probabilita[1], "prob_2": probabilita[2],
        "pronostico": ESITI[i],
        "confidenza": probabilita[i],
        "quota_pronostico": quote[i],
        "margine": sum(1.0 / q for q in quote) - 1.0,
    }


def distribuzione_numero_esatti(probabilita):
    """Distribuzione del numero di pronostici azzeccati (Poisson binomiale).

    Le partite hanno probabilita' diverse fra loro, quindi il numero di esiti
    corretti NON segue una binomiale: serve la convoluzione esatta, calcolata
    qui in programmazione dinamica. `dist[k]` e' la probabilita' di azzeccarne
    esattamente k."""
    probabilita = list(probabilita)
    dist = np.zeros(len(probabilita) + 1)
    dist[0] = 1.0
    for indice, p in enumerate(probabilita, start=1):
        nuovo = np.zeros_like(dist)
        nuovo[0] = dist[0] * (1 - p)
        nuovo[1:indice + 1] = dist[1:indice + 1] * (1 - p) + dist[0:indice] * p
        dist = nuovo
    return dist


def riepiloga_schedina(selezioni):
    """Metriche di una schedina, a partire dalle partite selezionate.

    `selezioni` e' una lista di dict come restituiti da analizza_partita.
    Restituisce moltiplicatore, probabilita' di schedina piena, probabilita' di
    sbagliarne al massimo una, ritorno atteso e distribuzione completa degli
    esiti corretti."""
    if not selezioni:
        return None

    confidenze = [s["confidenza"] for s in selezioni]
    quote = [s["quota_pronostico"] for s in selezioni]

    moltiplicatore = float(np.prod(quote))
    p_tutte = float(np.prod(confidenze))
    distribuzione = distribuzione_numero_esatti(confidenze)
    n = len(selezioni)

    return {
        "n_partite": n,
        "moltiplicatore": moltiplicatore,
        "p_tutte": p_tutte,
        # "una su N": quante schedine servono in media per farne una piena
        "una_su": (1.0 / p_tutte) if p_tutte > 0 else float("inf"),
        "p_almeno_n_meno_1": float(distribuzione[n] + distribuzione[n - 1]) if n >= 1 else 0.0,
        # Ritorno atteso di una multipla che paga solo a schedina piena.
        "ritorno_atteso": moltiplicatore * p_tutte,
        "confidenza_media": float(np.mean(confidenze)),
        "margine_medio": float(np.mean([s["margine"] for s in selezioni])),
        "distribuzione": distribuzione,
    }


def ordina_per_confidenza(partite):
    """Partite dalla piu' sicura alla meno sicura: e' l'ordine in cui conviene
    sceglierle, perche' la selezione e' la leva principale sulla probabilita'
    di schedina piena."""
    return sorted(partite, key=lambda p: p["confidenza"], reverse=True)
