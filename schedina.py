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

3. **Il mercato scelto e' un vincolo sulla lunghezza della schedina, prima che
   sulla precisione.** L'1X2 ha tre caselle e il pareggio se ne prende
   stabilmente un quarto: la confidenza mediana del favorito e' 51.4% e solo il
   30% delle partite arriva al 60% (2.660 partite di Serie A, quote di
   chiusura). Chiedere una schedina lunga fatta di soli esiti 1X2 sicuri e'
   chiedere qualcosa che il mercato non contiene — su 266 giornate simulate una
   da 10 partite non e' mai uscita piena. La doppia chance, ricavabile dalle
   stesse tre quote senza dati nuovi, sposta la mediana a 78.2% e rende
   giocabile la giornata intera, al prezzo di un moltiplicatore molto piu'
   basso. Misure in valida_doppia_chance.py.

Il ritorno atteso calcolato qui e' quindi sempre < 1. Il modulo serve a scegliere
consapevolmente il punto sulla curva rischio/premio e a sapere in anticipo quanto
e' improbabile una schedina piena, non a promettere un guadagno.
"""
import numpy as np

from modello import probabilita_shin

ESITI = ("1", "X", "2")
ESITI_OU = ("Over 2.5", "Under 2.5")
# Doppia chance: nome dell'esito e indici degli esiti 1X2 che lo fanno vincere.
DOPPIE_CHANCE = (("1X", (0, 1)), ("12", (0, 2)), ("X2", (1, 2)))


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
        "mercato": "1X2",
    }


def analizza_doppia_chance(quota_1, quota_x, quota_2, probabilita=None):
    """Il miglior esito di doppia chance (1X, 12, X2) per una partita.

    Non serve nessuna quota in piu' rispetto all'1X2. Dividendo la posta fra i
    due esiti coperti in proporzione a 1/q, il pagamento e' lo stesso quale che
    sia dei due a uscire e vale

        q_dc = 1 / (1/q_a + 1/q_b)

    Non e' un'approssimazione della quota che offrirebbe un bookmaker: e' la
    quota di una doppia chance costruita davvero con due giocate sull'1X2, e
    porta esattamente lo stesso margine delle due quote di partenza.

    `probabilita`: le tre probabilita' 1X2 gia' calcolate (ad es. il blend con
    il modello statistico). Se assente si usano quelle di Shin dalle quote.

    Perche' esiste questo mercato nello strumento: su 2.660 partite di Serie A
    la confidenza mediana del miglior esito 1X2 e' 51.4%, e solo il 30% delle
    partite arriva al 60%. Una schedina lunga in 1X2 e' quindi fuori portata —
    su 266 giornate simulate una da 10 partite non e' mai uscita piena — mentre
    in doppia chance la stessa giornata e' uscita piena 30 volte. Il prezzo e'
    un moltiplicatore molto piu' basso: vedi valida_doppia_chance.py."""
    quote = [float(quota_1), float(quota_x), float(quota_2)]
    if any(q <= 1.0 for q in quote):
        raise ValueError("Le quote decimali devono essere maggiori di 1")

    p = list(probabilita) if probabilita is not None else probabilita_shin(quote)

    candidati = []
    for nome, indici in DOPPIE_CHANCE:
        q_dc = 1.0 / sum(1.0 / quote[i] for i in indici)
        p_dc = sum(p[i] for i in indici)
        candidati.append({
            "pronostico": nome,
            "confidenza": p_dc,
            "quota_pronostico": q_dc,
            # Stessa definizione di margine usata in analizza_partita
            # (probabilita' implicita / probabilita' vera - 1), qui ristretta
            # ai due esiti coperti: sull'intero mercato 1X2 le probabilita'
            # vere sommano a 1 e la formula si riduce a sum(1/q) - 1.
            "margine": sum(1.0 / quote[i] for i in indici) / p_dc - 1.0,
            "mercato": "Doppia chance",
            "esiti_coperti": tuple(ESITI[i] for i in indici),
        })
    return max(candidati, key=lambda c: c["confidenza"])


def analizza_over_under(quota_over, quota_under, prob_over_modello=None, peso_quote=1.0):
    """Pronostico Over/Under 2.5 gol per una partita.

    quota_over / quota_under: quote decimali offerte sui due esiti.
    prob_over_modello: probabilita' di Over secondo il modello statistico
        (dalla matrice Poisson-Dixon-Coles). Se fornita, viene fusa con il
        mercato secondo peso_quote, con la stessa logica del blend 1X2.

    Restituisce un dict nello stesso formato di analizza_partita, cosi' che una
    schedina possa mescolare liberamente pronostici 1X2 e Over/Under."""
    quote = [float(quota_over), float(quota_under)]
    if any(q <= 1.0 for q in quote):
        raise ValueError("Le quote decimali devono essere maggiori di 1")

    p_mercato = probabilita_shin(quote)
    p_over = p_mercato[0]
    if prob_over_modello is not None:
        p_over = (1 - peso_quote) * float(prob_over_modello) + peso_quote * p_mercato[0]
    p_under = 1.0 - p_over

    i = 0 if p_over >= p_under else 1
    return {
        "prob_over": p_over, "prob_under": p_under,
        "pronostico": ESITI_OU[i],
        "confidenza": (p_over, p_under)[i],
        "quota_pronostico": quote[i],
        "margine": sum(1.0 / q for q in quote) - 1.0,
        "mercato": "Over/Under 2.5",
        "prob_over_mercato": p_mercato[0],
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
