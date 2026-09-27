"""
Costruzione e analisi di una schedina NBA a partire dalle quote: moneyline
(vittoria), spread (handicap sui punti) e totale punti (Over/Under).

A differenza dell'1X2 del calcio, questi sono tutti mercati A DUE ESITI, come
l'Over/Under 2.5 gia' gestito da schedina.py per il calcio: basta un'unica
funzione generica invece di scriverla tre volte. Niente pareggio in NBA,
quindi niente doppia chance: quel mercato esiste nel calcio solo perche' il
pareggio spezza un mercato a tre esiti in tre coperture parziali.

Le metriche di schedina (moltiplicatore, probabilita' piena, distribuzione
degli esiti corretti) sono generiche e gia' in schedina.py: qui si riusano
sc.riepiloga_schedina / sc.ordina_per_confidenza senza duplicarle.
"""
from modello import probabilita_shin

MONEYLINE = "Moneyline"
SPREAD = "Spread"
TOTALE = "Totale punti"


def analizza_mercato_due_vie(mercato, esito_1, esito_2, quota_1, quota_2,
                             prob_1_modello=None, peso_quote=1.0):
    """Pronostico su un mercato a due esiti.

    quota_1 / quota_2: quote decimali sui due esiti, in questo ordine.
    prob_1_modello: probabilita' del primo esito secondo il modello Elo (vedi
        pronostico_nba.py). Se fornita, viene fusa con la probabilita' di
        mercato (ripulita dal margine con Shin) secondo peso_quote — stessa
        logica del blend usato per il calcio in schedina.analizza_over_under.
    """
    quote = [float(quota_1), float(quota_2)]
    if any(q <= 1.0 for q in quote):
        raise ValueError("Le quote decimali devono essere maggiori di 1")

    p_mercato = probabilita_shin(quote)
    p_1 = p_mercato[0]
    if prob_1_modello is not None:
        p_1 = (1 - peso_quote) * float(prob_1_modello) + peso_quote * p_mercato[0]
    p_2 = 1.0 - p_1

    esiti = (esito_1, esito_2)
    probabilita = (p_1, p_2)
    i = 0 if p_1 >= p_2 else 1
    return {
        "prob_1": p_1, "prob_2": p_2,
        "pronostico": esiti[i],
        "confidenza": probabilita[i],
        "quota_pronostico": quote[i],
        "margine": sum(1.0 / q for q in quote) - 1.0,
        "mercato": mercato,
        "prob_1_mercato": p_mercato[0],
    }
