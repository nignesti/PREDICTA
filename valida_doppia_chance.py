"""
Perche' il costruttore di schedina ne seleziona 3 su 10, e la doppia chance.

Il problema, riportato su una giornata reale di Serie A: si inseriscono 10
partite, la soglia di confidenza e' al default (60%) e la schedina che esce ha
3-4 partite. Non e' un bug: e' una conseguenza aritmetica di come e' fatto il
mercato 1X2, e questo script la misura.

Un mercato a tre esiti distribuisce la probabilita' su tre caselle, e il
pareggio ne prende stabilmente il 25-27%. Il tetto pratico dell'esito piu'
probabile e' quindi molto piu' basso di quanto suggerisca l'intuizione: nella
maggioranza delle partite di Serie A il favorito sta sotto il 55%. Una soglia al
60% non e' "prudente", e' fuori scala rispetto al mercato su cui e' applicata, e
taglia strutturalmente 3 partite su 4.

Il modulo mette alla prova una risposta che non richiede nessun dato nuovo: la
**doppia chance** (1X, 12, X2), ricavabile dalle stesse tre quote gia' inserite.
Puntando una quota s divisa in proporzione a 1/q su due dei tre esiti, il
pagamento e' identico qualunque dei due esca e vale

    q_dc = 1 / (1/q_a + 1/q_b)

Non e' un'approssimazione: e' la quota di una doppia chance costruita davvero
con due giocate sullo stesso mercato 1X2. La probabilita' corrispondente e' la
somma delle due probabilita' di Shin gia' calcolate.

Le domande misurate qui, e cosa e' venuto fuori su 2.660 partite di Serie A con
quote di chiusura (2019-2026):

1. **Quante partite superano la soglia?** La confidenza mediana del miglior
   esito 1X2 e' 51.4%, e solo il 30.1% delle partite arriva al 60%: su una
   giornata da 10 ne passano 3.0 in media. Il "ne seleziona 3" non e' un caso
   sfortunato, e' il valore atteso. Con la doppia chance la mediana sale a
   78.2% e al 60% passano tutte.
2. **La doppia chance e' calibrata?** Si': Brier 0.1905, identico all'1X2, e
   scarti fra probabilita' dichiarata e frequenza osservata entro 3.4 punti su
   tutte le fasce. Conta perche' altrimenti la "probabilita' di schedina piena"
   mostrata in pagina sarebbe una bugia.
3. **Quanto costa una giocata di doppia chance?** Quanto una qualsiasi altra:
   il ritorno realizzato per singola giocata sta fra il 92% e il 98% su tutti e
   sei i tipi di esito, differenze dentro il rumore. Questa e' la correzione a
   un'ipotesi iniziale sbagliata di questo script: sembrava che la doppia
   chance dovesse costare meno, evitando via favorite-longshot bias la casella
   piu' caricata di margine. Misurato, il vantaggio non c'e'. **La doppia
   chance non e' piu' economica: e' solo piu' probabile.**
4. **A parita' di numero di partite, conviene?** Qui la risposta si ribalta a
   seconda della lunghezza della schedina, ed e' il risultato che decide:

       schedina da  3   1X2: piena nel 32% dei casi   doppia chance: 68%
       schedina da 10   1X2: piena nello 0.2%         doppia chance: 9.7%

   Su 266 giornate simulate, una schedina da 10 in 1X2 non e' **mai** uscita
   piena (0 volte, 0.5 attese); la stessa giornata giocata in doppia chance e'
   uscita piena 30 volte. Sotto le 5 partite invece l'1X2 resta preferibile: il
   moltiplicatore e' molto piu' alto e il ritorno realizzato pure (96.1% contro
   91.2% sulla schedina da 3).

Il ritorno atteso resta comunque sotto il 100% ovunque, e cala man mano che si
aggiungono partite: ogni gamba paga il suo margine, e il prodotto peggiora.
Cambiare mercato sposta il punto sulla curva rischio/premio, non la curva.

Protocollo coerente col resto del progetto: solo partite con quote di chiusura
(dal 2019), Shin per togliere il margine, nessun dato futuro nella selezione
(la selezione usa solo le quote della partita stessa).
"""
import sys

import numpy as np
import pandas as pd

from modello import probabilita_shin

PARTITE_PER_GIORNATA = 10
SOGLIE = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)

# Doppia chance: (nome, indici degli esiti 1X2 coperti)
DOPPIE = (("1X", (0, 1)), ("12", (0, 2)), ("X2", (1, 2)))
ESITI = ("1", "X", "2")
# serie_a.csv codifica l'esito come H/D/A: va tradotto prima di confrontarlo
# con i pronostici, altrimenti ogni giocata risulta persa e i conteggi tornano
# tutti a zero senza che nulla segnali l'errore.
ESITO_DA_FTR = {"H": "1", "D": "X", "A": "2"}


def carica():
    """Partite di Serie A con quote di chiusura complete (dal 2019)."""
    df = pd.read_csv("serie_a.csv")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["OddsAvgCH", "OddsAvgCD", "OddsAvgCA", "FTR", "Date"])
    df["esito"] = df["FTR"].map(ESITO_DA_FTR)
    df = df.dropna(subset=["esito"])
    return df.sort_values("Date").reset_index(drop=True)


def quota_doppia(quote, indici):
    """Quota realizzabile di una doppia chance, dalle sole quote 1X2.

    Dividendo la posta in proporzione a 1/q fra i due esiti coperti, il ritorno
    e' lo stesso quale che sia dei due a uscire, e vale 1/(1/q_a + 1/q_b)."""
    return 1.0 / sum(1.0 / quote[i] for i in indici)


def candidati_partita(quote, con_doppia):
    """Tutte le giocate disponibili su una partita, come dict confrontabili.

    Ogni candidato ha probabilita' (Shin), quota e l'insieme di esiti 1X2 che lo
    fanno vincere — abbastanza per verificarlo a posteriori contro il risultato."""
    p = probabilita_shin(quote)
    fuori = [{"nome": ESITI[i], "mercato": "1X2", "p": p[i],
              "quota": quote[i], "vincenti": {ESITI[i]}} for i in range(3)]
    if not con_doppia:
        return fuori
    for nome, indici in DOPPIE:
        fuori.append({"nome": nome, "mercato": "Doppia chance",
                      "p": sum(p[i] for i in indici),
                      "quota": quota_doppia(quote, indici),
                      "vincenti": {ESITI[i] for i in indici}})
    return fuori


def migliore(quote, con_doppia):
    """La giocata piu' probabile su una partita, fra i mercati ammessi."""
    return max(candidati_partita(quote, con_doppia), key=lambda c: c["p"])


def migliore_sopra_soglia(quote, soglia):
    """La giocata con la quota piu' alta fra quelle che superano la soglia.

    Regola alternativa a "prendi sempre la piu' probabile". Quella, ammessa la
    doppia chance, sceglie la doppia chance su *ogni* partita — copre due esiti
    su tre, quindi vince sempre il confronto sulla confidenza — e su un
    Inter-Monza da 1.22 finisce a giocare 1X a quota 1.01: una gamba che puo'
    solo far perdere e non paga nulla. Sembra uno spreco evidente. Se la soglia
    e' gia' superata dall'1X2, perche' non tenersi la quota piu' alta?

    Misurato, **non e' uno spreco: e' un punto diverso sulla stessa curva.** Le
    due regole hanno lo stesso ritorno atteso a ogni soglia (a soglia 60%:
    66.5% contro 65.6%, differenza dentro il rumore). Cambia solo la forma del
    rischio: a soglia 60% su 266 giornate "max confidenza" ha prodotto 30
    schedine piene con moltiplicatore mediano 7x, "max quota" ne ha prodotte 6
    con moltiplicatore mediano 24.5x. Nessuna delle due domina l'altra, ed e'
    la conferma piu' netta della tesi in cima a schedina.py: la selezione
    sposta il punto sulla curva rischio/premio, non la curva.

    Se nessuna giocata supera la soglia si restituisce comunque la piu'
    probabile, cosi' che la partita compaia fra le escluse invece di sparire."""
    candidati = candidati_partita(quote, True)
    ammessi = [c for c in candidati if c["p"] >= soglia]
    if not ammessi:
        return max(candidati, key=lambda c: c["p"])
    return max(ammessi, key=lambda c: c["quota"])


def quote_riga(riga):
    return [float(riga["OddsAvgCH"]), float(riga["OddsAvgCD"]), float(riga["OddsAvgCA"])]


# ------------------------------------------------------------
# 1. Quante partite superano la soglia
# ------------------------------------------------------------
def copertura(df):
    print("\n1. QUANTE PARTITE SUPERANO LA SOGLIA")
    print(f"   {len(df):,} partite di Serie A con quote di chiusura "
          f"({df['Date'].min():%Y-%m} → {df['Date'].max():%Y-%m})\n")

    conf_1x2, conf_dc = [], []
    for _, r in df.iterrows():
        q = quote_riga(r)
        conf_1x2.append(migliore(q, False)["p"])
        conf_dc.append(migliore(q, True)["p"])
    conf_1x2, conf_dc = np.array(conf_1x2), np.array(conf_dc)

    print(f"   Confidenza del miglior esito 1X2: mediana {np.median(conf_1x2):.1%}, "
          f"90° percentile {np.percentile(conf_1x2, 90):.1%}, max {conf_1x2.max():.1%}")
    print(f"   Con la doppia chance:             mediana {np.median(conf_dc):.1%}, "
          f"90° percentile {np.percentile(conf_dc, 90):.1%}, max {conf_dc.max():.1%}\n")

    print(f"   {'soglia':>7} {'solo 1X2':>10} {'+ doppia ch.':>13}   partite ammesse su una giornata da 10")
    for s in SOGLIE:
        a, b = (conf_1x2 >= s).mean(), (conf_dc >= s).mean()
        print(f"   {s:>6.0%} {a:>10.1%} {b:>13.1%}   {a*10:>4.1f}  →  {b*10:.1f}")
    return conf_1x2, conf_dc


# ------------------------------------------------------------
# 2. Calibrazione
# ------------------------------------------------------------
def calibrazione(df):
    """La probabilita' dichiarata corrisponde alla frequenza osservata?

    Se la doppia chance fosse mal calibrata, tutte le metriche di schedina
    (probabilita' piena, "1 su N") sarebbero ottimistiche per costruzione."""
    print("\n2. CALIBRAZIONE DELLA DOPPIA CHANCE")
    print("   Una giocata data al 72% deve uscire nel 72% dei casi.\n")

    righe = []
    for _, r in df.iterrows():
        q = quote_riga(r)
        for c in candidati_partita(q, True):
            righe.append((c["mercato"], c["p"], r["esito"] in c["vincenti"]))
    d = pd.DataFrame(righe, columns=["mercato", "p", "vinta"])

    for mercato in ("1X2", "Doppia chance"):
        m = d[d["mercato"] == mercato]
        print(f"   {mercato}  ({len(m):,} giocate valutate)")
        print(f"   {'fascia':>14} {'n':>7} {'prevista':>10} {'osservata':>11} {'scarto':>9}")
        bordi = np.arange(0.0, 1.01, 0.10)
        for lo, hi in zip(bordi[:-1], bordi[1:]):
            f = m[(m["p"] >= lo) & (m["p"] < hi)]
            if len(f) < 50:
                continue
            prev, oss = f["p"].mean(), f["vinta"].mean()
            print(f"   {lo:>5.0%}-{hi:<7.0%} {len(f):>7,} {prev:>10.1%} "
                  f"{oss:>11.1%} {(oss - prev) * 100:>+7.1f}pt")
        brier = float(np.mean((m["p"] - m["vinta"]) ** 2))
        print(f"   Brier medio: {brier:.4f}\n")


# ------------------------------------------------------------
# 3. Effetto su una schedina reale, giornata per giornata
# ------------------------------------------------------------
def giornate(df):
    """Spezza lo storico in giornate da 10 partite, cronologicamente."""
    fuori = []
    for _, gruppo in df.groupby("Stagione", sort=False):
        gruppo = gruppo.sort_values("Date")
        for i in range(0, len(gruppo) - PARTITE_PER_GIORNATA + 1, PARTITE_PER_GIORNATA):
            fuori.append(gruppo.iloc[i:i + PARTITE_PER_GIORNATA])
    return fuori


def simula(df, soglia, strategia, max_partite=13):
    """Gioca una schedina per giornata e conta cosa succede davvero.

    `strategia`: "1x2" (solo esiti singoli), "dc" (doppia chance ammessa, si
    prende sempre la giocata piu' probabile) o "misto" (doppia chance ammessa,
    ma fra le giocate sopra soglia si prende quella che paga di piu')."""
    n_giornate = n_vuote = piene = 0
    partite_per_schedina, esiti_corretti, moltiplicatori = [], [], []
    ritorno_reale = ritorno_atteso = 0.0

    for g in giornate(df):
        n_giornate += 1
        scelte = []
        for _, r in g.iterrows():
            if strategia == "misto":
                c = migliore_sopra_soglia(quote_riga(r), soglia)
            else:
                c = migliore(quote_riga(r), strategia == "dc")
            c["esito_reale"] = r["esito"]
            scelte.append(c)
        scelte.sort(key=lambda c: c["p"], reverse=True)
        sel = [c for c in scelte if c["p"] >= soglia][:max_partite]
        if not sel:
            n_vuote += 1
            continue

        molt = float(np.prod([c["quota"] for c in sel]))
        giuste = sum(c["esito_reale"] in c["vincenti"] for c in sel)
        partite_per_schedina.append(len(sel))
        esiti_corretti.append(giuste)
        moltiplicatori.append(molt)
        # Una multipla paga solo a schedina piena: 1 euro giocato per giornata.
        vinta = giuste == len(sel)
        piene += vinta
        ritorno_reale += molt if vinta else 0.0
        ritorno_atteso += molt * float(np.prod([c["p"] for c in sel]))

    giocate = n_giornate - n_vuote
    if giocate == 0:
        return {"giornate": n_giornate, "giocabili": 0}
    return {
        "giornate": n_giornate,
        "giocabili": giocate,
        "vuote": n_vuote,
        "partite_medie": float(np.mean(partite_per_schedina)),
        "moltiplicatore_mediano": float(np.median(moltiplicatori)),
        "piene": piene,
        "frazione_esatti": float(np.mean([c / n for c, n in
                                          zip(esiti_corretti, partite_per_schedina)])),
        # Un euro per giornata giocata, incassato solo a schedina piena.
        "ritorno_reale": ritorno_reale / giocate,
        "ritorno_atteso": ritorno_atteso / giocate,
    }


def confronto_schedine(df):
    print("\n3. EFFETTO SU UNA SCHEDINA REALE, GIORNATA PER GIORNATA")
    print(f"   Giornate da {PARTITE_PER_GIORNATA} partite, 1 euro giocato per giornata, "
          "incasso solo a schedina piena.\n")
    print(f"   {'soglia':>7} {'mercati':>16} {'giornate':>9} {'no-bet':>7} "
          f"{'part.':>6} {'molt.med':>9} {'piene':>6} {'%esatti':>8} "
          f"{'ritorno':>8} {'atteso':>7}")
    for s in SOGLIE:
        for strategia, etichetta in (("1x2", "solo 1X2"),
                                     ("dc", "+ dc, max conf."),
                                     ("misto", "+ dc, max quota")):
            r = simula(df, s, strategia)
            if not r["giocabili"]:
                print(f"   {s:>6.0%} {etichetta:>16} {r['giornate']:>9}   nessuna schedina")
                continue
            print(f"   {s:>6.0%} {etichetta:>16} {r['giocabili']:>9} {r['vuote']:>7} "
                  f"{r['partite_medie']:>6.1f} {r['moltiplicatore_mediano']:>9.1f} "
                  f"{r['piene']:>6} {r['frazione_esatti']:>8.1%} "
                  f"{r['ritorno_reale']:>8.1%} {r['ritorno_atteso']:>7.1%}")
        print()


# ------------------------------------------------------------
# 4. Dove finisce il margine
# ------------------------------------------------------------
def margine_per_mercato(df):
    """Quanto costa una singola giocata, per tipo di esito.

    Due colonne, e servono entrambe perche' misurano cose diverse:

    - **implicito** e' quota × probabilita' di Shin. NON e' una misura empirica:
      la probabilita' di Shin viene ricavata dalle stesse quote, quindi questa
      colonna descrive solo *come Shin ripartisce il margine* fra gli esiti. E'
      utile per vedere la favorite-longshot bias, non per giudicare il mercato.
    - **realizzato** e' quota media × frequenza osservata dell'esito: quanto ha
      davvero reso un euro puntato su quel tipo di giocata su tutto lo storico.
      Questa e' la misura vera, e va letta con la sua incertezza (una manciata
      di punti percentuali su 2.660 partite)."""
    print("\n4. QUANTO COSTA UNA SINGOLA GIOCATA")
    print("   100% sarebbe un mercato equo; sotto 100% e' il margine del bookmaker.\n")
    implicito, realizzato = {}, {}
    for _, r in df.iterrows():
        q = quote_riga(r)
        for c in candidati_partita(q, True):
            implicito.setdefault(c["nome"], []).append(c["quota"] * c["p"])
            realizzato.setdefault(c["nome"], []).append(
                c["quota"] if r["esito"] in c["vincenti"] else 0.0)
    print(f"   {'giocata':>10} {'implicito (Shin)':>18} {'realizzato':>12} {'±2σ':>8}")
    for nome in ("1", "X", "2", "1X", "12", "X2"):
        imp = np.array(implicito[nome])
        rea = np.array(realizzato[nome])
        errore = 2 * rea.std(ddof=1) / np.sqrt(len(rea))
        print(f"   {nome:>10} {imp.mean():>17.1%} {rea.mean():>12.1%} {errore:>7.1%}")
    print("\n   Il costo per giocata e' sostanzialmente lo stesso su tutti i mercati:")
    print("   la doppia chance non e' piu' economica dell'1X2, e' solo piu' probabile.")


# ------------------------------------------------------------
# 5. A parita' di numero di partite: il confronto che decide
# ------------------------------------------------------------
def confronto_a_parita_di_partite(df, taglie=(3, 5, 8, 10)):
    """La domanda vera dell'utente non e' "quale soglia", e' "voglio una
    schedina da N partite: come le scelgo?".

    Il confronto per soglia (sezione 3) e' sleale: a parita' di soglia i due
    mercati producono schedine di lunghezza diversa, quindi confronta un 3 con
    un 10. Qui si fissa N e si prendono le N giocate piu' probabili della
    giornata, con e senza doppia chance."""
    print("\n5. A PARITA' DI NUMERO DI PARTITE")
    print("   Le N giocate piu' probabili di ogni giornata, 1 euro per giornata.\n")
    print(f"   {'N':>3} {'mercati':>16} {'conf.media':>11} {'molt.med':>9} "
          f"{'p_piena':>8} {'piene oss.':>11} {'attese':>7} {'ritorno':>8}")
    for n in taglie:
        for con_dc, etichetta in ((False, "solo 1X2"), (True, "+ doppia chance")):
            confidenze, molt, piene, attese, incasso, giocate = [], [], 0, 0.0, 0.0, 0
            for g in giornate(df):
                scelte = sorted((dict(migliore(quote_riga(r), con_dc), esito_reale=r["esito"])
                                 for _, r in g.iterrows()),
                                key=lambda c: c["p"], reverse=True)[:n]
                if len(scelte) < n:
                    continue
                giocate += 1
                m = float(np.prod([c["quota"] for c in scelte]))
                p_piena = float(np.prod([c["p"] for c in scelte]))
                confidenze.append(float(np.mean([c["p"] for c in scelte])))
                molt.append(m)
                attese += p_piena
                if all(c["esito_reale"] in c["vincenti"] for c in scelte):
                    piene += 1
                    incasso += m
            print(f"   {n:>3} {etichetta:>16} {np.mean(confidenze):>11.1%} "
                  f"{np.median(molt):>9.1f} {attese / giocate:>8.1%} "
                  f"{piene:>4} / {giocate:<4} {attese:>7.1f} {incasso / giocate:>8.1%}")
        print()


if __name__ == "__main__":
    df = carica()
    if "--veloce" in sys.argv:
        df = df.tail(1000)
    copertura(df)
    calibrazione(df)
    margine_per_mercato(df)
    confronto_schedine(df)
    confronto_a_parita_di_partite(df)
