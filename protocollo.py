"""
Protocollo di misura per gli esperimenti di PredictA (Fase 0 della ROADMAP).

Perche' esiste questo modulo. Fino a Fase 3 ogni esperimento veniva giudicato
confrontando due accuratezze medie: "54.87% contro 54.17%, quindi peggio". La
verifica di significativita' ha poi mostrato che con ~1.100 partite di test
NESSUNA di quelle differenze era distinguibile dal rumore — ne' le positive ne'
le negative. Il motivo e' strutturale: fra due configurazioni cambia previsione
solo il ~2% delle partite, quindi il 98% del campione non porta informazione
sulla differenza, e l'accuratezza non ha la potenza per misurarla.

Le regole che questo modulo impone:

1. **RPS come criterio primario**, con intervallo di confidenza bootstrap
   appaiato. A differenza dell'accuratezza usa ogni partita, non solo quelle
   che cambiano previsione, e ha quindi molta piu' potenza a parita' di dati.
2. **Accuratezza come metrica descrittiva**, testata con McNemar (il test
   corretto per due classificatori sugli stessi esempi) e mai riportata come
   "meglio/peggio" senza il test.
3. **Finestra di test a 7 stagioni** (2019-2025), non 3: e' l'intera copertura
   delle quote di chiusura e piu' che raddoppia il campione senza dati nuovi.
4. **Verdetto esplicito a tre valori**: MEGLIO / PEGGIO / INDISTINGUIBILE.
   "Indistinguibile" e' un risultato legittimo e va dichiarato come tale,
   invece di essere arrotondato a "leggermente meglio".

Uso tipico:

    from protocollo import confronta, STAGIONI_TEST
    esito = confronta("nuova feature", prob_a, "baseline", prob_b, reali)
    print(esito)
"""
import numpy as np
from scipy.stats import binomtest

from modello import rps

ORDINE_CLASSI = ("1", "X", "2")

# Finestra di test standard: le 7 stagioni coperte dalle quote di chiusura.
# Il protocollo storico usava solo le ultime 3 (2023-2025), su cui pero' era
# stata fatta anche la grid search dei pesi: allargare riduce sia il rumore sia
# il rischio di leggere come segnale una fluttuazione favorevole.
STAGIONI_TEST = ["2025", "2024", "2023", "2022", "2021", "2020", "2019"]
STAGIONI_TEST_STORICHE = ["2025", "2024", "2023"]

N_BOOTSTRAP = 10000
ALFA = 0.05


def rps_per_partita(probabilita, reali):
    """RPS di ogni singola previsione. `probabilita` e' una sequenza di triple
    nell'ordine (1, X, 2); `reali` la sequenza degli esiti veri."""
    return np.array([rps({"1": p[0], "X": p[1], "2": p[2]}, r)
                     for p, r in zip(probabilita, reali)])


def predizioni_da_probabilita(probabilita):
    return np.array(ORDINE_CLASSI)[np.argmax(np.asarray(probabilita), axis=1)]


class EsitoConfronto:
    """Risultato di un confronto fra due configurazioni. `verdetto` e' il
    giudizio basato sull'RPS (criterio primario): MEGLIO / PEGGIO /
    INDISTINGUIBILE, dal punto di vista della configurazione A."""

    def __init__(self, nome_a, nome_b, n, acc_a, acc_b, solo_a, solo_b, p_mcnemar,
                 rps_a, rps_b, ic_rps, verdetto, verdetto_accuratezza):
        self.nome_a, self.nome_b, self.n = nome_a, nome_b, n
        self.acc_a, self.acc_b = acc_a, acc_b
        self.solo_a, self.solo_b, self.p_mcnemar = solo_a, solo_b, p_mcnemar
        self.rps_a, self.rps_b, self.ic_rps = rps_a, rps_b, ic_rps
        self.verdetto, self.verdetto_accuratezza = verdetto, verdetto_accuratezza

    @property
    def discordanti(self):
        return self.solo_a + self.solo_b

    def __str__(self):
        simbolo = {"MEGLIO": "✅", "PEGGIO": "❌", "INDISTINGUIBILE": "➖"}[self.verdetto]
        righe = [
            f"  {self.nome_a}  vs  {self.nome_b}   (n = {self.n} partite)",
            f"    RPS (primario):  {self.rps_a:.4f} vs {self.rps_b:.4f}   "
            f"differenza {self.rps_a - self.rps_b:+.5f}  "
            f"IC95% [{self.ic_rps[0]:+.5f}, {self.ic_rps[1]:+.5f}]",
            f"    accuratezza:     {self.acc_a:.2%} vs {self.acc_b:.2%}   "
            f"({100 * (self.acc_a - self.acc_b):+.2f} pp)  "
            f"discordanti {self.discordanti} ({self.solo_a}/{self.solo_b})  "
            f"McNemar p={self.p_mcnemar:.3f} -> {self.verdetto_accuratezza}",
            f"    VERDETTO:        {simbolo} {self.verdetto}",
        ]
        return "\n".join(righe)


def confronta(nome_a, probabilita_a, nome_b, probabilita_b, reali, rng=None):
    """Confronta due configurazioni sulle STESSE partite.

    probabilita_a / probabilita_b: sequenze di triple (p1, pX, p2).
    reali: sequenza di esiti "1"/"X"/"2".

    Il verdetto e' basato sull'RPS con bootstrap appaiato (criterio primario);
    l'accuratezza e' riportata con McNemar ma non determina il verdetto, perche'
    a questi campioni non ha potenza sufficiente."""
    reali = list(reali)
    n = len(reali)
    if not (len(probabilita_a) == len(probabilita_b) == n):
        raise ValueError("Le due configurazioni devono coprire le stesse partite")
    if n == 0:
        raise ValueError("Nessuna partita da confrontare")

    pred_a = predizioni_da_probabilita(probabilita_a)
    pred_b = predizioni_da_probabilita(probabilita_b)
    reali_arr = np.array(reali)
    ok_a, ok_b = pred_a == reali_arr, pred_b == reali_arr

    solo_a, solo_b = int((ok_a & ~ok_b).sum()), int((ok_b & ~ok_a).sum())
    discordanti = solo_a + solo_b
    p_mcnemar = binomtest(solo_a, discordanti, 0.5).pvalue if discordanti else 1.0
    if p_mcnemar >= ALFA:
        verdetto_acc = "indistinguibile"
    else:
        verdetto_acc = "meglio" if solo_a > solo_b else "peggio"

    r_a, r_b = rps_per_partita(probabilita_a, reali), rps_per_partita(probabilita_b, reali)
    differenza = r_a - r_b
    rng = rng or np.random.default_rng(12345)
    idx = rng.integers(0, n, size=(N_BOOTSTRAP, n))
    boot = differenza[idx].mean(axis=1)
    ic = tuple(np.percentile(boot, [100 * ALFA / 2, 100 * (1 - ALFA / 2)]))

    # RPS piu' basso = migliore, quindi una differenza NEGATIVA e' un miglioramento.
    if ic[0] <= 0 <= ic[1]:
        verdetto = "INDISTINGUIBILE"
    elif ic[1] < 0:
        verdetto = "MEGLIO"
    else:
        verdetto = "PEGGIO"

    return EsitoConfronto(nome_a, nome_b, n, ok_a.mean(), ok_b.mean(), solo_a, solo_b,
                          p_mcnemar, r_a.mean(), r_b.mean(), ic, verdetto, verdetto_acc)


def partite_necessarie_accuratezza(solo_a, solo_b, n_osservate):
    """Quante partite di test servirebbero perche' una differenza di accuratezza
    come quella osservata diventi significativa, mantenendone la proporzione.
    Restituisce None se la differenza e' gia' significativa o se non converge
    entro un campione plausibile."""
    discordanti = solo_a + solo_b
    if discordanti == 0:
        return None
    if binomtest(solo_a, discordanti, 0.5).pvalue < ALFA:
        return None
    quota = solo_a / discordanti
    tasso_discordanza = discordanti / n_osservate
    for nd in range(discordanti + 1, 20000):
        if binomtest(round(nd * quota), nd, 0.5).pvalue < ALFA:
            return int(round(nd / tasso_discordanza))
    return None


def partite_necessarie_rps(ic, differenza_osservata, n_osservate):
    """Analogo per l'RPS: la semiampiezza dell'IC scala come 1/sqrt(n), quindi
    per risolvere una differenza servono n * (semiampiezza / differenza)^2
    partite. Restituisce None se e' gia' risolta o se la differenza e' nulla."""
    semiampiezza = (ic[1] - ic[0]) / 2
    if differenza_osservata == 0:
        return None
    if not (ic[0] <= 0 <= ic[1]):
        return None
    return int(round(n_osservate * (semiampiezza / abs(differenza_osservata)) ** 2))


def riepiloga(esiti):
    """Stampa una tabella riassuntiva di piu' confronti, con il calcolo di quanti
    dati servirebbero per quelli rimasti indistinguibili."""
    print(f"\n{'=' * 92}")
    print(f"{'Confronto':<44} {'Δ RPS':>10} {'Δ acc':>8} {'disc.':>7} {'verdetto':>16}")
    print("-" * 92)
    for e in esiti:
        etichetta = f"{e.nome_a} vs {e.nome_b}"
        if len(etichetta) > 43:
            etichetta = etichetta[:40] + "..."
        print(f"{etichetta:<44} {e.rps_a - e.rps_b:>+10.5f} "
              f"{100 * (e.acc_a - e.acc_b):>+7.2f}p {e.discordanti:>7} {e.verdetto:>16}")
    print("=" * 92)

    indistinguibili = [e for e in esiti if e.verdetto == "INDISTINGUIBILE"]
    if indistinguibili:
        print("\nPer rendere misurabili i confronti rimasti indistinguibili servirebbero:")
        for e in indistinguibili:
            n_rps = partite_necessarie_rps(e.ic_rps, e.rps_a - e.rps_b, e.n)
            n_acc = partite_necessarie_accuratezza(e.solo_a, e.solo_b, e.n)
            def formatta(n):
                return f"{n:,} partite (~{n / 380:.0f} stagioni)" if n else "gia' misurabile o effetto nullo"
            print(f"  {e.nome_a} vs {e.nome_b}:")
            print(f"    su RPS:         {formatta(n_rps)}")
            print(f"    su accuratezza: {formatta(n_acc)}")


# ------------------------------------------------------------
# Mercati a due esiti (Over/Under, Goal/NoGoal, doppia chance)
# ------------------------------------------------------------

def brier_per_partita(prob_positivo, esiti):
    """Brier score di ogni previsione binaria. Su due soli esiti l'RPS coincide
    con il Brier, quindi resta la stessa famiglia di metrica usata per l'1X2:
    proper scoring rule, sensibile alla calibrazione e non solo alla scelta."""
    p = np.asarray([float(x) for x in prob_positivo])
    y = np.asarray([1.0 if bool(e) else 0.0 for e in esiti])
    return (p - y) ** 2


def confronta_binario(nome_a, prob_a, nome_b, prob_b, esiti, rng=None):
    """Confronto fra due previsori su un mercato a due esiti.

    prob_a / prob_b: probabilita' dell'esito "positivo" (es. Over 2.5).
    esiti: sequenza di booleani (True = esito positivo verificato).

    Stesso impianto di `confronta`: Brier con bootstrap appaiato come criterio
    primario, accuratezza con McNemar come metrica descrittiva."""
    esiti = [bool(e) for e in esiti]
    n = len(esiti)
    if not (len(prob_a) == len(prob_b) == n):
        raise ValueError("Le due configurazioni devono coprire le stesse partite")
    if n == 0:
        raise ValueError("Nessuna partita da confrontare")

    y = np.array(esiti)
    ok_a = (np.asarray([float(x) for x in prob_a]) >= 0.5) == y
    ok_b = (np.asarray([float(x) for x in prob_b]) >= 0.5) == y

    solo_a, solo_b = int((ok_a & ~ok_b).sum()), int((ok_b & ~ok_a).sum())
    discordanti = solo_a + solo_b
    p_mcnemar = binomtest(solo_a, discordanti, 0.5).pvalue if discordanti else 1.0
    if p_mcnemar >= ALFA:
        verdetto_acc = "indistinguibile"
    else:
        verdetto_acc = "meglio" if solo_a > solo_b else "peggio"

    b_a, b_b = brier_per_partita(prob_a, esiti), brier_per_partita(prob_b, esiti)
    differenza = b_a - b_b
    rng = rng or np.random.default_rng(12345)
    idx = rng.integers(0, n, size=(N_BOOTSTRAP, n))
    boot = differenza[idx].mean(axis=1)
    ic = tuple(np.percentile(boot, [100 * ALFA / 2, 100 * (1 - ALFA / 2)]))

    if ic[0] <= 0 <= ic[1]:
        verdetto = "INDISTINGUIBILE"
    elif ic[1] < 0:
        verdetto = "MEGLIO"
    else:
        verdetto = "PEGGIO"

    return EsitoConfronto(nome_a, nome_b, n, ok_a.mean(), ok_b.mean(), solo_a, solo_b,
                          p_mcnemar, b_a.mean(), b_b.mean(), ic, verdetto, verdetto_acc)
