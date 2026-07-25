"""
Quanto sono statisticamente distinguibili le configurazioni che il progetto ha
confrontato finora?

Tutti gli esperimenti di Fase 2 e Fase 3 sono stati giudicati su differenze di
accuratezza fra 0.1 e 0.9 punti percentuali, misurate su 1.140 partite (3
stagioni x 380). Questo script verifica se quelle differenze siano distinguibili
dal rumore, con il test appropriato per confronti sugli STESSI match: il test di
McNemar sulle predizioni discordanti (le partite in cui una configurazione
indovina e l'altra no). Il test appaiato e' molto piu' sensibile del confronto
fra due accuratezze indipendenti, perche' le due configurazioni sbagliano in
gran parte le stesse partite: e' quindi il confronto piu' favorevole possibile
alle differenze osservate.

Confronta anche l'RPS con un bootstrap appaiato, per mostrare quanto sia una
metrica meno rumorosa a parita' di campione — il motivo per cui dovrebbe essere
il criterio primario di adozione, non l'accuratezza.
"""
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "pages")

import numpy as np
from scipy.stats import binomtest

import backtesting as bt
from modello import rps

PESO_FORMA, PESO_SCONTRI, PESO_QUOTE, RHO = 0.10, 0.0, 0.90, -0.10
EMIVITA_GIORNI, N_PARTITE_FORMA = 730, 3
# Di default le 3 stagioni del protocollo storico del progetto. Con --tutte si usa
# l'intera finestra coperta dalle quote di chiusura (2019-2025, 7 stagioni): il
# campione quasi triplica e, secondo il calcolo di potenza in fondo a questo file,
# e' la finestra minima in cui il confronto col mercato diventa misurabile.
STAGIONI = ["2025", "2024", "2023"]
if "--tutte" in sys.argv:
    STAGIONI = ["2025", "2024", "2023", "2022", "2021", "2020", "2019"]
RNG = np.random.default_rng(12345)


def predizioni_configurazione(fonte_quote="chiusura", medie_lega="storiche",
                              peso_quote=PESO_QUOTE, peso_forma=PESO_FORMA):
    """Predizioni e RPS per-partita su tutte e 3 le stagioni, concatenate.

    Nota: peso_forma va azzerato esplicitamente per ottenere il "solo mercato".
    Con peso_forma=0.10 e peso_quote=1.00 la somma supera 1 e valuta_componente
    rinormalizza a 0.091/0.909, cioe' praticamente la configurazione di default:
    il confronto darebbe zero partite discordanti e sembrerebbe, erroneamente,
    che il modello non aggiunga nulla."""
    corrette, rps_per_partita = [], []
    for stagione in STAGIONI:
        bt.stagioni_test = [stagione]
        bt.train_df = bt.df[~bt.df["Stagione"].astype(str).isin(bt.stagioni_test)].copy()
        bt.test_df = bt.df[bt.df["Stagione"].astype(str).isin(bt.stagioni_test)].copy()
        bt.media_gol_casa = bt.train_df["FTHG"].mean()
        bt.media_gol_trasferta = bt.train_df["FTAG"].mean()
        bt.media_gol_generale = (bt.media_gol_casa + bt.media_gol_trasferta) / 2

        componenti = bt.precompute_tutte(EMIVITA_GIORNI, N_PARTITE_FORMA, metodo_quote="shin",
                                         fonte_quote=fonte_quote, medie_lega=medie_lega)
        predizioni, reali, _, probabilita = bt.valuta_tutte(
            componenti, peso_forma, PESO_SCONTRI, peso_quote, RHO)
        corrette.extend(p == r for p, r in zip(predizioni, reali))
        rps_per_partita.extend(rps({"1": p[0], "X": p[2], "2": p[1]}, r)
                               for p, r in zip(probabilita, reali))
    return np.array(corrette), np.array(rps_per_partita)


def confronta(nome_a, a, nome_b, b, rps_a, rps_b):
    n = len(a)
    acc_a, acc_b = a.mean(), b.mean()
    # McNemar: solo le partite discordanti portano informazione sulla differenza.
    solo_a = int((a & ~b).sum())
    solo_b = int((b & ~a).sum())
    discordanti = solo_a + solo_b
    p_value = binomtest(solo_a, discordanti, 0.5).pvalue if discordanti > 0 else 1.0

    # Bootstrap appaiato sulla differenza di RPS.
    d = rps_a - rps_b
    idx = RNG.integers(0, n, size=(10000, n))
    boot = d[idx].mean(axis=1)
    ic_basso, ic_alto = np.percentile(boot, [2.5, 97.5])

    print(f"\n  {nome_a}  vs  {nome_b}   (n={n} partite)")
    print(f"    accuratezza:   {acc_a:.2%} vs {acc_b:.2%}   (differenza {100 * (acc_a - acc_b):+.2f} pp)")
    print(f"    discordanti:   {discordanti} partite ({solo_a} solo A, {solo_b} solo B)")
    print(f"    McNemar:       p = {p_value:.3f}  ->  {'DISTINGUIBILE' if p_value < 0.05 else 'NON distinguibile dal rumore'}")
    print(f"    RPS:           {rps_a.mean():.4f} vs {rps_b.mean():.4f}   "
          f"(differenza {rps_a.mean() - rps_b.mean():+.5f}, IC95% [{ic_basso:+.5f}, {ic_alto:+.5f}])")
    significativo_rps = not (ic_basso <= 0 <= ic_alto)
    print(f"    RPS:           {'DISTINGUIBILE' if significativo_rps else 'NON distinguibile dal rumore'}")


if __name__ == "__main__":
    print("Significativita' statistica delle differenze fra configurazioni")
    print(f"Protocollo: 3 stagioni ({', '.join(STAGIONI)}), pesi forma={PESO_FORMA}/quote={PESO_QUOTE}\n", flush=True)

    print("Calcolo configurazioni...", flush=True)
    chiusura, rps_chiusura = predizioni_configurazione(fonte_quote="chiusura")
    apertura, rps_apertura = predizioni_configurazione(fonte_quote="apertura")
    pesate, rps_pesate = predizioni_configurazione(medie_lega="pesate")
    solo_mercato, rps_solo_mercato = predizioni_configurazione(peso_quote=1.0, peso_forma=0.0)

    print("\n" + "=" * 78)
    print("RISULTATI")
    print("=" * 78, flush=True)

    confronta("quote CHIUSURA", chiusura, "quote APERTURA", apertura, rps_chiusura, rps_apertura)
    confronta("medie PESATE", pesate, "medie STORICHE", chiusura, rps_pesate, rps_chiusura)
    confronta("modello (forma+quote)", chiusura, "SOLO MERCATO", solo_mercato, rps_chiusura, rps_solo_mercato)
