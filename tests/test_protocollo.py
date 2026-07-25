"""
Test del protocollo di misura (Fase 0).

Il modulo serve a evitare che una differenza casuale venga letta come un
miglioramento: i test verificano quindi entrambi i lati dell'errore, cioe' che
riconosca un effetto vero E che dichiari indistinguibile il rumore.
"""
import numpy as np
import pytest

import protocollo


def _probabilita_da_verita(reali, forza, rng):
    """Genera previsioni che assegnano probabilita' `forza` all'esito vero e si
    dividono il resto: piu' `forza` e' alta, migliore e' il previsore."""
    resto = (1 - forza) / 2
    ordine = list(protocollo.ORDINE_CLASSI)
    prob = []
    for r in reali:
        p = [resto] * 3
        p[ordine.index(r)] = forza
        # Un po' di rumore per non avere previsioni identiche fra loro.
        rumore = rng.normal(0, 0.01, 3)
        p = np.clip(np.array(p) + rumore, 1e-6, None)
        prob.append(p / p.sum())
    return prob


@pytest.fixture
def reali():
    rng = np.random.default_rng(0)
    return list(rng.choice(list(protocollo.ORDINE_CLASSI), size=1200, p=[0.42, 0.26, 0.32]))


def test_riconosce_un_previsore_nettamente_migliore(reali):
    rng = np.random.default_rng(1)
    forte = _probabilita_da_verita(reali, 0.70, rng)
    debole = _probabilita_da_verita(reali, 0.40, rng)

    esito = protocollo.confronta("forte", forte, "debole", debole, reali)

    assert esito.verdetto == "MEGLIO"
    assert esito.rps_a < esito.rps_b
    # L'IC deve stare interamente sotto lo zero (RPS piu' basso = meglio).
    assert esito.ic_rps[1] < 0


def test_dichiara_indistinguibili_due_previsori_equivalenti(reali):
    # Stessa qualita', solo rumore diverso: il protocollo NON deve dichiarare
    # un vincitore. E' l'errore che questo modulo esiste per prevenire.
    rng = np.random.default_rng(2)
    a = _probabilita_da_verita(reali, 0.55, rng)
    b = _probabilita_da_verita(reali, 0.55, rng)

    esito = protocollo.confronta("a", a, "b", b, reali)

    assert esito.verdetto == "INDISTINGUIBILE"
    assert esito.ic_rps[0] <= 0 <= esito.ic_rps[1]


def test_verdetto_peggio_e_simmetrico_rispetto_a_meglio(reali):
    rng = np.random.default_rng(3)
    forte = _probabilita_da_verita(reali, 0.70, rng)
    debole = _probabilita_da_verita(reali, 0.40, rng)

    assert protocollo.confronta("d", debole, "f", forte, reali).verdetto == "PEGGIO"
    assert protocollo.confronta("f", forte, "d", debole, reali).verdetto == "MEGLIO"


def test_mcnemar_conta_solo_le_partite_discordanti():
    # Due previsori che sbagliano/indovinano esattamente le stesse partite non
    # hanno discordanti: nessuna evidenza di differenza, p = 1.
    reali = ["1", "X", "2"] * 100
    rng = np.random.default_rng(4)
    identico = _probabilita_da_verita(reali, 0.60, rng)

    esito = protocollo.confronta("a", identico, "b", identico, reali)

    assert esito.discordanti == 0
    assert esito.p_mcnemar == 1.0
    assert esito.verdetto_accuratezza == "indistinguibile"


def test_accuratezza_non_determina_il_verdetto():
    # Caso costruito: A indovina qualche partita in piu' di B (accuratezza
    # migliore) ma e' molto peggio calibrato, quindi ha RPS peggiore. Il
    # verdetto segue l'RPS, che e' il criterio primario dichiarato.
    reali = ["1"] * 500 + ["2"] * 500
    # A: sicurissimo e spesso sbagliato. B: prudente e ben calibrato.
    a = [[0.98, 0.01, 0.01] if r == "1" else [0.98, 0.01, 0.01] for r in reali]
    b = [[0.45, 0.25, 0.30] if r == "1" else [0.30, 0.25, 0.45] for r in reali]

    esito = protocollo.confronta("sicuro", a, "calibrato", b, reali)

    assert esito.acc_a == pytest.approx(0.5)   # A indovina solo gli "1"
    assert esito.acc_b == pytest.approx(1.0)   # B li indovina tutti
    assert esito.rps_a > esito.rps_b
    assert esito.verdetto == "PEGGIO"


def test_rps_per_partita_coerente_con_il_modulo_modello():
    import modello
    probabilita = [[0.5, 0.3, 0.2], [0.1, 0.2, 0.7]]
    reali = ["1", "2"]
    attesi = [modello.rps({"1": 0.5, "X": 0.3, "2": 0.2}, "1"),
              modello.rps({"1": 0.1, "X": 0.2, "2": 0.7}, "2")]
    assert protocollo.rps_per_partita(probabilita, reali) == pytest.approx(attesi)


def test_predizioni_da_probabilita_prende_l_argmax():
    probabilita = [[0.5, 0.3, 0.2], [0.1, 0.6, 0.3], [0.2, 0.3, 0.5]]
    assert list(protocollo.predizioni_da_probabilita(probabilita)) == ["1", "X", "2"]


def test_partite_necessarie_cresce_al_ridursi_dell_effetto():
    # Entrambi i rapporti sono ancora NON significativi su 100 discordanti
    # (altrimenti la funzione restituisce None per costruzione), ma 52/48 e'
    # piu' vicino al caso e richiede quindi molti piu' dati di 58/42.
    tanti = protocollo.partite_necessarie_accuratezza(52, 48, 1000)
    pochi = protocollo.partite_necessarie_accuratezza(58, 42, 1000)
    assert tanti is not None and pochi is not None
    assert tanti > pochi


def test_partite_necessarie_none_se_gia_significativo():
    # 90 contro 10 su 100 discordanti e' ampiamente significativo.
    assert protocollo.partite_necessarie_accuratezza(90, 10, 1000) is None


def test_confronto_su_lunghezze_diverse_fallisce():
    with pytest.raises(ValueError):
        protocollo.confronta("a", [[0.4, 0.3, 0.3]], "b", [[0.4, 0.3, 0.3]] * 2, ["1", "X"])


def test_stagioni_test_copre_le_sette_stagioni_con_quote_di_chiusura():
    # La finestra standard deve corrispondere alla copertura reale delle quote
    # di chiusura in serie_a.csv, altrimenti il protocollo misurerebbe stagioni
    # in cui la configurazione di produzione non e' applicabile.
    import pandas as pd
    df = pd.read_csv("serie_a.csv")
    con_chiusura = sorted(df[df["OddsAvgCH"].notna()]["Stagione"].astype(str).unique())
    assert sorted(protocollo.STAGIONI_TEST) == con_chiusura
