import pandas as pd
import pytest

import predicta.serie_a.bonus.valuta_bonus as vb


def test_freebet_perde_solo_il_margine():
    valore = vb.valore_netto({"tipo": "freebet", "importo": 100.0, "margine_stimato": 0.05})
    assert valore == pytest.approx(95.0)


def test_deposito_perde_il_margine_sul_rollover():
    valore = vb.valore_netto({"tipo": "deposito", "importo": 50.0, "rollover": 3, "margine_stimato": 0.05})
    assert valore == pytest.approx(50.0 - 50.0 * 3 * 0.05)


def test_valore_clampato_a_zero():
    valore = vb.valore_netto({"tipo": "deposito", "importo": 50.0, "rollover": 10, "margine_stimato": 0.5})
    assert valore == 0.0


def test_margine_default_se_assente():
    valore = vb.valore_netto({"tipo": "freebet", "importo": 100.0})
    assert valore == pytest.approx(100.0 * (1 - vb.MARGINE_DEFAULT))


def test_tipo_sconosciuto_solleva_errore():
    with pytest.raises(ValueError):
        vb.valore_netto({"tipo": "matched", "importo": 10.0})


def test_classifica_bonus_ordina_dal_piu_conveniente():
    df = pd.DataFrame([
        {"bookmaker": "A", "tipo": "freebet", "importo": 50.0, "margine_stimato": 0.10},
        {"bookmaker": "B", "tipo": "freebet", "importo": 50.0, "margine_stimato": 0.02},
    ])
    classificato = vb.classifica_bonus(df)
    assert list(classificato["bookmaker"]) == ["B", "A"]
    assert list(classificato["valore_netto"]) == sorted(classificato["valore_netto"], reverse=True)


def test_classifica_bonus_vuota():
    assert vb.classifica_bonus(pd.DataFrame()).empty
