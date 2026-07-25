# 🔮 PredictA - Pronostici Serie A

Dashboard predittiva per partite di Serie A basata su modelli statistici e quote dei bookmaker.

---

## 📋 Funzionalità

### Dashboard Principale
- **Pronostico 1X2** per qualsiasi match storico di Serie A
- **Gol attesi (xG)** calcolati con modello Poisson
- **Risultati esatti** più probabili (top 5)
- **Over/Under** 1.5 e 2.5 con gauge visivo
- **Forma recente** visualizzata con pallini 🟢🟡🔴
- **Scontri diretti** con tabella degli ultimi precedenti
- **Quote bookmaker** integrate nel modello (consenso multi-bookmaker, quota di chiusura quando disponibile)
- **Slider interattivi** per regolare i pesi del modello

### Backtesting
- **Validazione walk-forward** su 1-5 stagioni recenti a scelta (ognuna ~380 partite)
- **Metriche**: Accuratezza 1X2, F1 Score, RPS e Log-loss (calibrazione delle probabilità, non solo la previsione più probabile), Matrice di Confusione, breakdown per stagione
- **Confronto tra configurazioni**: solo storico, storico+forma, vecchi default, solo quote, ottimale validato
- **Iperparametri Dixon-Coles regolabili**: emivita delle statistiche storiche e correzione tau per i pareggi
- **Selettori** per metodo di conversione delle quote (Shin / proporzionale) e fonte delle quote (chiusura / apertura)

---

## 🧠 Il Modello

Il sistema combina **quattro componenti** con pesi regolabili:

| Componente | Descrizione | Peso Default |
|------------|-------------|--------------|
| **Media storica** | Gol fatti/subiti, pesati con decadimento temporale (le partite recenti contano di più) | 0% |
| **Forma recente** | Performance nelle ultime N partite | 10% |
| **Scontri diretti** | Precedenti tra le due squadre | 0% |
| **Quote bookmaker** | Saggezza del mercato (consenso multi-bookmaker, quota di chiusura) | 90% |

I pesi di default sono il risultato di una grid search su backtest walk-forward, **validata su tre stagioni indipendenti** (2023, 2024, 2025) e non solo su quella usata per cercarli — vedi la sezione Risultati sotto per i dettagli. Il risultato più sorprendente: né lo storico pesato nel tempo né gli scontri diretti aggiungono valore misurabile una volta pesate bene le quote (probabilmente perché il mercato incorpora già efficientemente quell'informazione); un piccolo peso alla forma recentissima invece aiuta in modo consistente. Restano comunque slider liberi se vuoi sperimentare altre combinazioni.

### Algoritmo
1. **Calcolo xG**: forza attacco/difesa di ogni squadra relativa alla media di campionato (stile Poisson classico), combinata pesando storico, forma e scontri diretti
2. **Vantaggio casa**: incorporato direttamente nelle formule (medie gol casa/trasferta separate), non come correzione aggiuntiva
3. **Decadimento temporale**: le statistiche storiche pesano ogni partita con un decadimento esponenziale (emivita configurabile, default 730 giorni) invece di una media semplice su 33 stagioni
4. **Distribuzione esatta dei punteggi**: calcolo diretto della matrice di probabilità Poisson (non più simulazione Monte Carlo), con la **correzione tau di Dixon & Coles (1997)** per i punteggi bassi — un Poisson indipendente puro sottostima sistematicamente i pareggi
5. **Conversione delle quote in probabilità**: **correzione di Shin (1992/1993)** invece della normalizzazione proporzionale `1/quota`, che ripartisce il margine del bookmaker in modo non uniforme tra gli esiti (corregge la *favorite-longshot bias*)
6. **Medie di lega pesate nel tempo**: la media gol di campionato usata come normalizzatore ha lo stesso decadimento delle statistiche di squadra, invece di essere una media semplice su 33 stagioni (correggeva un bias pro-casa del ~15% sul rapporto degli xG)
7. **Fonte delle quote**: **quota di chiusura** (a ridosso del fischio d'inizio, incorpora più informazione di mercato dell'apertura) quando disponibile — copertura dal 2019 in poi, con fallback automatico sull'apertura
8. **Blend quote**: le probabilità finali fondono il modello statistico con le probabilità implicite delle quote

---

## 📊 Risultati Backtesting

Validati su **tre stagioni di test indipendenti** (2023, 2024, 2025), ognuna con le stagioni precedenti come training — non solo sull'unica stagione su cui è girata la grid search, per escludere che i pesi scelti fossero semplicemente rumore statistico. Tutte le configurazioni sotto usano gli stessi pesi ottimali (forma=0.10, scontri=0, quote=0.90) salvo dove indicato:

| Configurazione | Acc. 2025 | Acc. 2024 | Acc. 2023 | **Media** | RPS medio |
|---|---|---|---|---|---|
| Solo storico (pesato nel tempo) | 43-46%* | - | - | ~44% | - |
| Vecchio default (forma=0.5, scontri=0.15, quote=0.15) | 50.3% | - | - | ~50% | - |
| Default pre-Fase 1 (forma=0, scontri=0.15, quote=0.85) | 54.1% | 53.4% | 52.9% | 53.5% | - |
| Solo mercato (quote, nessun modello) | 54.4% | 52.6% | 54.7% | 53.9% | - |
| Fine Fase 1 (apertura, conversione proporzionale) | 56.2% | 53.7% | 54.7% | 54.9% | 0.1889 |
| + correzione di Shin | 55.9% | 53.7% | 54.7% | 54.8% | 0.1886 |
| Shin + quote di chiusura | 55.4% | 54.7% | 55.5% | 55.2% | 0.1885 |
| **Default attuale (+ medie di lega pesate)** | **55.4%** | **54.5%** | **55.3%** | **55.1%** | **0.1884** |

*\*Numero di riferimento dalla validazione iniziale su una sola stagione, prima dell'introduzione del decadimento temporale.*

*Benchmark "predici sempre 1" sul test 2025: 38.9%. Oltre all'accuratezza, il backtesting misura anche RPS e log-loss (calibrazione delle probabilità): più basso è meglio; una previsione uniforme dà circa 0.28, una perfetta dà 0. Usa il bottone **🔬 Confronta configurazioni** nella pagina di Backtesting per riprodurre questi numeri con i tuoi dati.*

### ⚠️ Quanto sono affidabili questi numeri

Poco, ed è la cosa più importante da sapere prima di leggerli. Ripetendo il confronto su **7 stagioni** (2019-2025, 2.659 partite — l'intera finestra coperta dalle quote di chiusura) invece che su 3, e verificando la significatività statistica con il test appropriato per confronti sugli stessi match:

| Confronto | Δ accuratezza | McNemar | Verdetto |
|---|---:|---:|---|
| Modello completo vs solo mercato | +0.34 pp | p = 0.28 | **non distinguibile dal rumore** |
| Quote di chiusura vs apertura | +0.38 pp | p = 0.25 | non distinguibile dal rumore |

**Il modello non batte il mercato in modo misurabile.** Il vantaggio dichiarato in passato (~1 punto percentuale su 3 stagioni) si dimezza allargando il campione a 7 — la firma tipica di un risultato dovuto al caso. Sull'RPS il modello è anzi leggermente peggiore del solo mercato.

Il motivo tecnico: fra due configurazioni cambia previsione solo il ~2% delle partite, quindi l'accuratezza su 380 partite a stagione non ha abbastanza potenza statistica per distinguere differenze di questa dimensione. Il dettaglio, il calcolo di potenza e le conseguenze metodologiche sono in [ROADMAP.md](ROADMAP.md); lo script che lo verifica è `valida_significativita.py`.

Questo non invalida il progetto: un mercato di scommesse liquido su un campionato importante *dovrebbe* essere difficile da battere, e misurarlo onestamente è un risultato in sé.

**Cosa abbiamo imparato costruendo questi numeri (in ordine cronologico di scoperta):**
1. Il calcolo della "forma recente" usava quasi solo le partite in trasferta invece delle ultime N in ordine cronologico — bug corretto.
2. Le formule storico/forma pesavano ogni statistica come media semplice con la media di campionato, azzerando le differenze reali tra squadre e facendo collassare il modello su "vince sempre la casa" — corretto usando una forza attacco/difesa relativa alla media di lega (stile Poisson classico).
3. **Il dataset conteneva un bug serio**: `stagioni/2010.txt` era una copia esatta di `stagioni/2009.txt` — la stagione 2009/10 era contata due volte e la 2010/11 mancava del tutto. Corretto sostituendo il file con i dati reali della stagione mancante.
4. Con `peso_quote` alto, storico+forma+scontri non venivano rinormalizzati a sommare 1 tra loro: l'xG stimato collassava verso 0 (es. 0.24 gol attesi invece di ~1.6) perché le "quote" non entrano nel calcolo dell'xG ma nel blend finale delle probabilità — bug corretto.
5. Con tutti i bug corretti e il modello validato su tre stagioni indipendenti: **né lo storico né gli scontri diretti aggiungono valore misurabile sopra le sole quote di mercato**; solo un peso piccolo (10%) alla forma recentissima aiuta in modo consistente.
6. **Dopo 12 esperimenti tra Fase 2 e Fase 3, il pattern è netto**: aggiungere *feature nuove* non funziona quasi mai (Elo, gradient boosting, tiri/corner, giorni di riposo, indice motivazionale, modello Bayesiano gerarchico: tutti negativi); funziona invece *estrarre meglio l'informazione già contenuta nelle quote* (Shin e quota di chiusura, gli unici due miglioramenti adottati).
7. **Una revisione del codice ha poi trovato sei bug**, il più grave dei quali nella dashboard: le quote degli scontri diretti non venivano orientate rispetto a chi giocava in casa, quindi la probabilità di vittoria della squadra di casa era mescolata con quella dell'avversaria — su Milan–Inter, 0.41 invece di 0.27 su una componente pesata al 90%. Corretti tutti, con test di regressione. Dettaglio in [ROADMAP.md](ROADMAP.md).
8. **La lezione finale è metodologica**: verificando la significatività statistica, *nessuno* dei risultati di Fase 2 e 3 — né i positivi né i negativi — era distinguibile dal rumore. Il collo di bottiglia non era il modello, era il campione su cui lo misuravamo.

Il dettaglio di ogni esperimento — con numeri, causa probabile del fallimento e codice di validazione — è in **[ROADMAP.md](ROADMAP.md)**.

---

## 🛠️ Tecnologie

- **Python 3.10+** (sviluppato e testato con Python 3.14)
- **Streamlit** - Dashboard interattiva
- **Pandas** - Manipolazione dati
- **NumPy** - Calcoli numerici
- **SciPy** - Distribuzione di Poisson (Dixon-Coles), ricerca della radice per Shin, test binomiale per McNemar
- **Plotly** - Grafici interattivi
- **Scikit-learn** - Metriche di validazione, regressione di Poisson per Elo, prototipi gradient boosting
- **Requests** - Download dello storico Elo da clubelo.com e delle altre leghe da football-data.co.uk
- **Pytest** - Test automatici sul modello (vedi `tests/`)

---

## 📁 Struttura del Progetto

```
PredictA/
├── app.py                                  # Dashboard principale
├── modello.py                              # Modulo condiviso: Dixon-Coles, decadimento temporale, Shin, RPS, Elo
├── unisci_dati.py                          # Unisce i CSV stagionali (Date, Stagione, quote apertura e chiusura)
├── scarica_elo.py                          # Scarica lo storico Elo delle squadre da clubelo.com
├── scarica_altre_leghe.py                  # Scarica Premier/Liga/Bundesliga/Ligue 1 (prototipo multi-lega)
├── test_dati.py                            # Script di ispezione rapida di un file stagione
├── serie_a.csv                             # Dataset completo (11.534 partite)
├── elo_storico.csv                         # Storico Elo delle 53 squadre (output di scarica_elo.py, gitignored)
├── requirements.txt                        # Dipendenze Python pinnate
│
├── pages/
│   └── backtesting.py                      # Pagina di validazione (walk-forward multi-stagione)
│
├── tests/
│   ├── conftest.py
│   └── test_model.py                       # Test automatici su forma, scontri diretti, pesi, Elo, Shin, integrità dati
│
├── stagioni/                               # File .txt delle singole stagioni (1993 → 2025)
├── altre_leghe/                            # Storico delle altre 4 leghe (gitignored)
├── archive/                                # Dataset xG Understat da Kaggle (gitignored, ~21MB)
│
├── valida_shin.py                          # Validazione: correzione di Shin (adottata)
├── valida_quote_chiusura.py                # Validazione: quote di chiusura (adottata)
├── valida_medie_lega.py                    # Validazione: medie di lega pesate nel tempo (adottata)
├── valida_pesi_medie_pesate.py             # Grid search dei pesi con le medie pesate
├── valida_significativita.py               # Test di significativita' (McNemar + bootstrap RPS)
├── valida_understat.py                     # Validazione: xG reali di Understat (non adottata)
├── prototipo_gradient_boosting.py          # Prototipo ML (+ flag --con-tiri, --con-riposo, --con-motivazione)
├── prototipo_gradient_boosting_multiliga.py# Prototipo ML con training multi-campionato
├── prototipo_bayesiano_gerarchico.py       # Prototipo: stima MAP del modello di Baio & Blangiardo
├── prototipo_ensemble_stacking.py          # Prototipo: meta-learner out-of-fold
│
├── readme.md                               # Questo file
└── ROADMAP.md                              # Piano tecnico dettagliato Fase 2/3 e registro degli esperimenti
```

Gli script `valida_*.py` e `prototipo_*.py` non fanno parte del percorso di produzione: sono il registro riproducibile degli esperimenti descritti in [ROADMAP.md](ROADMAP.md), tenuti in repo perché rieseguibili.

---

## 🚀 Installazione

### Prerequisiti
- Python 3.10 o superiore
- pip (package manager Python)

### Setup

```bash
# 1. Clona o crea la cartella del progetto
mkdir predicta
cd predicta

# 2. Crea ambiente virtuale
python -m venv venv

# 3. Attiva ambiente virtuale
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 4. Installa dipendenze
pip install -r requirements.txt

# 5. Scarica i dati
# Vai su https://www.football-data.co.uk/italym.php
# Scarica tutte le stagioni dal 1993 a oggi come .txt
# Mettili nella cartella "stagioni/"

# 6. Unisci i dati
python unisci_dati.py

# 7. (opzionale) Scarica lo storico Elo (non usato dai pesi di default, ma disponibile per sperimentare)
python scarica_elo.py

# 8. (opzionale) Esegui i test automatici
pytest tests/

# 9. Avvia la dashboard
streamlit run app.py
```

---

## 🎮 Utilizzo

### Dashboard Principale
1. Seleziona due squadre dai menu a tendina
2. Regola i pesi nella sidebar (Forma, Scontri, Quote)
3. Clicca **CALCOLA**
4. Analizza probabilità, gol attesi, Over/Under e scontri diretti

### Backtesting
1. Clicca su **Backtesting** nella sidebar di Streamlit
2. Scegli quante stagioni recenti usare come test (walk-forward: quelle precedenti sono training), imposta i pesi del modello, gli iperparametri Dixon-Coles (emivita, rho), il metodo di conversione delle quote e la fonte delle quote
3. Clicca **🚀 Esegui Backtesting** per validare i pesi scelti, oppure **🔬 Confronta configurazioni** per vedere fianco a fianco solo storico / storico+forma / vecchi default / solo quote / ottimale
4. Visualizza accuratezza, RPS, log-loss, matrice di confusione, metriche per classe e (se selezioni più stagioni) l'accuratezza stagione per stagione

---

## 📈 Prossimi Sviluppi

Fase 1, Fase 2 e Fase 3 della roadmap sono **chiuse**: 15 voci testate, 3 adottate in produzione (correzione di Shin, quote di chiusura, medie di lega pesate nel tempo).

La lezione operativa è che **le feature nuove non pagano, mentre estrarre meglio l'informazione dalle quote sì** — ma con l'avvertenza pesante della sezione sull'affidabilità sopra: nessuna di queste differenze è statisticamente distinguibile. Da qui l'ordine dei prossimi passi:

- [ ] **Protocollo di misura** (priorità 0, prerequisito per tutto il resto): RPS con intervallo di confidenza bootstrap come criterio primario, finestra di test a 7 stagioni invece di 3, McNemar sull'accuratezza. Nessun dato nuovo richiesto
- [ ] **Rivalutare ensemble stacking e multi-campionato** col protocollo nuovo: erano stati bocciati sull'accuratezza, che non aveva la potenza per bocciarli, mentre sull'RPS erano avanti
- [ ] **Movimento apertura→chiusura e dispersione tra bookmaker** come feature separate (non come sostituzione della quota): l'unica idea rimasta che segue la stessa logica dei miglioramenti adottati
- [ ] **pi-ratings** (Constantinou & Fenton): l'unico sistema di rating continuo mai provato — Elo è stato testato ed è risultato negativo
- [ ] **Valore di mercato Transfermarkt** come variazione temporale (slope 30/90/180gg), non come feature statica
- [ ] **Fonte xG aggiornabile**: il segnale di Understat è debolmente positivo ma il dataset disponibile si ferma a settembre 2024

Il piano dettagliato (fonti dati specifiche, letteratura accademica, tecniche di modellazione, errori comuni da evitare, e il registro completo degli esperimenti falliti con la causa probabile) è in **[ROADMAP.md](ROADMAP.md)**.

---

## ⚠️ Disclaimer

Questo software è a scopo **dimostrativo ed educativo**. Non costituisce invito al gioco d'azzardo. Le previsioni sono basate su modelli statistici e non garantiscono risultati reali. Il gioco d'azzardo può causare dipendenza.

La dashboard principale permette di scegliere **qualsiasi coppia di squadre** presenti nello storico, usando tutti i dati disponibili (anche successivi, se scegli un accoppiamento "datato"): è pensata come simulazione "chi vincerebbe oggi tra queste due squadre", non come previsione retrospettiva di una partita realmente giocata in una data specifica. La pagina di **Backtesting**, invece, rispetta rigorosamente l'ordine cronologico (nessun dato futuro rispetto alla partita da prevedere) ed è quella da usare per valutare l'accuratezza reale del modello.

---

## 📊 Dataset

- **Fonte**: [Football-Data.co.uk](https://www.football-data.co.uk/italym.php)
- **Periodo**: 1993 - 2026 (stagione 2025/26 in corso)
- **Partite**: 11.534, tutte le 33 stagioni presenti senza duplicati né buchi (verificato: nessuna partita duplicata, nessuna stagione mancante tra la prima e l'ultima — controllo automatico in `tests/test_model.py`)
- **Squadre**: 53
- **Colonne**: Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR, Stagione, `OddsAvgH/D/A` (quota di consenso multi-bookmaker in apertura), `OddsAvgCH/CD/CA` (quota di consenso in chiusura, 2.660 partite dal 2019), B365H/D/A, PSH/D/A, oltre a tiri/corner/cartellini quando disponibili (testati e non usati dal modello di produzione)

---

## 👨‍💻 Sviluppato con
- Python
- Streamlit
- Claude
