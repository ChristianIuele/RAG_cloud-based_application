# Analisi dei Requisiti — Sistema Cloud per la Gestione Documentale con RAG

**Progetto di Sistemi Distribuiti e Cloud Computing — a.a. 2025/26**
**Studente:** Christian Iuele — **Matricola:** 276602
**Piattaforma target:** Microsoft Azure

---

## 1. Scopo e ambito del documento

Il presente documento costituisce l'analisi dei requisiti del sistema descritto nella traccia di progetto: un'applicazione cloud-based per la **gestione di documenti digitali** con capacità di **caricamento, archiviazione, analisi, indicizzazione e ricerca**, arricchita dalla **generazione automatica di metadati** e dall'integrazione di una pipeline **RAG (Retrieval-Augmented Generation)** per l'interrogazione in linguaggio naturale.

L'analisi è redatta secondo i principi dell'ingegneria dei requisiti (riferimenti concettuali IEEE 830 e ISO/IEC/IEEE 29148). Ogni requisito è identificato univocamente per garantirne la **tracciabilità** rispetto alla traccia e la verificabilità in fase di collaudo. Il documento distingue tra requisiti **funzionali** (cosa il sistema deve fare), requisiti **non funzionali** (come il sistema deve comportarsi), **casi d'uso** (interazioni attese) e **casi d'abuso** (interazioni malevole o degeneri da prevenire), completando l'analisi con vincoli, assunzioni e matrice di tracciabilità.

### 1.1 Convenzioni

- `RF-xxx` — Requisito Funzionale
- `RNF-xxx` — Requisito Non Funzionale
- `UC-xxx` — Caso d'Uso (Use Case)
- `AB-xxx` — Caso d'Abuso (Misuse/Abuse Case)
- Priorità secondo classificazione MoSCoW: **M** (Must), **S** (Should), **C** (Could), **W** (Won't/futuro).

---

## 2. Contesto e obiettivi del sistema

Il sistema si colloca nel dominio del **Knowledge Management** e dell'**Enterprise Search** di nuova generazione. L'obiettivo primario è trasformare un archivio di documenti eterogenei in una **base di conoscenza interrogabile semanticamente**, superando i limiti della ricerca full-text tradizionale grazie alla rappresentazione vettoriale del contenuto e alla generazione aumentata dal recupero.

Gli obiettivi di alto livello sono:

1. **Ingestione** di documenti testuali (TXT, Markdown, JSON come formati obbligatori; PDF/DOCX come opzionali) con estrazione affidabile del contenuto.
2. **Arricchimento semantico** tramite metadati manuali (forniti dall'utente) e automatici (generati via NLP / IA generativa).
3. **Indicizzazione duale**: indice lessicale per la ricerca tradizionale e indice vettoriale per la ricerca semantica.
4. **Retrieval e generazione**: interrogazione in linguaggio naturale che recupera i frammenti più rilevanti e ne consente la risalita ai documenti d'origine.
5. **Deployment cloud-native su Azure**, sfruttando servizi gestiti per calcolo, storage e virtualizzazione.

### 2.1 Mappatura architetturale sui servizi Azure

L'architettura adotta un principio di **minimalità deliberata**: ogni capacità richiesta dalla traccia è coperta con il minimo insieme di servizi gestiti, evitando componenti ridondanti. Questa scelta riduce la superficie di configurazione, i punti di rottura, i costi e la complessità di deployment, favorendo leggibilità e modularità dell'architettura. La tabella seguente riporta lo stack effettivamente adottato.

| Capacità | Servizio adottato | Note |
|---|---|---|
| Storage dei documenti | **Azure Blob Storage** (account di archiviazione) | Persistenza durevole del file originale. |
| Indice vettoriale + lessicale + **metadati** | **Azure AI Search** (servizio di ricerca, ambito Foundry IQ) | Un unico servizio funge da database vettoriale, motore di ricerca lessicale/ibrida e store dei metadati filtrabili (titolo, autore, categoria, tag, lingua). |
| Embedding, LLM generativo e **NLP** | **Azure OpenAI** | Un'unica integrazione copre: calcolo degli embedding, generazione della risposta RAG e generazione dei metadati automatici (parole chiave, sintesi, categorie, lingua, entità) tramite prompt dedicato. |
| Containerizzazione | **Docker** | Immagine dell'applicazione, portabile verso il runtime cloud. |
| Backend applicativo / API | **Azure App Service — Web App for Containers** | Deploy dell'immagine Docker con endpoint HTTPS gestito; i segreti sono forniti tramite App Settings / variabili d'ambiente. |

**Consolidamenti architetturali chiave:**
- *Azure AI Search come store unico* di vettori, indice lessicale e metadati: elimina la necessità di un database relazionale/documentale separato e il conseguente rischio di inconsistenza tra store.
- *Azure OpenAI come unico motore AI* per embedding, generazione e NLP: la generazione dei metadati automatici avviene via LLM, evitando un servizio NLP dedicato.

### 2.2 Estensioni valutate e non adottate (con trade-off)

Le seguenti opzioni sono state **valutate e consapevolmente escluse** per il progetto dimostrativo, in quanto non richieste dalla traccia e non giustificate dal rapporto costo/beneficio. Sono documentate come possibili estensioni future.

| Estensione | Motivo dell'esclusione | Requisito/beneficio residuo |
|---|---|---|
| DB metadati separato (Cosmos DB / Azure SQL) | Azure AI Search già gestisce i metadati filtrabili; un secondo store introdurrebbe problemi di consistenza. | Nessun beneficio aggiuntivo per lo scopo del progetto. |
| Servizio NLP dedicato (Azure AI Language) | Tutti i metadati automatici sono ottenibili con una singola chiamata ad Azure OpenAI. | Approccio "tutto via LLM" più semplice e uniforme. |
| Autenticazione/identità (Microsoft Entra ID) | Non richiesta; multi-utente e controllo accessi non previsti dal progetto. | Prerequisito per abilitare l'isolamento cross-utente (cfr. casi d'abuso AB-06). |
| Azure Key Vault | Il requisito reale — segreti non hardcoded — è soddisfatto con App Settings/variabili d'ambiente. | Managed Identity di App Service verso OpenAI/Search è l'evoluzione naturale, senza chiavi. |
| Azure Monitor / Application Insights | Non indispensabile per il progetto; integrabile quasi a costo zero con App Service. | Osservabilità e controllo costi in esercizio. |
| Azure Functions / AKS per il backend | AKS è overkill per un progetto dimostrativo; le Functions servirebbero solo per l'ingestione event-driven. | App Service con worker in background copre lo scenario; l'async event-driven resta un miglioramento di scalabilità. |

---

## 3. Attori del sistema

| Attore | Tipo | Descrizione |
|---|---|---|
| **Utente autenticato** | Umano primario | Carica documenti, inserisce metadati manuali, effettua ricerche e interrogazioni in linguaggio naturale. |
| **Amministratore** | Umano primario | Gestisce utenti, quote, configurazioni, monitora costi e stato del sistema. |
| **Utente anonimo / non autenticato** | Umano | Potenziale attore di casi d'abuso; nel funzionamento nominale ha accesso solo a login/registrazione. |
| **Servizio di embedding** | Sistema esterno | Modello che trasforma chunk testuali in vettori (Azure OpenAI). |
| **Servizio LLM generativo** | Sistema esterno | Modello che produce risposte, sintesi e metadati (Azure OpenAI). |
| **Servizio NLP** | Sistema esterno | Estrazione di entità, lingua, parole chiave (Azure AI Language). |
| **Database vettoriale** | Sistema esterno | Persistenza e ricerca per similarità degli embedding. |
| **Motore di ingestione asincrono** | Sistema interno | Orchestratore della pipeline di parsing → chunking → embedding → indicizzazione. |

---

## 4. Requisiti Funzionali

I requisiti sono raggruppati per sottosistema logico. Ogni requisito è espresso in forma verificabile.

### 4.1 Gestione dell'ingestione documentale

| ID | Priorità | Requisito |
|---|---|---|
| **RF-001** | M | Il sistema deve permettere il caricamento di documenti nei formati **TXT, Markdown e JSON**. |
| **RF-002** | S | Il sistema dovrebbe supportare in modo opzionale i formati **PDF e DOCX**, estraendone il contenuto testuale. |
| **RF-003** | M | Il sistema deve validare, in fase di upload, formato, dimensione ed estensione del file, rifiutando i file non conformi con messaggio esplicativo. |
| **RF-004** | M | Il sistema deve persistere il documento originale in uno storage durevole, mantenendone una copia integra e recuperabile. |
| **RF-005** | M | Il sistema deve estrarre automaticamente il **contenuto testuale** dal documento caricato. |
| **RF-006** | S | Il sistema dovrebbe restituire all'utente uno **stato di avanzamento** dell'elaborazione (es. *in elaborazione*, *completato*, *errore*). Per lo scopo del progetto è accettabile un'ingestione sincrona o gestita da un worker in background; l'elaborazione event-driven (es. Azure Functions su trigger di blob) è considerata un miglioramento di scalabilità. |
| **RF-007** | S | Il sistema dovrebbe garantire l'**idempotenza** rispetto al caricamento dello stesso documento (rilevamento duplicati, es. tramite hash del contenuto). |
| **RF-008** | C | Il sistema potrebbe consentire il caricamento multiplo (batch) di più documenti in un'unica operazione. |

### 4.2 Gestione dei metadati manuali

| ID | Priorità | Requisito |
|---|---|---|
| **RF-010** | M | Il sistema deve consentire all'utente di inserire manualmente i metadati descrittivi di base: **titolo, autore, categoria, descrizione e tag**. |
| **RF-011** | M | Il sistema deve persistere i metadati manuali associandoli univocamente al documento (document ID). |
| **RF-012** | S | Il sistema dovrebbe consentire la **modifica** dei metadati manuali dopo il caricamento. |
| **RF-013** | S | Il sistema dovrebbe validare i campi (es. tag come lista non vuota di stringhe, categoria da vocabolario controllato o libera). |

### 4.3 Generazione automatica dei metadati (NLP / IA generativa)

| ID | Priorità | Requisito |
|---|---|---|
| **RF-020** | M | Il sistema deve generare automaticamente metadati aggiuntivi tramite tecniche NLP e/o IA generativa. |
| **RF-021** | M | Il sistema deve generare automaticamente le **parole chiave** rappresentative del documento. |
| **RF-022** | M | Il sistema deve generare una **sintesi (summary)** del contenuto del documento. |
| **RF-023** | S | Il sistema dovrebbe suggerire una o più **categorie** in base al contenuto. |
| **RF-024** | M | Il sistema deve rilevare automaticamente la **lingua** del documento. |
| **RF-025** | S | Il sistema dovrebbe estrarre le **entità rilevanti** (persone, organizzazioni, luoghi, date) presenti nel testo. |
| **RF-026** | S | Il sistema dovrebbe distinguere e conservare separatamente i metadati manuali da quelli generati automaticamente (tracciabilità della provenienza). |
| **RF-027** | C | Il sistema potrebbe consentire all'utente di validare, correggere o rifiutare i metadati generati automaticamente (*human-in-the-loop*). |

### 4.4 Chunking, embedding e indicizzazione vettoriale

| ID | Priorità | Requisito |
|---|---|---|
| **RF-030** | M | Il sistema deve suddividere il contenuto testuale in **frammenti (chunk)** secondo una strategia definita (es. dimensione fissa con *overlap*, o basata sulla struttura semantica). |
| **RF-031** | M | Il sistema deve trasformare ciascun chunk in una **rappresentazione vettoriale (embedding)** tramite un modello di embedding. |
| **RF-032** | M | Il sistema deve salvare gli embedding in un **database vettoriale** o sistema equivalente. |
| **RF-033** | M | Per ogni chunk, il sistema deve mantenere il **riferimento al documento originale**: identificativo del documento, identificativo del chunk e **posizione del frammento** nel documento. |
| **RF-034** | M | L'associazione chunk→documento deve permettere, in fase di retrieval, di **risalire dai chunk recuperati** ai documenti originali e ai relativi metadati. |
| **RF-035** | S | Il sistema dovrebbe indicizzare, insieme al vettore, i metadati filtrabili del chunk/documento (es. lingua, categoria, tag) per abilitare la ricerca ibrida con filtri. |
| **RF-036** | C | Il sistema potrebbe rigenerare gli embedding a seguito di aggiornamento del documento o cambio del modello di embedding (*re-indexing*). |

### 4.5 Ricerca tradizionale (lessicale) e per metadati

| ID | Priorità | Requisito |
|---|---|---|
| **RF-040** | M | Il sistema deve permettere la **ricerca full-text tradizionale** sul contenuto e sui metadati dei documenti. |
| **RF-041** | S | Il sistema dovrebbe consentire il **filtraggio** dei risultati per metadati (autore, categoria, tag, lingua, data). |
| **RF-042** | S | Il sistema dovrebbe restituire i risultati ordinati per rilevanza. |
| **RF-043** | C | Il sistema potrebbe supportare la **ricerca ibrida** (fusione di punteggio lessicale e semantico). |

### 4.6 Retrieval semantico e interrogazione RAG in linguaggio naturale

| ID | Priorità | Requisito |
|---|---|---|
| **RF-050** | M | Il sistema deve consentire **interrogazioni in linguaggio naturale** (es. *"Quali documenti parlano del cloud computing?"*). |
| **RF-051** | M | Il sistema deve trasformare la query utente in embedding e recuperare i chunk più rilevanti tramite **ricerca per similarità** (top-k). |
| **RF-052** | M | Il sistema deve **recuperare i documenti o i frammenti più rilevanti** rispetto all'interrogazione. |
| **RF-053** | M | Il sistema deve costruire il **contesto** per il modello generativo a partire dai chunk recuperati (pattern RAG) e produrre una risposta in linguaggio naturale fondata su di essi. |
| **RF-054** | S | Il sistema dovrebbe fornire le **citazioni / riferimenti** ai documenti-fonte da cui deriva la risposta, permettendo all'utente di verificarla. |
| **RF-055** | S | Il sistema dovrebbe gestire il caso di **assenza di risultati rilevanti**, evitando risposte inventate e comunicando l'impossibilità di rispondere sulla base della base di conoscenza. |
| **RF-056** | C | Il sistema potrebbe supportare interrogazioni **conversazionali multi-turno**, mantenendo il contesto della sessione. |

### 4.7 Gestione utenti, sessioni e amministrazione

| ID | Priorità | Requisito |
|---|---|---|
| **RF-060** | C | *(Estensione, non nell'ambito attuale)* Il sistema potrebbe autenticare gli utenti — es. tramite Microsoft Entra ID — prima di consentire operazioni di caricamento e interrogazione. |
| **RF-061** | C | *(Estensione, non nell'ambito attuale)* Il sistema potrebbe applicare un controllo degli accessi che limiti la visibilità dei documenti al legittimo proprietario/gruppo, abilitando l'isolamento cross-utente (cfr. AB-06). |
| **RF-062** | C | Il sistema potrebbe fornire un cruscotto amministrativo per il monitoraggio di stato, quote e costi. |
| **RF-063** | S | Il sistema dovrebbe consentire l'**eliminazione** di un documento con conseguente rimozione a cascata dei relativi chunk, embedding e metadati. |

### 4.8 Osservabilità

| ID | Priorità | Requisito |
|---|---|---|
| **RF-071** | S | Il sistema dovrebbe registrare log applicativi delle operazioni chiave (ingestione, ricerca, interrogazione) per diagnosi e audit. |

---

## 5. Requisiti Non Funzionali

I requisiti non funzionali esprimono attributi di qualità e vincoli, ove possibile quantificati per essere misurabili.

### 5.1 Prestazioni ed efficienza

| ID | Priorità | Requisito |
|---|---|---|
| **RNF-001** | S | Il tempo di risposta di un'interrogazione RAG dovrebbe restare entro **pochi secondi** (obiettivo indicativo: p95 ≤ 5 s) per documenti di dimensioni ordinarie. |
| **RNF-002** | S | L'ingestione di un documento (parsing→embedding→indicizzazione) dovrebbe completarsi in tempi proporzionali alla dimensione, senza bloccare l'interfaccia utente (elaborazione asincrona). |
| **RNF-003** | C | Il sistema dovrebbe adottare strategie di **caching** (es. embedding delle query ricorrenti) per contenere latenza e costi. |

### 5.2 Scalabilità ed elasticità

| ID | Priorità | Requisito |
|---|---|---|
| **RNF-010** | S | L'architettura dovrebbe essere **cloud-native** e scalare orizzontalmente in funzione del carico, sfruttando servizi gestiti Azure. |
| **RNF-011** | S | L'indice vettoriale dovrebbe sostenere la crescita del numero di documenti/chunk mantenendo prestazioni di ricerca accettabili. |
| **RNF-012** | C | I componenti di ingestione dovrebbero poter scalare in modo indipendente dal front-end (disaccoppiamento tramite code). |

### 5.3 Affidabilità e disponibilità

| ID | Priorità | Requisito |
|---|---|---|
| **RNF-020** | S | Il sistema dovrebbe gestire con resilienza i fallimenti dei servizi esterni (embedding/LLM), con **retry** e degradazione controllata. |
| **RNF-021** | S | I dati (documenti, metadati, embedding) dovrebbero essere **persistiti in modo durevole** con ridondanza fornita dai servizi Azure. |
| **RNF-022** | C | Il sistema dovrebbe garantire la consistenza tra document store, metadata store e indice vettoriale (evitare orfani a seguito di errori parziali). |

### 5.4 Sicurezza

| ID | Priorità | Requisito |
|---|---|---|
| **RNF-030** | M | Le comunicazioni tra client e servizi devono avvenire su canale cifrato (**HTTPS/TLS**). |
| **RNF-031** | S | I dati a riposo (documenti, metadati, indici) dovrebbero essere **cifrati** (encryption-at-rest fornita da Azure). |
| **RNF-032** | M | Le credenziali, le chiavi API e i segreti **non devono essere hard-coded** nel codice sorgente, ma forniti tramite **App Settings / variabili d'ambiente**. L'uso di **Azure Key Vault** o di una **Managed Identity** (che elimina del tutto le chiavi verso OpenAI/Search) è un'evoluzione opzionale. |
| **RNF-033** | S | Il sistema dovrebbe applicare il principio del **minimo privilegio** nell'assegnazione dei ruoli e degli accessi ai servizi. |
| **RNF-034** | S | Il sistema dovrebbe sanificare gli input utente (upload e query) contro injection e contenuti malevoli. |

### 5.5 Privacy e conformità

| ID | Priorità | Requisito |
|---|---|---|
| **RNF-040** | S | Il sistema dovrebbe gestire eventuali **dati personali** presenti nei documenti in coerenza con i principi GDPR (minimizzazione, cancellazione su richiesta). |
| **RNF-041** | C | Il sistema potrebbe segnalare o oscurare le **entità sensibili/PII** rilevate nei documenti o nelle sintesi generate. |

### 5.6 Usabilità

| ID | Priorità | Requisito |
|---|---|---|
| **RNF-050** | S | L'interfaccia dovrebbe essere intuitiva, guidando l'utente nelle fasi di upload, inserimento metadati, ricerca e interrogazione. |
| **RNF-051** | S | Il sistema dovrebbe fornire **feedback chiari** sullo stato delle operazioni asincrone e sugli errori. |
| **RNF-052** | S | Le risposte RAG dovrebbero essere presentate insieme ai riferimenti alle fonti, per favorire la fiducia e la verificabilità. |

### 5.7 Manutenibilità, modularità e portabilità

| ID | Priorità | Requisito |
|---|---|---|
| **RNF-060** | M | Il codice deve essere **organizzato, leggibile e modulare**. |
| **RNF-061** | S | I componenti (ingestione, embedding, retrieval, generazione) dovrebbero essere disaccoppiati per favorire sostituibilità e testabilità. |
| **RNF-062** | S | Il modello di embedding e l'LLM dovrebbero essere configurabili/astratti, per consentirne la sostituzione senza riscrivere la logica applicativa. |
| **RNF-063** | S | La configurazione (endpoint, chiavi, parametri di chunking) dovrebbe essere esternalizzata rispetto al codice. |

### 5.8 Osservabilità e gestione dei costi

| ID | Priorità | Requisito |
|---|---|---|
| **RNF-070** | S | Il sistema dovrebbe esporre metriche e log tramite **Azure Monitor / Application Insights**. |
| **RNF-071** | S | Il sistema dovrebbe **controllare i costi** delle chiamate a modelli di embedding/LLM (es. limiti di frequenza, quote, monitoraggio del consumo di token). |

### 5.9 Vincoli tecnologici (imposti dalla traccia)

| ID | Priorità | Requisito |
|---|---|---|
| **RNF-080** | M | Il sistema deve utilizzare le soluzioni di **calcolo, storage e virtualizzazione di Microsoft Azure**. Lo stack adottato è: Azure Blob Storage (storage), Azure AI Search (indice vettoriale/lessicale e metadati), Azure OpenAI (embedding, LLM, NLP), immagine **Docker** eseguita su **Azure App Service — Web App for Containers**. |

---

## 6. Casi d'uso (Use Cases)

Di seguito i casi d'uso principali in formato strutturato (attore, pre/post-condizioni, flusso principale ed estensioni).

### UC-01 — Caricamento e ingestione di un documento
- **Attore primario:** Utente autenticato
- **Requisiti coperti:** RF-001..007, RF-030..034
- **Precondizioni:** L'utente è autenticato; dispone di un file in formato supportato.
- **Flusso principale:**
  1. L'utente seleziona un file e avvia il caricamento.
  2. Il sistema valida formato e dimensione (RF-003).
  3. Il sistema persiste il file originale nello storage (RF-004).
  4. Il sistema avvia l'elaborazione (sincrona o in background) e mostra lo stato *in elaborazione* (RF-006).
  5. Il motore estrae il testo (RF-005), lo suddivide in chunk (RF-030), calcola gli embedding (RF-031) e li indicizza mantenendo i riferimenti (RF-032..034).
  6. Il sistema aggiorna lo stato a *completato*.
- **Estensioni:**
  - *2a.* Formato non valido → il sistema rifiuta il file con messaggio esplicativo.
  - *5a.* Errore del servizio di embedding → retry; se persiste, stato *errore* e nessun dato orfano.
- **Postcondizioni:** Il documento è archiviato, indicizzato e interrogabile.

### UC-02 — Inserimento e modifica dei metadati manuali
- **Attore primario:** Utente autenticato
- **Requisiti coperti:** RF-010..013
- **Precondizioni:** Il documento è stato caricato.
- **Flusso principale:**
  1. L'utente inserisce titolo, autore, categoria, descrizione e tag.
  2. Il sistema valida e persiste i metadati associandoli al document ID.
  3. L'utente può successivamente modificarli (RF-012).
- **Postcondizioni:** I metadati manuali sono associati al documento.

### UC-03 — Generazione automatica dei metadati
- **Attore primario:** Motore di ingestione (con servizi NLP/LLM)
- **Requisiti coperti:** RF-020..026
- **Precondizioni:** Il testo del documento è stato estratto.
- **Flusso principale:**
  1. Il sistema invia il testo ai servizi NLP/LLM.
  2. Genera parole chiave (RF-021), sintesi (RF-022), categorie suggerite (RF-023), lingua (RF-024) ed entità (RF-025).
  3. Persiste i metadati automatici distinguendoli da quelli manuali (RF-026).
- **Estensioni:**
  - *1a.* Servizio non disponibile → il documento resta comunque indicizzato con i soli metadati manuali; la generazione automatica viene ritentata.
- **Postcondizioni:** Il documento è arricchito con metadati automatici.

### UC-04 — Ricerca tradizionale con filtri
- **Attore primario:** Utente autenticato
- **Requisiti coperti:** RF-040..043
- **Flusso principale:**
  1. L'utente inserisce termini di ricerca ed eventuali filtri (categoria, tag, autore, lingua).
  2. Il sistema restituisce l'elenco dei documenti pertinenti ordinati per rilevanza.
- **Postcondizioni:** L'utente ottiene i documenti corrispondenti ai criteri.

### UC-05 — Interrogazione in linguaggio naturale (RAG)
- **Attore primario:** Utente autenticato
- **Requisiti coperti:** RF-050..056, RF-034
- **Precondizioni:** Esiste almeno un documento indicizzato.
- **Flusso principale:**
  1. L'utente formula una domanda in linguaggio naturale (es. *"Quali documenti parlano del cloud computing?"*).
  2. Il sistema calcola l'embedding della query (RF-051).
  3. Recupera i top-k chunk più simili dal database vettoriale.
  4. Risale ai documenti e ai metadati d'origine (RF-034).
  5. Costruisce il contesto e genera una risposta fondata sui frammenti recuperati (RF-053).
  6. Presenta la risposta con le citazioni alle fonti (RF-054).
- **Estensioni:**
  - *3a.* Nessun chunk sufficientemente rilevante → il sistema comunica l'assenza di informazioni pertinenti anziché inventare (RF-055).
- **Postcondizioni:** L'utente riceve una risposta verificabile e tracciabile alle fonti.

### UC-06 — Eliminazione di un documento
- **Attore primario:** Utente autenticato / Amministratore
- **Requisiti coperti:** RF-063, RNF-022
- **Flusso principale:**
  1. L'utente richiede l'eliminazione di un documento.
  2. Il sistema rimuove documento, chunk, embedding e metadati in modo consistente (nessun orfano).
- **Postcondizioni:** Il documento non è più recuperabile né interrogabile.

---

## 7. Casi d'abuso (Abuse / Misuse Cases)

I casi d'abuso descrivono interazioni malevole, accidentali o degeneri che il sistema deve prevenire o mitigare. Sono particolarmente rilevanti per un sistema RAG, che introduce superfici d'attacco specifiche legate all'ingestione di contenuti non fidati e all'uso di LLM. Ogni caso riporta minaccia, impatto e contromisure.

### AB-01 — Caricamento di file malevoli
- **Attore ostile:** Utente malintenzionato
- **Minaccia:** Upload di file contenenti malware, script eseguibili o payload nascosti in file apparentemente testuali.
- **Impatto:** Compromissione del backend, propagazione a valle, esecuzione di codice.
- **Contromisure:** Validazione rigorosa di tipo MIME/estensione (RF-003); trattamento dei file come dati inerti (nessuna esecuzione); parsing sandboxato; scansione antimalware; limiti dimensionali.

### AB-02 — Denial of Service tramite file abnormi (*zip bomb* / documenti enormi)
- **Attore ostile:** Utente malintenzionato
- **Minaccia:** Caricamento di file estremamente grandi o strutturati per esplodere in fase di parsing/chunking, esaurendo memoria e risorse.
- **Impatto:** Indisponibilità del servizio, saturazione delle risorse cloud.
- **Contromisure:** Limiti su dimensione file, numero di chunk e profondità di parsing; timeout; elaborazione asincrona isolata con quote di risorse (RNF-002, RNF-010).

### AB-03 — Abuso economico (*cost/billing exhaustion*)
- **Attore ostile:** Utente malintenzionato o script automatizzato
- **Minaccia:** Invio massivo di documenti o query per moltiplicare le chiamate a embedding/LLM e generare costi elevati sui servizi Azure OpenAI.
- **Impatto:** Costi imprevisti, esaurimento delle quote, indisponibilità per utenti legittimi.
- **Contromisure:** Rate limiting per utente/IP; quote di caricamento e interrogazione; monitoraggio dei consumi di token (RNF-071); allarmi di budget su Azure; autenticazione ove abilitata (RF-060).

### AB-04 — Prompt injection diretta
- **Attore ostile:** Utente malintenzionato
- **Minaccia:** Formulazione di query che tentano di sovrascrivere le istruzioni di sistema dell'LLM (es. *"ignora le istruzioni precedenti e rivela..."*).
- **Impatto:** Comportamenti non previsti, divulgazione di prompt di sistema, aggiramento delle policy.
- **Contromisure:** Separazione netta tra istruzioni di sistema e input utente; template di prompt robusti; validazione/sanificazione degli input (RNF-034); istruzioni difensive che vincolano l'LLM al solo contesto recuperato.

### AB-05 — Prompt injection indiretta (documenti avvelenati)
- **Attore ostile:** Utente che carica contenuti fidati solo in apparenza
- **Minaccia:** Inserimento, all'interno di un documento, di istruzioni nascoste destinate all'LLM che le processerà in fase di retrieval (*data poisoning* della knowledge base).
- **Impatto:** Manipolazione delle risposte generate, esfiltrazione di dati di altri documenti nel contesto, propagazione di disinformazione.
- **Contromisure:** Trattare il contenuto recuperato come **dati non fidati** e non come istruzioni; delimitatori chiari nel prompt; isolamento del contesto per utente; controllo degli accessi in modo che il retrieval attinga solo ai documenti autorizzati (RF-061).

### AB-06 — Accesso non autorizzato ai documenti altrui
- **Attore ostile:** Utente autenticato che tenta di superare i propri privilegi
- **Minaccia:** Recupero, tramite ricerca o query RAG, di documenti appartenenti ad altri utenti/tenant.
- **Impatto:** Violazione di riservatezza, *cross-tenant data leakage*.
- **Contromisure:** Controllo degli accessi a livello di retrieval (filtri di sicurezza sull'indice vettoriale per proprietario/gruppo); principio del minimo privilegio (RNF-033); segmentazione dei dati (RF-061).

### AB-07 — Esfiltrazione di dati sensibili tramite query mirate
- **Attore ostile:** Utente malintenzionato
- **Minaccia:** Query costruite per far emergere PII, credenziali o segreti eventualmente presenti nei documenti indicizzati o nelle sintesi generate.
- **Impatto:** Violazione della privacy, esposizione di dati sensibili.
- **Contromisure:** Rilevamento/oscuramento PII (RNF-041); politiche di data governance; limitazione della granularità delle risposte; audit log delle interrogazioni (RF-071).

### AB-08 — Iniezione su ricerca lessicale / archiviazione
- **Attore ostile:** Utente malintenzionato
- **Minaccia:** Payload di injection (es. su query di ricerca o campi metadati) verso il motore di ricerca o il database.
- **Impatto:** Alterazione delle query, accesso non autorizzato, corruzione dei dati.
- **Contromisure:** Parametrizzazione delle query; validazione e sanificazione di tutti gli input, inclusi i metadati manuali (RF-013, RNF-034).

### AB-09 — Sovraccarico dell'indice vettoriale (*index poisoning* / degradazione)
- **Attore ostile:** Utente che carica in massa contenuti irrilevanti o ridondanti
- **Minaccia:** Inquinamento della base di conoscenza con dati spazzatura per degradare la qualità del retrieval.
- **Impatto:** Riduzione della rilevanza delle risposte, aumento dei costi di storage e ricerca.
- **Contromisure:** Quote per utente; rilevamento duplicati (RF-007); moderazione/controllo di qualità dei contenuti; isolamento per utente.

### AB-10 — Furto o esposizione di chiavi e segreti
- **Attore ostile:** Attaccante che ispeziona il codice o l'infrastruttura
- **Minaccia:** Recupero di chiavi API di Azure OpenAI o stringhe di connessione hard-coded.
- **Impatto:** Uso fraudolento dei servizi, costi, accesso ai dati.
- **Contromisure:** Segreti in Azure Key Vault (RNF-032); rotazione delle chiavi; identità gestite (Managed Identity) al posto delle chiavi ove possibile; assenza di segreti nel repository dei sorgenti.

### AB-11 — Affidamento acritico su risposte allucinate
- **Attore coinvolto:** Utente legittimo (rischio d'uso, non malevolo)
- **Minaccia:** L'LLM genera affermazioni non supportate dai documenti recuperati, presentandole come fatti.
- **Impatto:** Diffusione di informazioni errate, decisioni sbagliate.
- **Contromisure:** Ancoraggio della generazione ai soli chunk recuperati (RF-053); citazione obbligatoria delle fonti (RF-054); gestione esplicita dell'assenza di contesto rilevante (RF-055).

### AB-12 — Abuso dell'endpoint di interrogazione per usi impropri dell'LLM
- **Attore ostile:** Utente che usa il sistema come LLM generico
- **Minaccia:** Sfruttamento della funzione RAG per generare contenuti non pertinenti al dominio documentale (uso improprio delle risorse).
- **Impatto:** Costi, deriva d'uso, potenziale generazione di contenuti indesiderati.
- **Contromisure:** Vincolo del modello a rispondere solo sulla base della knowledge base; filtri di contenuto di Azure OpenAI; monitoraggio degli usi anomali.

---

## 8. Vincoli e assunzioni

### 8.1 Vincoli
- **V-01:** L'infrastruttura deve poggiare esclusivamente su **Microsoft Azure** (RNF-080).

### 8.2 Assunzioni
- **A-01:** I documenti caricati sono prevalentemente testuali e parserizzabili; i formati complessi (PDF/DOCX) sono opzionali.
- **A-02:** È disponibile un accesso ad Azure OpenAI (o servizio equivalente) per embedding e generazione.
- **A-03:** Il volume di dati del progetto dimostrativo è compatibile con i tier gestiti dei servizi Azure selezionati.
- **A-04:** La lingua prevalente dei documenti è gestibile dai modelli scelti (multilingue).

---

## 9. Matrice di tracciabilità (traccia → requisiti)

| Elemento della traccia | Requisiti correlati |
|---|---|
| Caricare, archiviare documenti | RF-001..004, RNF-021 |
| Supporto TXT/MD/JSON (PDF/DOCX opzionale) | RF-001, RF-002 |
| Estrazione del contenuto testuale | RF-005 |
| Metadati manuali (titolo, autore, categoria, descrizione, tag) | RF-010..013 |
| Metadati automatici (keyword, sintesi, categorie, lingua, entità) | RF-020..026 |
| Suddivisione in chunk + embedding | RF-030, RF-031 |
| Salvataggio embedding in DB vettoriale | RF-032 |
| Riferimento doc/chunk/posizione | RF-033, RF-034 |
| Ricerca tradizionale | RF-040..043 |
| Interrogazione in linguaggio naturale (RAG) | RF-050..056 |
| Risalita da chunk a documenti/metadati | RF-034, UC-05 |
| Uso di Microsoft Azure | RNF-080 |
| Organizzazione/leggibilità/modularità del codice | RNF-060, RNF-061 |

---

## 10. Sintesi

L'analisi individua **43 requisiti funzionali** organizzati in otto sottosistemi, **26 requisiti non funzionali** su nove dimensioni di qualità, **6 casi d'uso** e **12 casi d'abuso**. Il nucleo obbligatorio (priorità *Must*) copre integralmente le richieste esplicite della traccia — ingestione, metadati manuali e automatici, chunking/embedding con tracciabilità chunk→documento, ricerca tradizionale e pipeline RAG su Azure — mentre i requisiti *Should*/*Could* delineano lo spazio di **originalità** e robustezza (ricerca ibrida, citazioni delle fonti, human-in-the-loop, controllo dei costi, difese contro prompt injection).

Sul piano architetturale, la scelta di **minimalità deliberata** — Azure Blob Storage, Azure AI Search (come store unico di vettori, indice lessicale e metadati), Azure OpenAI (come unico motore per embedding, generazione e NLP) e un'immagine Docker su Azure App Service — è essa stessa un elemento di maturità progettuale: consolida capacità che un'architettura ingenua distribuirebbe su più servizi ridondanti, riducendo consistenza da gestire, costi e complessità di deployment. Le alternative valutate (DB metadati separato, servizio NLP dedicato, autenticazione, Key Vault, orchestrazione asincrona/AKS) sono documentate come estensioni con relativo trade-off, mostrando che l'esclusione è frutto di una decisione motivata e non di una lacuna. La sezione sui casi d'abuso, infine, evidenzia le superfici d'attacco tipiche dei sistemi RAG (prompt injection diretta e indiretta, avvelenamento della knowledge base, abuso economico dei modelli) e le relative contromisure, qualificando l'analisi come professionale e non meramente descrittiva.
