Ecco una spiegazione chiara e strutturata del Teorema CAP, formattata appositamente per essere incollata nel tuo file Markdown.

---

## Il Teorema CAP nei Sistemi Distribuiti

Il **Teorema CAP** (formulato dall'informatico Eric Brewer) è un principio fondamentale per chiunque progetterà o lavorerà con database e sistemi distribuiti. Il teorema stabilisce che, in un sistema informatico distribuito, è impossibile garantire simultaneamente più di due delle seguenti tre proprietà:

* **Consistency (Coerenza):** Ogni operazione di lettura riceve l'aggiornamento più recente o restituisce un errore. In altre parole, tutti i nodi del sistema vedono esattamente gli stessi dati nello stesso momento, indipendentemente da quale nodo riceva la richiesta del client.
* **Availability (Disponibilità):** Ogni richiesta riceve sempre una risposta (senza errori) dal sistema, anche se alcuni nodi sono offline. Tuttavia, non c'è la garanzia che la risposta contenga la versione più aggiornata dei dati.
* **Partition Tolerance (Tolleranza alle Partizioni):** Il sistema continua a funzionare nonostante l'interruzione (o il ritardo) delle comunicazioni di rete tra i nodi. Una "partizione" si verifica quando la rete si divide e i nodi non riescono più a parlarsi.

### La Regola Pratica: Sceglierne due su tre

Poiché i guasti di rete in un sistema distribuito sono inevitabili (la Tolleranza alle Partizioni è quindi un requisito obbligatorio), la scelta reale per gli architetti del software si riduce quasi sempre a un compromesso tra **Coerenza (C)** e **Disponibilità (A)** in caso di problemi di rete.

Questo porta a due scenari principali:

1. **Sistemi CP (Consistency + Partition Tolerance):** In caso di divisione della rete, il sistema preferisce restituire un errore o bloccarsi piuttosto che fornire dati obsoleti.
2. **Sistemi AP (Availability + Partition Tolerance):** In caso di divisione della rete, il sistema risponde sempre, ma accetta il rischio di fornire dati non aggiornati (garantendo quella che viene chiamata *Eventual Consistency*, o coerenza eventuale).

---

### Come scelgono i Database famosi?

Le architetture dei database moderni sono progettate attorno a questi compromessi. Quando si sceglie un database per un'applicazione, si sta essenzialmente scegliendo da che parte stare nel Teorema CAP:

* **MongoDB (Modello CP):** MongoDB privilegia la **Coerenza**. Utilizza un'architettura basata su un nodo primario (Primary Replica) che gestisce tutte le scritture per garantire che i dati siano sempre coerenti. Se la rete si partiziona e il nodo primario diventa irraggiungibile, MongoDB smetterà di accettare scritture (sacrificando la Disponibilità) finché non viene eletto un nuovo nodo primario, garantendo che i dati non diventino mai discordanti. È la scelta ideale per sistemi finanziari o e-commerce dove l'accuratezza del dato è vitale.
* **Apache Cassandra (Modello AP):** Cassandra privilegia la **Disponibilità**. Utilizza un'architettura *peer-to-peer* (masterless) in cui ogni nodo può accettare letture e scritture. Se una partizione di rete isola alcuni nodi, Cassandra continuerà ad accettare scritture su entrambi i lati della partizione. Quando la rete viene ripristinata, il database risolverà i conflitti e sincronizzerà i dati in background. È la scelta perfetta per sistemi di messaggistica, log o social network, dove un breve ritardo nell'aggiornamento dei dati è preferibile a un sistema offline.
* **Database Relazionali Tradizionali (Modello CA):** Database come **MySQL** o **PostgreSQL** (nella loro configurazione classica a nodo singolo) scelgono Coerenza e Disponibilità, ma **non** tollerano le partizioni. Se il server cade o la rete salta, il database diventa inaccessibile. Non sono veri sistemi distribuiti di default, motivo per cui rientrano in questa specifica categoria.