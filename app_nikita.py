import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import json

# --- Datenstruktur für Dokumente ---
class Document:
    def __init__(self, doc_id: str, text: str, metadata: Dict[str, Any]):
        self.doc_id = doc_id
        self.text = text
        self.metadata = metadata
        self.embedding = None

# --- Vektordatenbank-Klasse unter Verwendung von FAISS ---
class VectorDatabase:
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.documents = []

    def add_documents(self, docs: List[Document], model: SentenceTransformer):
        embeddings = []
        for doc in docs:
            emb = model.encode(doc.text)
            doc.embedding = emb
            embeddings.append(emb)
            self.documents.append(doc)
        embeddings = np.array(embeddings).astype('float32')
        self.index.add(embeddings)

    def query(self, query_text: str, model: SentenceTransformer, top_k: int = 5) -> List[Document]:
        query_embedding = model.encode(query_text).astype('float32')
        query_embedding = np.expand_dims(query_embedding, axis=0)
        distances, indices = self.index.search(query_embedding, top_k)
        results = []
        for idx in indices[0]:
            if idx < len(self.documents):
                results.append(self.documents[idx])
        return results

# --- Funktion zum Aufruf des ollama LLM über eine REST API ---
def call_ollama(prompt: str) -> str:
    api_url = "http://localhost:11434/api/generate"  # Hier ggf. anpassen!
    headers = {"Content-Type": "application/json"}
    model = "llama3.1"
    payload = {
        "prompt": prompt,
        model: model,
        # "max_tokens": 500  # Je nach Bedarf anpassen
    }
    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        data = response.json()
        # Es wird angenommen, dass die Antwort im Feld "text" zurückgegeben wird.
        return data.get("text", "Keine Antwort erhalten.")
    except requests.RequestException as e:
        return f"LLM API-Fehler: {e}"

# --- Funktion zur Erstellung eines erweiterten Prompts ---
def create_prompt(user_query: str, retrieved_docs: List[Document]) -> str:
    context = "\n\n".join(
        [
            f"Quelle: {doc.metadata.get('source', 'Unbekannt')}\nModul: {doc.metadata.get('module', 'All')}\nText: {doc.text}"
            for doc in retrieved_docs
        ]
    )
    # Der Prompt kombiniert die Nutzeranfrage und den abgerufenen Kontext
    prompt = (
        f"User query: {user_query}\n\n"
        f"Kontext (aus abgerufenen Dokumenten):\n{context}\n\n"
        "Erstelle ein Schulungsskript für IT-Sicherheit im medizinischen Kontext, das die folgenden Anforderungen erfüllt:\n"
        "- Integration der 7-Kompetenz-Stufen (Threat Awareness, Threat Identification, Threat Impact Assessment, "
        "Tactic Choice, Tactic Justification, Tactic Mastery, Tactic Check & Follow-Up).\n"
        "- Konkrete Fallbeispiele und Szenarien, die Elemente der Social Learning Theory (SLT) integrieren (z.B. "
        "Vorbildverhalten, Beobachtungslernen).\n"
        "- Darstellung von Konsequenzen gemäß der Protection Motivation Theory (PMT), um die Dringlichkeit von Maßnahmen "
        "auf persönlicher und organisatorischer Ebene zu verdeutlichen.\n\n"
        "Bitte integriere diese Informationen in ein konsistentes Schulungsskript."
    )
    return prompt

# --- Hauptfunktion, die den gesamten Prozess koordiniert ---
def main():
    # Initialisiere das Embedding-Modell (z. B. all-MiniLM-L6-v2)
    model_name = "all-MiniLM-L6-v2"
    embedding_model = SentenceTransformer(model_name)

    # Beispielhafte Dokumente (diese können durch echte Schulungsunterlagen ersetzt werden)
    documents = [
        Document(
            doc_id="doc1",
            text=(
                "Skript 'Umgang mit Phishing': Dieses Dokument beschreibt, wie Phishing-Mails "
                "erkannt werden und welche Maßnahmen zu ergreifen sind. Es enthält praxisnahe "
                "Fallbeispiele und interaktive Elemente."
            ),
            metadata={"source": "Phishing Skript", "module": "Threat Awareness", "timestamp": "2024-01-01"}
        ),
        Document(
            doc_id="doc2",
            text=(
                "7-Stufen-Kompetenz-Entwicklung: Das Template enthält die sieben Kompetenzdimensionen "
                "von Threat Awareness bis Tactic Check & Follow-Up, die in Schulungen systematisch abgehandelt werden."
            ),
            metadata={"source": "7-Stufen Template", "module": "All", "timestamp": "2024-01-02"}
        ),
        Document(
            doc_id="doc3",
            text=(
                "Protection Motivation Theory (PMT) Fallbeispiele: Dieser Text zeigt, welche negativen "
                "Konsequenzen (z.B. Datenverlust, Reputationsschaden) eintreten können, wenn keine präventiven "
                "Maßnahmen ergriffen werden."
            ),
            metadata={"source": "PMT Paper", "module": "Threat Impact Assessment", "timestamp": "2024-01-03"}
        ),
        Document(
            doc_id="doc4",
            text=(
                "Social Learning Theory (SLT) in der IT-Sicherheit: Dieses Dokument demonstriert, wie "
                "durch Beobachtung von Vorbildern sich sichere Verhaltensweisen etablieren lassen – beispielsweise "
                "im Umgang mit verdächtigen E-Mails."
            ),
            metadata={"source": "SLT Paper", "module": "Threat Awareness, Tactic Mastery", "timestamp": "2024-01-04"}
        )
    ]

    # Initialisiere die Vektordatenbank und füge die Dokumente hinzu
    embedding_dim = embedding_model.get_sentence_embedding_dimension()
    vector_db = VectorDatabase(embedding_dim)
    vector_db.add_documents(documents, embedding_model)

    # Simuliere eine Nutzeranfrage (Requirements Management Input)
    user_query = (
        "Erstelle ein Schulungsskript für IT-Sicherheit im medizinischen Kontext, das Phishing-Angriffe "
        "behandelt und die 7-Kompetenz-Stufen integriert. Bitte berücksichtige dabei konkrete Fallbeispiele "
        "aus der Social Learning Theory sowie Konsequenzen gemäß Protection Motivation Theory."
    )

    # Abfrage der Vektordatenbank
    retrieved_docs = vector_db.query(user_query, embedding_model, top_k=3)

    # Erstelle den Prompt für das LLM
    prompt = create_prompt(user_query, retrieved_docs)
    print("Generierter Prompt für ollama LLM:")
    print(prompt)

    # Rufe das LLM (ollama) auf und erhalte die generierte Antwort
    generated_text = call_ollama(prompt)
    print("\nGenerierte Antwort von ollama LLM:")
    print(generated_text)

if __name__ == "__main__":
    main()
