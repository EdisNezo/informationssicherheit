import os
import json
import copy
import re
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path
from datetime import datetime
import html

# Externe Bibliotheken
import torch
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS  # Geändert von Chroma zu FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.base import BaseCallbackHandler

# Logging konfigurieren
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Verantwortlich für das Laden, Verarbeiten und Indexieren von Dokumenten
    für den RAG-basierten E-Learning-Generator für Informationssicherheit.
    """
    
    def __init__(self, documents_dir: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialisiert den DocumentProcessor.
        
        Args:
            documents_dir: Verzeichnis, in dem die zu indexierenden Dokumente gespeichert sind
            chunk_size: Größe der Textabschnitte für die Indexierung
            chunk_overlap: Überlappung zwischen Textabschnitten
        """
        self.documents_dir = Path(documents_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
    def load_documents(self) -> List[Document]:
        """
        Lädt alle Dokumente aus dem angegebenen Verzeichnis.
        
        Returns:
            Liste von Document-Objekten
        """
        documents = []
        
        # Stelle sicher, dass das Verzeichnis existiert
        if not self.documents_dir.exists():
            logger.warning(f"Dokumentverzeichnis {self.documents_dir} existiert nicht.")
            return documents
        
        # Durchlaufe alle Dateien im Verzeichnis
        file_count = 0
        for file_path in self.documents_dir.glob("**/*.*"):
            if file_path.is_file():
                try:
                    # Lese den Inhalt der Datei
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    
                    # Bestimme den Dokumenttyp anhand des Dateinamens/Pfads
                    doc_type = self._determine_document_type(file_path)
                    
                    # Erstelle ein Document-Objekt mit Metadaten
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": str(file_path),
                            "file_name": file_path.name,
                            "doc_type": doc_type
                        }
                    )
                    
                    documents.append(doc)
                    logger.info(f"Dokument geladen: {file_path}")
                    file_count += 1
                    
                except Exception as e:
                    logger.error(f"Fehler beim Laden von {file_path}: {e}")
        
        logger.info(f"{file_count} Dokumente erfolgreich geladen")
        return documents
    
    def _determine_document_type(self, file_path: Path) -> str:
        """
        Bestimmt den Dokumenttyp anhand des Dateipfads oder -namens.
        
        Args:
            file_path: Pfad zur Datei
            
        Returns:
            String mit dem Dokumenttyp
        """
        path_str = str(file_path).lower()
        
        if "template" in path_str:
            return "template"
        elif "policy" in path_str or "richtlinie" in path_str:
            return "policy"
        elif "compliance" in path_str or "vorschrift" in path_str:
            return "compliance"
        elif "best_practice" in path_str or "empfehlung" in path_str:
            return "best_practice"
        elif "beispiel" in path_str or "example" in path_str:
            return "example"
        elif "threat" in path_str or "bedrohung" in path_str or "risiko" in path_str:
            return "threat"
        elif "learning_theory" in path_str or "lerntheorie" in path_str:
            return "learning_theory"
        elif "security" in path_str or "sicherheit" in path_str:
            return "security_content"
        elif "industry" in path_str or "branche" in path_str:
            return "industry_specific"
        elif "process" in path_str or "prozess" in path_str:
            return "process"
        # Neue Dokumenttypen für die verbesserte Struktur
        elif "awareness" in path_str:
            return "threat_awareness"
        elif "identification" in path_str:
            return "threat_identification"
        elif "assessment" in path_str or "impact" in path_str:
            return "threat_impact_assessment"
        elif "choice" in path_str or "tactic_choice" in path_str:
            return "tactic_choice"
        elif "justification" in path_str:
            return "tactic_justification"
        elif "mastery" in path_str:
            return "tactic_mastery"
        elif "follow" in path_str or "check" in path_str:
            return "tactic_check_follow_up"
        else:
            # Versuche, anhand des Verzeichnisnamens zu bestimmen
            parent_dir = file_path.parent.name.lower()
            for doc_type in ["policies", "compliance", "templates", "examples", "threats", "best_practices", 
                           "security", "learning_theories", "industries", "processes"]:
                if parent_dir.startswith(doc_type) or doc_type in parent_dir:
                    return doc_type[:-1] if doc_type.endswith("s") else doc_type
            
            return "generic"
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """
        Verarbeitet Dokumente für die Indexierung: Chunking und Metadaten-Anreicherung.
        
        Args:
            documents: Liste von Document-Objekten
            
        Returns:
            Liste von verarbeiteten Document-Objekten
        """
        processed_docs = []
        
        for doc in documents:
            # Teile das Dokument in Chunks
            chunks = self.text_splitter.split_documents([doc])
            
            # Füge zusätzliche Metadaten hinzu, wenn nötig
            for i, chunk in enumerate(chunks):
                # Übernehme die Metadaten des Originaldokuments
                chunk.metadata = doc.metadata.copy()
                # Füge Chunk-spezifische Metadaten hinzu
                chunk.metadata["chunk_id"] = i
                chunk.metadata["chunk_total"] = len(chunks)
                
                # Extrahiere Abschnittstyp, falls möglich
                if doc.metadata["doc_type"] == "template":
                    section_type = self._extract_section_type(chunk.page_content)
                    if section_type:
                        chunk.metadata["section_type"] = section_type
                
                processed_docs.append(chunk)
        
        logger.info(f"Dokumente verarbeitet: {len(documents)} Dokumente in {len(processed_docs)} Chunks aufgeteilt")
        return processed_docs
    
    def _extract_section_type(self, content: str) -> Optional[str]:
        """
        Extrahiert den Abschnittstyp aus dem Inhalt eines Template-Chunks.
        
        Args:
            content: Textinhalt des Chunks
            
        Returns:
            String mit dem Abschnittstyp oder None, wenn nicht gefunden
        """
        # Verbesserte Heuristik für die Extraktion basierend auf dem Beispiel-Skript
        content_lower = content.lower()
        
        # Spezifische Abschnittstypen aus dem Beispiel-Skript
        if any(term in content_lower for term in ["threat awareness", "bedrohungsbewusstsein"]):
            return "threat_awareness"
        elif any(term in content_lower for term in ["threat identification", "bedrohungserkennung"]):
            return "threat_identification"
        elif any(term in content_lower for term in ["threat impact assessment", "bedrohungsausmaß"]):
            return "threat_impact_assessment"
        elif any(term in content_lower for term in ["tactic choice", "taktische maßnahmenauswahl"]):
            return "tactic_choice"
        elif any(term in content_lower for term in ["tactic justification", "maßnahmenrechtfertigung"]):
            return "tactic_justification"
        elif any(term in content_lower for term in ["tactic mastery", "maßnahmenbeherrschung"]):
            return "tactic_mastery"
        elif any(term in content_lower for term in ["tactic check", "follow-up", "anschlusshandlungen"]):
            return "tactic_check_follow_up"
        
        # Allgemeinere Heuristik als Fallback
        elif any(term in content_lower for term in ["lernziel", "learning objective", "ziel"]):
            return "learning_objectives"
        elif any(term in content_lower for term in ["inhalt", "content", "thema"]):
            return "content"
        elif any(term in content_lower for term in ["methode", "didaktik", "method", "format"]):
            return "methods"
        elif any(term in content_lower for term in ["prüfung", "assessment", "evaluation", "test", "quiz"]):
            return "assessment"
        elif any(term in content_lower for term in ["bedrohung", "threat", "risiko", "risk", "vulnerability"]):
            return "threats"
        elif any(term in content_lower for term in ["maßnahme", "control", "schutz", "protection"]):
            return "controls"
        elif any(term in content_lower for term in ["kontext", "context", "umgebung", "environment"]):
            return "context"
        elif any(term in content_lower for term in ["prozess", "process", "workflow", "ablauf"]):
            return "process"
        
        return None


# Callback-Handler für verbesserte Logging und Überwachung der LLM-Antworten
class LLMCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.current_token_count = 0
        self.current_tokens = []
        self.hallucination_patterns = [
            r"ich weiß nicht",
            r"ich bin mir nicht sicher",
            r"es tut mir leid",
            r"entschuldigung",
            r"ich habe keine information",
            r"ich wurde nicht trainiert",
            r"ich kann nicht",
        ]
        self.potential_hallucinations = []
        
    def on_llm_start(self, serialized, prompts, **kwargs):
        """Wird aufgerufen, wenn das LLM eine Anfrage erhält."""
        self.current_token_count = 0
        self.current_tokens = []
        self.potential_hallucinations = []
        
    def on_llm_new_token(self, token: str, **kwargs):
        """Wird aufgerufen, wenn das LLM ein neues Token generiert."""
        self.current_token_count += 1
        self.current_tokens.append(token)
        
        # Überprüfe auf potenzielle Halluzinationen
        current_text = "".join(self.current_tokens[-50:])  # Prüfe nur die letzten 50 Tokens
        for pattern in self.hallucination_patterns:
            if re.search(pattern, current_text, re.IGNORECASE):
                self.potential_hallucinations.append((pattern, current_text))
                logger.warning(f"Potenzielle Halluzination erkannt: {pattern} in '{current_text}'")
        
    def on_llm_end(self, response, **kwargs):
        """Wird aufgerufen, wenn das LLM eine Antwort abschließt."""
        full_response = "".join(self.current_tokens)
        logger.info(f"LLM-Antwort abgeschlossen. Generierte {self.current_token_count} Tokens.")
        
        # Zusammenfassung der potenziellen Halluzinationen
        if self.potential_hallucinations:
            logger.warning(f"Insgesamt {len(self.potential_hallucinations)} potenzielle Halluzinationen erkannt.")
        else:
            logger.info("Keine potenziellen Halluzinationen erkannt.")
            
    def on_llm_error(self, error, **kwargs):
        """Wird aufgerufen, wenn ein Fehler im LLM auftritt."""
        logger.error(f"LLM-Fehler aufgetreten: {error}")


class VectorStoreManager:
    """
    Verwaltet die Vektordatenbank für das Retrieval von Dokumenten.
    """
    
    def __init__(self, persist_directory: str = "./data/faiss_index"):
        """
        Initialisiert den VectorStoreManager.
        
        Args:
            persist_directory: Verzeichnis zum Speichern der Vektordatenbank
        """
        self.persist_directory = persist_directory
        
        # Initialisiere das Embedding-Modell
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.vectorstore = None
    
    def create_vectorstore(self, documents: List[Document]) -> None:
        """
        Erstellt eine Vektordatenbank aus den gegebenen Dokumenten.
        
        Args:
            documents: Liste von Document-Objekten
        """
        # Stelle sicher, dass das Verzeichnis existiert
        os.makedirs(os.path.dirname(self.persist_directory), exist_ok=True)
        
        # Erstelle eine neue Vektordatenbank mit FAISS
        self.vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
        
        # Persistiere die Datenbank
        self.vectorstore.save_local(self.persist_directory)
        
        logger.info(f"Vektordatenbank erstellt mit {len(documents)} Dokumenten")
    
    def load_vectorstore(self) -> bool:
        """
        Lädt eine bestehende Vektordatenbank.
        
        Returns:
            True, wenn die Datenbank geladen wurde, False sonst
        """
        try:
            # Versuche, die Vektordatenbank zu laden
            if os.path.exists(self.persist_directory) and os.path.isdir(self.persist_directory):
                if len(os.listdir(self.persist_directory)) > 0:  # Überprüfe, ob Dateien im Verzeichnis vorhanden sind
                    self.vectorstore = FAISS.load_local(
                        folder_path=self.persist_directory,
                        embeddings=self.embeddings
                    )
                    
                    logger.info("Bestehende Vektordatenbank geladen")
                    return True
                else:
                    logger.warning(f"Vektordatenbank-Verzeichnis {self.persist_directory} ist leer.")
                    return False
            else:
                logger.warning(f"Vektordatenbank-Verzeichnis {self.persist_directory} existiert nicht oder ist kein Verzeichnis.")
                return False
        except Exception as e:
            logger.error(f"Fehler beim Laden der Vektordatenbank: {e}", exc_info=True)
            return False
    
    def get_retriever(self, search_type: str = "similarity", search_kwargs: Dict[str, Any] = None):
        """
        Gibt einen Retriever für die Vektordatenbank zurück.
        
        Args:
            search_type: Art der Suche (FAISS unterstützt "similarity")
            search_kwargs: Zusätzliche Suchparameter
            
        Returns:
            Retriever-Objekt
        """
        if self.vectorstore is None:
            raise ValueError("Vektordatenbank wurde nicht initialisiert")
        
        if search_kwargs is None:
            search_kwargs = {"k": 5}
        
        return self.vectorstore.as_retriever(
            search_kwargs=search_kwargs
        )
    
    def retrieve_documents(self, query: str, filter: Dict[str, Any] = None, k: int = 5) -> List[Document]:
        """
        Führt eine Suche in der Vektordatenbank durch.
        
        Args:
            query: Suchanfrage
            filter: Filter für die Suche (wird nach der FAISS-Suche manuell angewendet)
            k: Anzahl der zurückzugebenden Dokumente
            
        Returns:
            Liste von gefundenen Document-Objekten
        """
        if self.vectorstore is None:
            raise ValueError("Vektordatenbank wurde nicht initialisiert")
        
        # FAISS unterstützt kein direktes Filtern wie Chroma
        # Wir rufen daher mehr Dokumente ab und filtern sie manuell
        search_k = k * 3 if filter else k  # Holen mehr Dokumente, wenn Filter angewendet wird
        
        results = self.vectorstore.similarity_search(
            query=query,
            k=search_k
        )
        
        # Manuelle Filterung, wenn Filter angegeben ist
        if filter is not None:
            filtered_results = []
            for doc in results:
                matches_filter = True
                for key, value in filter.items():
                    doc_value = doc.metadata.get(key)
                    
                    # Fix für den "argument of type 'int' is not iterable"-Fehler
                    # Stelle sicher, dass wir Werte richtig vergleichen, auch wenn sie unterschiedliche Typen haben
                    if isinstance(doc_value, int) and isinstance(value, str):
                        # Versuche, den String in eine Zahl zu konvertieren
                        try:
                            value = int(value)
                        except ValueError:
                            matches_filter = False
                            break
                    
                    if doc_value != value:
                        matches_filter = False
                        break
                
                if matches_filter:
                    filtered_results.append(doc)
            
            # Begrenze auf die angeforderte Anzahl
            return filtered_results[:k]
        
        return results[:k]
    
    def retrieve_with_multiple_queries(self, queries: List[str], filter: Dict[str, Any] = None, top_k: int = 3) -> List[Document]:
        """
        Führt mehrere Retrieval-Anfragen durch und aggregiert die Ergebnisse.
        
        Args:
            queries: Liste von Suchanfragen
            filter: Filter für die Suche
            top_k: Anzahl der zurückzugebenden Dokumente pro Anfrage
            
        Returns:
            Liste von aggregierten Document-Objekten
        """
        all_docs = []
        seen_ids = set()  # Verhindert Duplikate
        
        for query in queries:
            docs = self.retrieve_documents(query, filter, top_k)
            
            for doc in docs:
                # Überspringen, wenn das Dokument bereits vorhanden ist
                doc_id = f"{doc.metadata.get('source')}_{doc.metadata.get('chunk_id')}"
                if doc_id in seen_ids:
                    continue
                    
                seen_ids.add(doc_id)
                all_docs.append(doc)
        
        # Begrenze die Gesamtanzahl
        return all_docs[:top_k*2]


class LLMManager:
    """
    Verwaltet die Interaktion mit dem Large Language Model.
    """
    
    def __init__(self, model_name: str = "llama3:8b"):
        """
        Initialisiert den LLMManager.
        
        Args:
            model_name: Name des zu verwendenden LLM-Modells
        """
        # LLM-Callback für verbesserte Überwachung
        self.callback_handler = LLMCallbackHandler()
        
        # Initialisiere das LLM mit Callback
        self.llm = Ollama(
            model=model_name,
            callbacks=[self.callback_handler],
            num_ctx=2048
        )
        
        # Definiere Standard-Prompts für verschiedene Aufgaben
        self.prompts = {
            "question_generation": self._create_question_generation_prompt(),
            "content_generation": self._create_content_generation_prompt(),
            "hallucination_check": self._create_hallucination_check_prompt(),
            "key_info_extraction": self._create_key_info_extraction_prompt()
        }
        
        # Erstelle LLM-Chains für die verschiedenen Aufgaben
        self.chains = {
            name: LLMChain(llm=self.llm, prompt=prompt)
            for name, prompt in self.prompts.items()
        }
    
    def _create_question_generation_prompt(self) -> PromptTemplate:
        """
        Erstellt ein Prompt-Template für die Fragengenerierung.
        
        Returns:
            PromptTemplate-Objekt
        """
        template = """
        Du bist ein freundlicher Berater, der auf Deutsch mit Kunden kommuniziert. Alle deine Antworten MÜSSEN auf Deutsch sein.
        
        Deine Aufgabe ist es, Fragen zu stellen, die einem nicht-technischen Kunden helfen, über die Prozesse und den Kontext seines Unternehmens 
        zu sprechen. Der Kunde hat KEIN Fachwissen über Informationssicherheit.
        
        Formuliere eine freundliche, leicht verständliche Frage auf Deutsch zu folgendem Thema:
        - Abschnitt des E-Learning-Kurses: {section_title}
        - Beschreibung: {section_description}
        
        Berücksichtige dabei:
        - Organisation: {organization}
        - Zielgruppe: {audience}
        - Relevanter Kontext: {context_text}
        
        Die Frage sollte:
        1. Sich auf die Geschäftsprozesse, tägliche Abläufe oder den Arbeitskontext des Kunden beziehen
        2. In einfacher, nicht-technischer Sprache formuliert sein
        3. Offen sein und ausführliche Antworten fördern
        4. KEINEN Fachjargon aus der Informationssicherheit enthalten
        5. Dem Kunden nicht das Gefühl geben, dass er Informationssicherheitswissen haben müsste
        
        Gib nur die Frage zurück, keine Erklärungen oder Einleitungen.
        
        WICHTIG: Deine Antwort muss auf Deutsch sein!
        """
        
        return PromptTemplate(
            template=template,
            input_variables=["section_title", "section_description", "context_text", 
                            "organization", "audience"]
        )
    
    def _create_content_generation_prompt(self) -> PromptTemplate:
        template = """
        Erstelle den Inhalt für den Abschnitt "{section_title}" eines E-Learning-Kurses zur Informationssicherheit.
        
        WICHTIG: Der gesamte Inhalt MUSS auf Deutsch sein! Verwende durchgehend die deutsche Sprache.
        
        Die Antwort des Kunden auf deine Frage nach den Unternehmensprozessen war:
        "{user_response}"
        
        Basierend auf dieser Antwort sollst du nun relevante Informationssicherheitsinhalte generieren, die:
        1. Speziell auf die beschriebenen Prozesse, Herausforderungen und den Unternehmenskontext zugeschnitten sind
        2. Praktische Sicherheitsmaßnahmen und bewährte Verfahren enthalten, die für diese Prozesse relevant sind
        3. Klare Anweisungen und Empfehlungen bieten, die die Zielgruppe verstehen und umsetzen kann
        4. Technische Konzepte auf eine zugängliche, nicht einschüchternde Weise erklären
        
        Kontext und weitere Informationen:
        - Abschnittsbeschreibung: {section_description}
        - Organisation: {organization}
        - Zielgruppe: {audience}
        - Dauer: {duration}
        - Relevante Informationen aus unserer Wissensdatenbank: {context_text}
        
        Der Inhalt sollte dem Format des Beispielskripts entsprechen und für Abschnitt "{section_title}" angemessen sein.
        
        Gib nur den fertigen Inhalt zurück, keine zusätzlichen Erklärungen.
        
        WICHTIG: Deine Antwort muss vollständig auf Deutsch sein!
        """
        
        return PromptTemplate(
            template=template,
            input_variables=["section_title", "section_description", "user_response", 
                            "organization", "audience", "duration", "context_text"]
        )
    
    def _create_hallucination_check_prompt(self) -> PromptTemplate:
        """
        Erstellt ein Prompt-Template für die Halluzinationsprüfung.
        
        Returns:
            PromptTemplate-Objekt
        """
        template = """
        Überprüfe den folgenden Inhalt für einen E-Learning-Kurs zur Informationssicherheit auf mögliche Ungenauigkeiten oder Halluzinationen.
        
        Zu prüfender Text: 
        {content}
        
        Kontext aus der Kundenantwort:
        {user_input}
        
        Verfügbare Fachinformationen:
        {context_text}
        
        Bitte identifiziere:
        1. Aussagen über Informationssicherheit, die nicht durch die verfügbaren Fachinformationen gestützt werden
        2. Empfehlungen oder Maßnahmen, die für den beschriebenen Unternehmenskontext ungeeignet sein könnten
        3. Technische Begriffe oder Konzepte, die falsch verwendet wurden
        4. Widersprüche zu bewährten Sicherheitspraktiken
        5. Unzutreffende Behauptungen über Bedrohungen oder deren Auswirkungen
        
        Für jede identifizierte Problemstelle:
        - Zitiere die betreffende Textpassage
        - Erkläre, warum dies problematisch ist
        - Schlage eine fachlich korrekte Alternative vor
        
        Falls keine Probleme gefunden wurden, antworte mit "KEINE_PROBLEME".
        """
        
        return PromptTemplate(
            template=template,
            input_variables=["content", "user_input", "context_text"]
        )
    
    def _create_key_info_extraction_prompt(self) -> PromptTemplate:
        """
        Erstellt ein Prompt-Template für die Extraktion von Schlüsselinformationen.
        
        Returns:
            PromptTemplate-Objekt
        """
        template = """
        Analysiere die folgende Antwort eines Kunden, der über die Prozesse und den Kontext seines Unternehmens spricht.
        Der Kunde hat auf eine Frage zu "{section_type}" geantwortet, für die wir passende Informationssicherheitsinhalte erstellen wollen.
        
        Kundenantwort:
        "{user_response}"
        
        Extrahiere:
        1. Die wichtigsten Geschäftsprozesse, Arbeitsabläufe oder Systeme, die erwähnt werden
        2. Potenzielle Informationssicherheits-Schwachstellen oder Risiken, die mit diesen Prozessen verbunden sein könnten
        3. Besondere Anforderungen oder Einschränkungen, die berücksichtigt werden sollten
        4. Branchenspezifische Aspekte, die relevant sein könnten
        5. Informationswerte oder schützenswerte Daten, die im Kontext wichtig sind
        
        Gib nur eine Liste von 5-8 Schlüsselbegriffen oder kurzen Phrasen zurück, die für die Suche nach relevanten Informationssicherheitsinhalten verwendet werden können. Schreibe keine Einleitungen oder Erklärungen.
        """
        
        return PromptTemplate(
            template=template,
            input_variables=["section_type", "user_response"]
        )
    
    def generate_question(self, section_title: str, section_description: str, 
                         context_text: str, organization: str, audience: str) -> str:
        """
        Generiert eine Frage für einen Abschnitt des Templates.
        
        Args:
            section_title: Titel des Abschnitts
            section_description: Beschreibung des Abschnitts
            context_text: Kontextinformationen aus dem Retrieval
            organization: Art der Organisation
            audience: Zielgruppe der Schulung
            
        Returns:
            Generierte Frage
        """
        max_context_length = 1000
        if len(context_text) > max_context_length:
            context_text = context_text[:max_context_length] + "..."
        
        try:
            response = self.chains["question_generation"].run({
                "section_title": section_title,
                "section_description": section_description,
                "context_text": context_text,
                "organization": organization,
                "audience": audience
            })
            return response.strip()
        except Exception as e:
            logger.error(f"Fehler bei der Fragengenerierung: {e}")
            # Fallback bei Fehler
            return f"Können Sie mir etwas über {section_title} in Ihrem Unternehmen erzählen?"
        
    def generate_content(self, section_title: str, section_description: str, 
                        user_response: str, organization: str, audience: str, 
                        duration: str, context_text: str) -> str:
        """
        Generiert Inhalt für einen Abschnitt der Schulung.
        
        Args:
            section_title: Titel des Abschnitts
            section_description: Beschreibung des Abschnitts
            user_response: Antwort des Nutzers
            organization: Art der Organisation
            audience: Zielgruppe der Schulung
            duration: Maximale Dauer der Schulung
            context_text: Kontextinformationen aus dem Retrieval
            
        Returns:
            Generierter Inhalt
        """
        response = self.chains["content_generation"].run({
            "section_title": section_title,
            "section_description": section_description,
            "user_response": user_response,
            "organization": organization,
            "audience": audience,
            "duration": duration,
            "context_text": context_text
        })
        
        return response.strip()
    
    def check_hallucinations(self, content: str, user_input: str, context_text: str) -> Tuple[bool, str]:
        """
        Überprüft den generierten Inhalt auf Halluzinationen.
        
        Args:
            content: Generierter Inhalt
            user_input: Ursprüngliche Eingabe des Nutzers
            context_text: Kontextinformationen aus dem Retrieval
            
        Returns:
            Tuple aus (hat_probleme, korrigierter_inhalt)
        """
        response = self.chains["hallucination_check"].run({
            "content": content,
            "user_input": user_input,
            "context_text": context_text
        })
        
        # Prüfe, ob Probleme gefunden wurden
        has_issues = "KEINE_PROBLEME" not in response
        
        # Korrigiere den Inhalt basierend auf dem Check
        if has_issues:
            corrected_content = self.generate_content_with_corrections(content, response)
        else:
            corrected_content = content
        
        return has_issues, corrected_content
    
    def generate_content_with_corrections(self, original_content: str, correction_feedback: str) -> str:
        """
        Generiert korrigierten Inhalt basierend auf dem Feedback.
        
        Args:
            original_content: Ursprünglicher Inhalt
            correction_feedback: Feedback zur Korrektur
            
        Returns:
            Korrigierter Inhalt
        """
        correction_prompt = f"""
        Überarbeite den folgenden E-Learning-Inhalt basierend auf dem Feedback:
        
        Originaltext:
        {original_content}
        
        Feedback zur Überarbeitung:
        {correction_feedback}
        
        Erstelle eine verbesserte Version des Textes, die die identifizierten Probleme behebt, fachlich korrekt ist und trotzdem verständlich und ansprechend bleibt.
        Achte darauf, dass der Text weiterhin didaktisch gut aufbereitet ist und alle wichtigen Informationen enthält.
        
        Gib nur den überarbeiteten Text zurück, keine zusätzlichen Erklärungen.
        """
        
        corrected_content = self.llm(correction_prompt)
        
        return corrected_content.strip()
    
    def extract_key_information(self, section_type: str, user_response: str) -> List[str]:
        """
        Extrahiert Schlüsselinformationen aus der Antwort des Nutzers.
        
        Args:
            section_type: Art des Abschnitts (z.B. "learning_objectives")
            user_response: Antwort des Nutzers
            
        Returns:
            Liste von Schlüsselbegriffen
        """
        response = self.chains["key_info_extraction"].run({
            "section_type": section_type,
            "user_response": user_response
        })
        
        # Verarbeite die Antwort zu einer Liste
        key_concepts = [
            concept.strip() 
            for concept in response.strip().split("\n") 
            if concept.strip()
        ]
        
        return key_concepts

    def advanced_hallucination_detection(self, content: str) -> Dict[str, Any]:
        """
        Führt eine erweiterte Halluzinationserkennung durch.
        
        Args:
            content: Zu prüfender Inhalt
            
        Returns:
            Dictionary mit Analyseergebnissen
        """
        # Muster für typische Halluzinationsindikatoren
        hallucination_patterns = {
            "Unsicherheit": [
                r"könnte sein", r"möglicherweise", r"eventuell", r"vielleicht",
                r"unter umständen", r"es ist denkbar", r"in der regel"
            ],
            "Widersprüche": [
                r"einerseits.*andererseits", r"jedoch", r"allerdings",
                r"im gegensatz dazu", r"wiederum"
            ],
            "Vage Aussagen": [
                r"irgendwie", r"gewissermaßen", r"im großen und ganzen",
                r"im allgemeinen", r"mehr oder weniger"
            ]
        }
        
        results = {
            "detected_patterns": {},
            "confidence_score": 1.0,  # Anfangswert, wird für jedes gefundene Muster reduziert
            "suspicious_sections": []
        }
        
        # Überprüfe den Text auf Muster
        content_lower = content.lower()
        
        for category, patterns in hallucination_patterns.items():
            category_matches = []
            for pattern in patterns:
                matches = re.finditer(pattern, content_lower)
                for match in matches:
                    start_pos = max(0, match.start() - 40)
                    end_pos = min(len(content_lower), match.end() + 40)
                    context = content_lower[start_pos:end_pos]
                    category_matches.append({
                        "pattern": pattern,
                        "context": context
                    })
                    
                    # Reduziere den Confidence-Score für jeden Fund
                    results["confidence_score"] = max(0.1, results["confidence_score"] - 0.05)
                    
                    # Speichere den Abschnitt als verdächtig
                    results["suspicious_sections"].append(context)
            
            if category_matches:
                results["detected_patterns"][category] = category_matches
        
        return results


class TemplateManager:
    """
    Verwaltet das Template für den E-Learning-Kurs.
    """
    
    def __init__(self, template_path: str = None):
        """
        Initialisiert den TemplateManager.
        
        Args:
            template_path: Pfad zur Template-Datei
        """
        self.template_path = template_path
        self.template = self.load_template()
    
    def load_template(self) -> Dict[str, Any]:
        """
        Lädt das Template aus der angegebenen Datei oder erstellt ein Standard-Template
        basierend auf dem Beispiel-Skript.
        
        Returns:
            Template als Dictionary
        """
        # Verwende ein an das Beispiel-Skript angepasstes Template
        default_template = {
            "title": "E-Learning-Kurs zur Informationssicherheit",
            "description": "Ein maßgeschneiderter Kurs zur Stärkung des Sicherheitsbewusstseins",
            "sections": [
                {
                    "id": "threat_awareness",
                    "title": "Threat Awareness",
                    "description": "Bedrohungsbewusstsein: Kontext und Ausgangssituationen, in denen Gefahren auftreten können",
                    "type": "threat_awareness"
                },
                {
                    "id": "threat_identification",
                    "title": "Threat Identification",
                    "description": "Bedrohungserkennung: Merkmale und Erkennungshinweise für potenzielle Gefahren",
                    "type": "threat_identification"
                },
                {
                    "id": "threat_impact_assessment",
                    "title": "Threat Impact Assessment",
                    "description": "Bedrohungsausmaß: Konsequenzen, die aus der Bedrohung entstehen können",
                    "type": "threat_impact_assessment"
                },
                {
                    "id": "tactic_choice",
                    "title": "Tactic Choice",
                    "description": "Taktische Maßnahmenauswahl: Handlungsoptionen zur Bedrohungsabwehr",
                    "type": "tactic_choice"
                },
                {
                    "id": "tactic_justification",
                    "title": "Tactic Justification",
                    "description": "Maßnahmenrechtfertigung: Begründung für die gewählten Maßnahmen",
                    "type": "tactic_justification"
                },
                {
                    "id": "tactic_mastery",
                    "title": "Tactic Mastery",
                    "description": "Maßnahmenbeherrschung: Konkrete Schritte zur Umsetzung der gewählten Handlungen",
                    "type": "tactic_mastery"
                },
                {
                    "id": "tactic_check_follow_up",
                    "title": "Tactic Check & Follow-Up",
                    "description": "Anschlusshandlungen: Schritte nach der Ausführung der Maßnahmen",
                    "type": "tactic_check_follow_up"
                }
            ]
        }
        
        if self.template_path:
            try:
                with open(self.template_path, "r", encoding="utf-8") as f:
                    template = json.load(f)
                
                logger.info(f"Template geladen: {self.template_path}")
                return template
            except Exception as e:
                logger.error(f"Fehler beim Laden des Templates: {e}")
                logger.info("Verwende Standard-Template basierend auf dem Beispiel-Skript")
                return default_template
        else:
            logger.info("Kein Template-Pfad angegeben. Verwende Standard-Template basierend auf dem Beispiel-Skript")
            return default_template
    
    def get_section_by_id(self, section_id: str) -> Optional[Dict[str, Any]]:
        """
        Gibt einen Abschnitt des Templates anhand seiner ID zurück.
        
        Args:
            section_id: ID des Abschnitts
            
        Returns:
            Abschnitt als Dictionary oder None, wenn nicht gefunden
        """
        # Konvertiere zu String, falls es ein Integer ist
        section_id_str = str(section_id)
        
        for section in self.template["sections"]:
            if str(section["id"]) == section_id_str:
                return section
        
        return None
    
    def get_next_section(self, completed_sections: List[str]) -> Optional[Dict[str, Any]]:
        """
        Gibt den nächsten nicht bearbeiteten Abschnitt zurück.
        
        Args:
            completed_sections: Liste der bereits bearbeiteten Abschnitte
            
        Returns:
            Nächster Abschnitt als Dictionary oder None, wenn alle bearbeitet
        """
        # Konvertiere die completed_sections zu Strings, falls sie Integers enthalten
        completed_sections_str = [str(section_id) for section_id in completed_sections]
        
        for section in self.template["sections"]:
            section_id = str(section["id"])
            if section_id not in completed_sections_str:
                return section
        
        return None
    
    def create_script_from_responses(self, section_responses: Dict[str, str], context_info: Dict[str, str]) -> Dict[str, Any]:
        """
        Erstellt ein Skript aus den gegebenen Antworten.
        
        Args:
            section_responses: Dictionary mit Abschnitts-IDs als Schlüssel und Inhalten als Werte
            context_info: Dictionary mit Kontextinformationen
            
        Returns:
            Vollständiges Skript als Dictionary
        """
        script = copy.deepcopy(self.template)
        
        # Passe den Titel und die Beschreibung an
        organization = context_info.get("Für welche Art von Organisation erstellen wir den E-Learning-Kurs (z.B. Krankenhaus, Bank, Behörde)?", "")
        audience = context_info.get("Welche Mitarbeitergruppen sollen geschult werden?", "")
        
        if organization:
            script["title"] = f'Skript „Umgang mit Informationssicherheit für {organization}"'
            script["description"] = f"Willkommen zum Trainingsmodul, in dem Sie lernen, wie Beschäftigte in {organization} mit Informationssicherheit umgehen."
        
        # Füge die Inhalte zu den Abschnitten hinzu
        for section in script["sections"]:
            section_id = section["id"]
            if section_id in section_responses:
                section["content"] = section_responses[section_id]
        
        return script


class TemplateGuidedDialog:
    """
    Implementiert einen dialogbasierten Prozess zur Erstellung eines E-Learning-Kurses.
    """
    
    def __init__(self, template_manager: TemplateManager, llm_manager: LLMManager, 
                vector_store_manager: VectorStoreManager):
        """
        Initialisiert den TemplateGuidedDialog.
        
        Args:
            template_manager: TemplateManager-Instanz
            llm_manager: LLMManager-Instanz
            vector_store_manager: VectorStoreManager-Instanz
        """
        self.template_manager = template_manager
        self.llm_manager = llm_manager
        self.vector_store_manager = vector_store_manager
        
        # Initialisiere den Gesprächszustand
        self.conversation_state = {
            "current_step": "greeting",
            "context_info": {},
            "section_responses": {},
            "generated_content": {},  # Vom LLM generierter Inhalt pro Abschnitt
            "completed_sections": [],
            "current_section": None,
            "content_quality_checks": {},
            "current_section_question_count": 0
        }
        
        # Liste der Kontextfragen
        self.context_questions = [
        "Für welche Art von Organisation erstellen wir den E-Learning-Kurs (z.B. Krankenhaus, Bank, Behörde)?",
        "Welche Mitarbeitergruppen sollen geschult werden?",
        "Wie lang sollte der E-Learning-Kurs maximal dauern?",
        "Mit welchen Arten von Informationen oder Daten arbeiten Ihre Mitarbeiter im Alltag?",
        "Wie kommunizieren Ihre Mitarbeiter typischerweise miteinander und mit externen Partnern?"
        ]
    
    def get_next_question(self) -> str:
        """
        Bestimmt die nächste Frage basierend auf dem aktuellen Gesprächszustand.
        
        Returns:
            Nächste Frage als String
        """
        if self.conversation_state["current_step"] == "greeting":
            # Wechsle zum nächsten Schritt
            self.conversation_state["current_step"] = "context_gathering"
            return "Willkommen! Ich unterstütze Sie heute bei der Erstellung eines maßgeschneiderten E-Learning-Kurses zur Informationssicherheit. Um einen auf Ihre Bedürfnisse zugeschnittenen Kurs zu entwickeln, würde ich gerne mehr über Ihre Organisation erfahren. Für welche Art von Organisation erstellen wir diesen Kurs?"
        
        elif self.conversation_state["current_step"] == "context_gathering":
            # Finde die nächste unbeantwortete Kontextfrage
            for question in self.context_questions:
                if question not in self.conversation_state["context_info"]:
                    return question
            
            # Alle Kontextfragen wurden beantwortet
            self.conversation_state["current_step"] = "template_navigation"
            return "Vielen Dank für diese Informationen! Jetzt werde ich Ihnen einige Fragen zu spezifischen Bereichen der Informationssicherheit stellen, damit wir einen gut angepassten Kurs erstellen können.\n\n" + self.get_next_template_question()
        
        elif self.conversation_state["current_step"] == "template_navigation":
            return self.get_next_template_question()
        
        elif self.conversation_state["current_step"] == "review":
            return "Ich habe basierend auf Ihren Eingaben einen E-Learning-Kurs zur Informationssicherheit entworfen. Möchten Sie das Ergebnis sehen?"
        
        elif self.conversation_state["current_step"] == "completion":
            return "Vielen Dank für Ihre Mitwirkung! Der E-Learning-Kurs wurde erfolgreich erstellt und kann jetzt heruntergeladen werden."
    
    def get_next_template_question(self) -> str:
        """
        Bestimmt die nächste Frage basierend auf dem Template mit robuster Fehlerbehandlung.
        
        Returns:
            Nächste Frage als String
        """
        # Vordefinierte Fragen für Notfälle
        predefined_questions = {
            "threat_awareness": "Wie sieht ein typischer Arbeitstag in Ihrem Unternehmen aus, besonders in Bezug auf den Umgang mit externen E-Mails oder Informationen?",
            "threat_identification": "Sind Ihnen schon einmal verdächtige E-Mails oder andere Kommunikation aufgefallen? Was hat Sie stutzig gemacht?",
            "threat_impact_assessment": "Welche Auswirkungen hätte es auf Ihre tägliche Arbeit, wenn wichtige Daten plötzlich nicht mehr verfügbar wären?",
            "tactic_choice": "Wie gehen Sie aktuell vor, wenn Sie etwas Verdächtiges bemerken?",
            "tactic_justification": "Warum halten Sie die aktuellen Sicherheitsmaßnahmen in Ihrem Unternehmen für sinnvoll?",
            "tactic_mastery": "Welche konkreten Schritte unternehmen Sie, wenn Sie eine verdächtige E-Mail erhalten?",
            "tactic_check_follow_up": "Was passiert in Ihrem Unternehmen, nachdem ein Sicherheitsvorfall gemeldet wurde?"
        }
        
        try:
            # Stelle sicher, dass completed_sections eine Liste von Strings ist
            completed_sections = [str(section_id) for section_id in self.conversation_state.get("completed_sections", [])]
            
            # Finde den nächsten nicht bearbeiteten Abschnitt
            next_section = self.template_manager.get_next_section(completed_sections)
            
            if next_section is None:
                # Alle Abschnitte wurden bearbeitet
                self.conversation_state["current_step"] = "review"
                return self.get_next_question()
            
            # Prüfe, ob wir zu einem neuen Abschnitt wechseln
            current_section = self.conversation_state.get("current_section")
            current_section_str = str(current_section) if current_section is not None else None
            next_section_id_str = str(next_section["id"])
            is_new_section = current_section_str != next_section_id_str
            
            # Initialisiere question_error_count, falls nicht vorhanden
            if "question_error_count" not in self.conversation_state:
                self.conversation_state["question_error_count"] = 0
            
            if is_new_section:
                # Setze Zähler zurück bei neuem Abschnitt
                self.conversation_state["current_section_question_count"] = 0
                self.conversation_state["question_error_count"] = 0
            
            # Abschnittsinformationen
            section_id = next_section["id"]
            section_title = next_section["title"]
            section_description = next_section["description"]
            section_type = next_section.get("type", "generic")
            
            # Aktualisiere den Gesprächszustand
            self.conversation_state["current_section"] = section_id
            
            try:
                # Hole relevante Dokumente für diesen Abschnitt
                retrieval_queries = self.generate_retrieval_queries(section_title, section_id)
                
                retrieved_docs = self.vector_store_manager.retrieve_with_multiple_queries(
                    queries=retrieval_queries,
                    filter={"section_type": section_type},
                    top_k=2  # Reduziere auf 2 statt 3 für weniger Speicherverbrauch
                )
                
                # Begrenze Kontext auf max. 1000 Zeichen
                context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
                if len(context_text) > 1000:
                    context_text = context_text[:1000] + "..."
                
                # Kontextinformationen für die Fragengenerierung
                organization = self.conversation_state["context_info"].get(
                    "Für welche Art von Organisation erstellen wir den E-Learning-Kurs (z.B. Krankenhaus, Bank, Behörde)?", "")
                audience = self.conversation_state["context_info"].get(
                    "Welche Mitarbeitergruppen sollen geschult werden?", "")
                
                # Versuche, Frage mit LLM zu generieren
                question = self.llm_manager.generate_question(
                    section_title=section_title,
                    section_description=section_description,
                    context_text=context_text,
                    organization=organization,
                    audience=audience
                )
                
                # Setze Fehlerzähler zurück bei Erfolg
                self.conversation_state["question_error_count"] = 0
                
            except Exception as e:
                # Protokolliere Fehler
                logger.error(f"Fehler bei der Fragengenerierung für {section_title}: {e}", exc_info=True)
                
                # Erhöhe Fehlerzähler
                self.conversation_state["question_error_count"] += 1
                
                # Bei zu vielen Fehlern zum nächsten Abschnitt wechseln
                if self.conversation_state["question_error_count"] > 2:
                    logger.warning(f"Zu viele Fehler bei {section_title}, überspringe Abschnitt")
                    self.conversation_state["completed_sections"].append(section_id)
                    return self.get_next_template_question()
                
                # Verwende vordefinierte Frage als Fallback
                section_type_str = str(section_type)
                question = predefined_questions.get(section_type_str, f"Können Sie mir mehr über {section_title} in Ihrem Arbeitsalltag erzählen?")
            
            # Füge Übergangsinfo hinzu bei neuem Abschnitt
            if is_new_section:
                return f"Nun kommen wir zum Abschnitt '{section_title}'.\n\n{question}"
            
            return question
            
        except Exception as e:
            logger.error(f"Kritischer Fehler in get_next_template_question: {e}", exc_info=True)
            # Absoluter Fallback bei kritischen Fehlern
            return "Können Sie mir mehr über die Informationssicherheit in Ihrem Unternehmen erzählen?"
    
    def generate_retrieval_queries(self, section_title: str, section_id: str) -> List[str]:
        """
        Erzeugt Retrieval-Anfragen für einen Abschnitt des Templates.
        
        Args:
            section_title: Titel des Abschnitts
            section_id: ID des Abschnitts
            
        Returns:
            Liste von Retrieval-Anfragen
        """
        # Konvertiere section_id zu String, falls es ein Integer ist
        section_id_str = str(section_id)
        
        # Basis-Anfrage aus dem Abschnittstitel
        base_query = section_title
        
        # Kontextspezifische Anfragen
        organization = self.conversation_state["context_info"].get(
            "Für welche Art von Organisation erstellen wir den E-Learning-Kurs (z.B. Krankenhaus, Bank, Behörde)?", "")
        
        # Schütze vor leeren oder ungültigen Werten
        organization = organization if organization else "Organisation"
        
        # Branchenspezifische Anfrage
        industry_query = f"Informationssicherheit für {organization}"
        
        # Verbesserte abschnittsspezifische Anfragetemplates entsprechend dem Beispiel-Skript
        section_templates = {
            "threat_awareness": f"Bedrohungsbewusstsein Kontext Informationssicherheit {organization}",
            "threat_identification": f"Merkmale Bedrohungserkennung Informationssicherheit {organization}",
            "threat_impact_assessment": f"Konsequenzen Bedrohungen Informationssicherheit {organization}",
            "tactic_choice": f"Handlungsoptionen Sicherheitsmaßnahmen {organization}",
            "tactic_justification": f"Begründung Rechtfertigung Sicherheitsmaßnahmen {organization}",
            "tactic_mastery": f"Konkrete Schritte Umsetzung Sicherheitsmaßnahmen {organization}",
            "tactic_check_follow_up": f"Anschlusshandlungen Nach Sicherheitsvorfall {organization}"
        }
        
        # Verwende das entsprechende Template, falls vorhanden
        section_query = section_templates.get(section_id_str, "")
        
        # Compliance-spezifische Anfrage hinzufügen, falls vorhanden
        compliance_query = ""
        compliance = self.conversation_state["context_info"].get(
            "Gibt es spezifische Compliance-Anforderungen oder Branchenstandards, die berücksichtigt werden müssen?", "")
        if compliance and compliance.strip():
            compliance_query = f"Informationssicherheit {compliance} {organization}"
        
        # Spezifische Anfragen für bestimmte Abschnitte
        specific_queries = []
        if section_id_str == "threat_awareness":
            specific_queries.append(f"Alltägliche Situationen Informationssicherheit {organization}")
        elif section_id_str == "threat_identification":
            specific_queries.append(f"Phishing Erkennung Merkmale {organization}")
        elif section_id_str == "threat_impact_assessment":
            specific_queries.append(f"Konsequenzen Datenverlust Cyberangriff {organization}")
        elif section_id_str == "tactic_choice":
            specific_queries.append(f"Schutzmaßnahmen Verdächtige E-Mails {organization}")
        elif section_id_str == "tactic_justification":
            specific_queries.append(f"Warum E-Mail-Sicherheit wichtig {organization}")
        elif section_id_str == "tactic_mastery":
            specific_queries.append(f"Schritte Überprüfung Verdächtige E-Mails {organization}")
        elif section_id_str == "tactic_check_follow_up":
            specific_queries.append(f"Meldeverfahren Informationssicherheitsvorfälle {organization}")
        
        # Erstelle die Liste der Anfragen
        queries = [base_query, industry_query]
        
        if section_query:
            queries.append(section_query)
        
        if compliance_query:
            queries.append(compliance_query)
        
        queries.extend(specific_queries)
        
        return queries
    
    def process_user_response(self, response: str) -> str:
        """
        Verarbeitet die Antwort des Nutzers und aktualisiert den Gesprächszustand.
        
        Args:
            response: Antwort des Nutzers
            
        Returns:
            Nächste Frage oder Nachricht
        """
        if self.conversation_state["current_step"] == "context_gathering":
            # Bestimme, welche Kontextfrage gerade beantwortet wurde
            for question in self.context_questions:
                if question not in self.conversation_state["context_info"]:
                    self.conversation_state["context_info"][question] = response
                    break
        
        elif self.conversation_state["current_step"] == "template_navigation":
            # Erhöhe den Fragenzähler für den aktuellen Abschnitt
            self.conversation_state["current_section_question_count"] += 1
            
            # Prüfe, ob wir das Maximum an Fragen erreicht haben oder die Antwort ausreichend ist
            max_questions_reached = self.conversation_state["current_section_question_count"] >= 3
            
            try:
                if max_questions_reached or self.is_response_adequate(response):
                    # Speichere die Antwort für den aktuellen Abschnitt
                    current_section = self.conversation_state["current_section"]
                    if current_section:
                        self.conversation_state["section_responses"][current_section] = response
                        
                        # Generiere Inhalt für diesen Abschnitt
                        self._generate_section_content(current_section)
                        
                        self.conversation_state["completed_sections"].append(current_section)
                        
                        # Setze den Fragenzähler für den nächsten Abschnitt zurück
                        self.conversation_state["current_section_question_count"] = 0
                else:
                    # Fordere eine detailliertere Antwort an
                    return self.generate_followup_question(response)
            except Exception as e:
                logger.error(f"Fehler bei der Verarbeitung der Antwort: {e}", exc_info=True)
                # Setze die Konversation bei einem Fehler fort
                current_section = self.conversation_state.get("current_section")
                if current_section:
                    self.conversation_state["section_responses"][current_section] = response
                    self.conversation_state["completed_sections"].append(current_section)
                    self.conversation_state["current_section_question_count"] = 0
                    logger.warning(f"Überspringen der Inhaltsgeneration für Sektion {current_section} aufgrund eines Fehlers.")
        
        elif self.conversation_state["current_step"] == "review":
            # Prüfe, ob der Nutzer das Ergebnis sehen möchte
            if any(word in response.lower() for word in ["ja", "gerne", "zeigen", "ansehen"]):
                self.conversation_state["current_step"] = "completion"
                return "Hier ist der entworfene E-Learning-Kurs zur Informationssicherheit basierend auf Ihren Eingaben:\n\n" + self.get_script_summary()
            else:
                # Frage erneut, ob der Nutzer das Ergebnis sehen möchte
                return "Möchten Sie den erstellten E-Learning-Kurs sehen? Bitte antworten Sie mit 'Ja' oder 'Nein'."
        
        # Bestimme die nächste Frage
        return self.get_next_question()
    
    def is_response_adequate(self, response: str) -> bool:
        """
        Prüft, ob die Antwort des Nutzers ausreichend detailliert ist.
        
        Args:
            response: Antwort des Nutzers
            
        Returns:
            True, wenn die Antwort ausreichend ist, False sonst
        """
        # Verbesserte Heuristik: Prüfe die Länge und den Inhalt der Antwort
        min_word_count = 15
        word_count = len(response.split())
        
        # Überprüfe, ob die Antwort relevant für den aktuellen Abschnitt ist
        current_section_id = self.conversation_state["current_section"]
        current_section = self.template_manager.get_section_by_id(current_section_id)
        
        if current_section:
            # Erstelle eine Liste von relevanten Begriffen für diesen Abschnitt
            relevant_terms = []
            
            # Konvertiere section_id in String, falls es ein Integer ist
            section_id_str = str(current_section_id)
            
            if "threat" in section_id_str:
                relevant_terms.extend(["gefahr", "risiko", "bedrohung", "sicherheit", "schaden"])
            if "tactic" in section_id_str:
                relevant_terms.extend(["maßnahme", "vorgehen", "schutz", "handlung", "prozess"])
            
            # Prüfe, ob mindestens ein relevanter Begriff in der Antwort vorkommt
            has_relevant_term = any(term in response.lower() for term in relevant_terms)
            
            return word_count >= min_word_count and has_relevant_term
        
        return word_count >= min_word_count
    
    def generate_followup_question(self, response: str) -> str:
        """
        Generiert eine Nachfrage für eine unzureichende Antwort.
        
        Args:
            response: Unzureichende Antwort des Nutzers
            
        Returns:
            Nachfrage als String
        """
        current_section_id = self.conversation_state["current_section"]
        section = self.template_manager.get_section_by_id(current_section_id)
        
        if not section:
            # Fallback, wenn die Sektion nicht gefunden wird
            return "Könnten Sie bitte mehr Details zu Ihren Prozessen oder Ihrem Arbeitskontext erläutern?"
        
        followup_prompt = f"""
        Die folgende Antwort des Kunden zu einer Frage über {section['title']} ist recht kurz oder allgemein:
        
        "{response}"
        
        Formuliere eine freundliche Nachfrage, die mehr Details zu ihren Prozessen oder ihrem Arbeitskontext erbittet.
        Die Nachfrage sollte:
        1. Wertschätzend für die bisherige Antwort sein
        2. Konkrete Aspekte ansprechen, zu denen mehr Details hilfreich wären
        3. Keine Fachbegriffe aus der Informationssicherheit verwenden
        4. Offen formuliert sein, um ausführlichere Antworten zu fördern
        5. Auf den spezifischen Abschnittstyp "{section['title']}" zugeschnitten sein
        
        Für "Threat Awareness" beispielsweise könntest du nach konkreten Situationen im Arbeitsalltag fragen.
        Für "Tactic Mastery" könntest du nach spezifischen Schritten in bestimmten Prozessen fragen.
        
        Stelle nur die Nachfrage, keine Einleitung oder zusätzliche Erklärungen.
        """
        
        followup_question = self.llm_manager.llm(followup_prompt)
        
        return followup_question
    
    def _generate_section_content(self, section_id: str) -> None:
        """
        Generiert den Inhalt für einen Abschnitt und führt Qualitätsprüfungen durch.
        
        Args:
            section_id: ID des Abschnitts
        """
        # Konvertiere zu String, falls es ein Integer ist
        section_id_str = str(section_id)
        
        # Hole die Antwort des Nutzers
        user_response = self.conversation_state["section_responses"][section_id]
        section = self.template_manager.get_section_by_id(section_id)
        
        if not section:
            logger.warning(f"Sektion mit ID {section_id} nicht gefunden!")
            return
        
        # Extrahiere Schlüsselinformationen für das Retrieval
        key_concepts = self.llm_manager.extract_key_information(
            section_type=section.get("type", "generic"),
            user_response=user_response
        )
        
        # Generiere Retrieval-Anfragen basierend auf den Schlüsselkonzepten
        retrieval_queries = [
            f"{concept} Informationssicherheit" 
            for concept in key_concepts
        ]
        
        # Füge abschnittsspezifische Anfragen hinzu
        retrieval_queries.extend(self.generate_retrieval_queries(section["title"], section_id_str))
        
        # Hole relevante Dokumente
        retrieved_docs = self.vector_store_manager.retrieve_with_multiple_queries(
            queries=retrieval_queries,
            filter={"section_type": section.get("type", "generic")},
            top_k=5
        )
        
        # Extrahiere Kontext aus den abgerufenen Dokumenten
        context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # Generiere Inhalt für diesen Abschnitt
        organization = self.conversation_state["context_info"].get(
            "Für welche Art von Organisation erstellen wir den E-Learning-Kurs (z.B. Krankenhaus, Bank, Behörde)?", "")
        audience = self.conversation_state["context_info"].get(
            "Welche Mitarbeitergruppen sollen geschult werden?", "")
        duration = self.conversation_state["context_info"].get(
            "Wie lang sollte der E-Learning-Kurs maximal dauern?", "")
        
        content = self.llm_manager.generate_content(
            section_title=section["title"],
            section_description=section.get("description", ""),
            user_response=user_response,
            organization=organization,
            audience=audience,
            duration=duration,
            context_text=context_text
        )
        
        # Führe erweiterte Halluzinationsprüfung durch
        advanced_check = self.llm_manager.advanced_hallucination_detection(content)
        
        # Führe Qualitätsprüfung durch
        has_issues, verified_content = self.llm_manager.check_hallucinations(
            content=content,
            user_input=user_response,
            context_text=context_text
        )
        
        # Speichere das Ergebnis der Qualitätsprüfung
        self.conversation_state["content_quality_checks"][section_id] = {
            "has_issues": has_issues,
            "confidence_score": advanced_check["confidence_score"],
            "suspicious_sections": advanced_check["suspicious_sections"]
        }
        
        # Speichere den generierten Inhalt
        self.conversation_state["generated_content"][section_id] = verified_content
    
    def generate_script(self) -> Dict[str, Any]:
        """
        Generiert den finalen E-Learning-Kurs.
        
        Returns:
            Kurs als Dictionary
        """
        # Erstelle ein Skript aus den generierten Inhalten
        script = self.template_manager.create_script_from_responses(
            self.conversation_state["generated_content"], 
            self.conversation_state["context_info"]
        )
        
        return script
    
    def get_script_summary(self) -> str:
        """
        Erstellt eine Zusammenfassung des generierten Kurses im Format des Beispiel-Skripts.
        
        Returns:
            Zusammenfassung als String
        """
        # Generiere den Kurs
        script = self.generate_script()
        
        # Erstelle eine Zusammenfassung im Format des Beispiel-Skripts
        organization = self.conversation_state["context_info"].get(
            "Für welche Art von Organisation erstellen wir den E-Learning-Kurs (z.B. Krankenhaus, Bank, Behörde)?", "")
        
        summary = f"# {script.get('title', 'E-Learning-Kurs zur Informationssicherheit')}\n\n"
        
        if "description" in script:
            summary += f"{script['description']}\n\n"
        
        # Formatiere die Abschnitte im gewünschten Tabellenformat
        for section in script["sections"]:
            title = section["title"]
            content = section.get("content", "Kein Inhalt verfügbar.")
            
            summary += f"## {title}\n\n"
            summary += f"{content}\n\n"
            
            # Füge Hinweis zur Qualitätsprüfung hinzu, falls relevant
            section_id = section["id"]
            if section_id in self.conversation_state["content_quality_checks"]:
                quality_check = self.conversation_state["content_quality_checks"][section_id]
                if quality_check["has_issues"]:
                    summary += "*Hinweis: Dieser Abschnitt wurde nach der Qualitätsprüfung überarbeitet.*\n\n"
        
        # Füge abschließende Nachricht hinzu, ähnlich dem Beispiel-Skript
        summary += "Sie können das Wissen jetzt bei Ihrer Arbeit umsetzen. Dadurch steigern Sie das\n"
        summary += f"Sicherheitsbewusstsein in {organization} und Sie schützen sich, Ihre Kolleginnen\n"
        summary += "und Kollegen und die gesamte Organisation.\n\n"
        summary += "Gut gemacht!\n"
        
        return summary
    
    def generate_html_script(self) -> str:
        """
        Generiert eine HTML-Version des Skripts im Format des Beispiel-Skripts.
        
        Returns:
            HTML-formatiertes Skript
        """
        script = self.generate_script()
        organization = self.conversation_state["context_info"].get(
            "Für welche Art von Organisation erstellen wir den E-Learning-Kurs (z.B. Krankenhaus, Bank, Behörde)?", "")
        
        html = f"""<!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>{html.escape(script.get('title', 'E-Learning-Kurs zur Informationssicherheit'))}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #2c3e50; }}
            .section {{ margin-bottom: 20px; }}
            .section-title {{ background-color: #f5f5f5; padding: 10px; font-weight: bold; }}
            .section-content {{ padding: 10px; }}
            .footer {{ margin-top: 30px; border-top: 1px solid #ddd; padding-top: 15px; }}
        </style>
    </head>
    <body>
        <h1>{html.escape(script.get('title', 'E-Learning-Kurs zur Informationssicherheit'))}</h1>
        <p>{html.escape(script.get('description', ''))}</p>
    """
        
        # Füge die Abschnitte hinzu
        for section in script["sections"]:
            title = section["title"]
            content = section.get("content", "Kein Inhalt verfügbar.")
            
            html += f"""    <div class="section">
            <div class="section-title">{html.escape(title)}</div>"""
            
            # Process content with backslashes outside the f-string
            escaped_content = html.escape(content).replace('\n', '<br>')
            html += f"""
            <div class="section-content">{escaped_content}</div>
        </div>
    """
        
        # Füge den Abschluss hinzu
        html += f"""    <div class="footer">
            <p>Sie können das Wissen jetzt bei Ihrer Arbeit umsetzen. Dadurch steigern Sie das 
            Sicherheitsbewusstsein in {html.escape(organization)} und Sie schützen sich, Ihre Kolleginnen 
            und Kollegen und die gesamte Organisation.</p>
            <p>Gut gemacht!</p>
        </div>
    </body>
    </html>"""
        
        return html
    
    def save_script(self, output_path: str, format: str = "txt") -> None:
        """
        Speichert den generierten Kurs in einer Datei.
        
        Args:
            output_path: Pfad zur Ausgabedatei
            format: Format der Ausgabe ("txt", "json" oder "html")
        """
        try:
            if format == "json":
                script = self.generate_script()
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(script, f, ensure_ascii=False, indent=2)
            elif format == "html":
                html_script = self.generate_html_script()
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(html_script)
            else:  # Standard: txt
                script_summary = self.get_script_summary()
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(script_summary)
            
            logger.info(f"E-Learning-Kurs gespeichert: {output_path}")
        except Exception as e:
            logger.error(f"Fehler beim Speichern des E-Learning-Kurses: {e}")


class ELearningCourseGenerator:
    """
    Hauptklasse, die alle Komponenten des E-Learning-Kurs-Generators integriert.
    """
    
    def __init__(self, config_path: str = "./config.json"):
        """
        Initialisiert den ELearningCourseGenerator.
        
        Args:
            config_path: Pfad zur Konfigurationsdatei
        """
        # Lade die Konfiguration
        self.config = self.load_config(config_path)
        
        # Erstelle die erforderlichen Verzeichnisse
        self.create_directories()
        
        # Initialisiere die Komponenten
        self.document_processor = DocumentProcessor(
            documents_dir=self.config["documents_dir"],
            chunk_size=self.config["chunk_size"],
            chunk_overlap=self.config["chunk_overlap"]
        )
        
        self.vector_store_manager = VectorStoreManager(
            persist_directory=self.config["vectorstore_dir"]
        )
        
        self.llm_manager = LLMManager(
            model_name=self.config["model_name"]
        )
        
        self.template_manager = TemplateManager(
            template_path=self.config.get("template_path")
        )
        
        self.dialog_manager = None
        
        # Statistik für die Evaluierung
        self.generated_scripts_count = 0
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Lädt die Konfiguration aus einer JSON-Datei.
        
        Args:
            config_path: Pfad zur Konfigurationsdatei
            
        Returns:
            Konfiguration als Dictionary
        """
        default_config = {
            "documents_dir": "./data/documents",
            "vectorstore_dir": "./data/faiss_index",  # Geändert für FAISS
            "output_dir": "./data/output",
            "model_name": "llama3:8b",
            "chunk_size": 1000,
            "chunk_overlap": 200
        }
        
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            
            # Füge fehlende Standardwerte hinzu
            for key, value in default_config.items():
                if key not in config:
                    config[key] = value
            
            return config
        except Exception as e:
            logger.warning(f"Fehler beim Laden der Konfiguration: {e}. Verwende Standardkonfiguration.")
            return default_config
    
    def create_directories(self) -> None:
        """Erstellt die erforderlichen Verzeichnisse, falls sie nicht existieren."""
        directories = [
            self.config["documents_dir"],
            os.path.dirname(self.config["vectorstore_dir"]),
            self.config["output_dir"]
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def setup(self) -> None:
        """
        Richtet den Generator ein, indem Dokumente geladen und die Vektordatenbank erstellt wird.
        """
        # Versuche, eine bestehende Vektordatenbank zu laden
        database_loaded = self.vector_store_manager.load_vectorstore()
        
        if not database_loaded:
            # Wenn keine Datenbank existiert, erstelle eine neue
            documents = self.document_processor.load_documents()
            processed_docs = self.document_processor.process_documents(documents)
            self.vector_store_manager.create_vectorstore(processed_docs)
        
        # Initialisiere den Dialog-Manager
        self.dialog_manager = TemplateGuidedDialog(
            template_manager=self.template_manager,
            llm_manager=self.llm_manager,
            vector_store_manager=self.vector_store_manager
        )
    
    def start_conversation(self) -> str:
        """
        Startet das Gespräch mit dem Nutzer.
        
        Returns:
            Erste Frage für den Nutzer
        """
        if self.dialog_manager is None:
            self.setup()
        
        return self.dialog_manager.get_next_question()
    
    def process_user_input(self, user_input: str) -> str:
        """
        Verarbeitet die Eingabe des Nutzers und gibt die nächste Frage zurück.
        
        Args:
            user_input: Eingabe des Nutzers
            
        Returns:
            Nächste Frage oder Nachricht
        """
        if self.dialog_manager is None:
            self.setup()
        
        response = self.dialog_manager.process_user_response(user_input)
        
        # Prüfe, ob ein Skript generiert wurde
        if "Hier ist der entworfene E-Learning-Kurs" in response:
            self.generated_scripts_count += 1
            logger.info(f"Skript {self.generated_scripts_count} generiert!")
        
        return response
    
    def save_generated_script(self, filename: str = None, format: str = "txt") -> str:
        """
        Speichert den generierten Kurs und gibt den Pfad zurück.
        
        Args:
            filename: Name der Ausgabedatei (optional)
            format: Format der Ausgabe ("txt", "json" oder "html")
            
        Returns:
            Pfad zur gespeicherten Datei
        """
        if self.dialog_manager is None:
            raise ValueError("Dialog-Manager wurde nicht initialisiert")
        
        if filename is None:
            # Generiere einen Dateinamen basierend auf dem Kontext
            organization = self.dialog_manager.conversation_state["context_info"].get(
                "Für welche Art von Organisation erstellen wir den E-Learning-Kurs (z.B. Krankenhaus, Bank, Behörde)?", "")
            audience = self.dialog_manager.conversation_state["context_info"].get(
                "Welche Mitarbeitergruppen sollen geschult werden?", "")
            
            sanitized_organization = ''.join(c for c in organization if c.isalnum() or c.isspace()).strip().replace(' ', '_')
            sanitized_audience = ''.join(c for c in audience if c.isalnum() or c.isspace()).strip().replace(' ', '_')
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"elearning_{sanitized_organization}_{sanitized_audience}_{timestamp}.{format}"
        
        output_path = os.path.join(self.config["output_dir"], filename)
        self.dialog_manager.save_script(output_path, format)
        
        return output_path
    
    def reset_conversation(self) -> None:
        """
        Setzt die Konversation zurück, um einen neuen Kurs zu erstellen.
        """
        if self.dialog_manager is not None:
            # Erstelle einen neuen Dialog-Manager mit denselben Komponenten
            self.dialog_manager = TemplateGuidedDialog(
                template_manager=self.template_manager,
                llm_manager=self.llm_manager,
                vector_store_manager=self.vector_store_manager
            )
    
    def reindex_documents(self):
        """
        Löscht die bestehende Vektordatenbank und erstellt eine neue mit allen aktuellen Dokumenten.
        """
        logger.info("Starte Neuindexierung der Dokumente...")
        
        # Lade alle Dokumente neu
        documents = self.document_processor.load_documents()
        processed_docs = self.document_processor.process_documents(documents)
        
        # Lösche die alte Vektordatenbank, falls sie existiert
        if os.path.exists(self.config["vectorstore_dir"]):
            import shutil
            shutil.rmtree(self.config["vectorstore_dir"])
            os.makedirs(os.path.dirname(self.config["vectorstore_dir"]), exist_ok=True)
            logger.info(f"Alte Vektordatenbank gelöscht: {self.config['vectorstore_dir']}")
        
        # Erstelle neue Vektordatenbank
        self.vector_store_manager.create_vectorstore(processed_docs)
        logger.info(f"Neuindexierung abgeschlossen. {len(processed_docs)} Dokumente indexiert.")
        
        return len(processed_docs)


# Flask-API für den E-Learning-Kurs-Generator
from flask import Flask, request, jsonify, send_file, render_template
import tempfile
import threading
import queue
import os

# Erstelle Flask-App mit statischen Dateien und Templates
app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')

# Globale Variable für den Generator
generator = None
chat_queue = queue.Queue()
processing_thread = None
is_processing = False

def initialize_generator():
    """Initialisiert den ELearningCourseGenerator, falls noch nicht geschehen."""
    global generator
    if generator is None:
        generator = ELearningCourseGenerator()
        generator.setup()
    return generator

# Stelle sicher, dass die Verzeichnisse für die Benutzeroberfläche existieren
def ensure_ui_directories():
    """Erstellt die notwendigen Verzeichnisse für die UI, falls nicht vorhanden"""
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    os.makedirs("static/css", exist_ok=True)
    os.makedirs("static/js", exist_ok=True)
    
    # Erstelle die UI-Dateien, wenn sie nicht existieren
    create_ui_files()

def process_chat_queue():
    """Verarbeitet die Chat-Anfragen in der Warteschlange."""
    global is_processing
    is_processing = True
    while not chat_queue.empty():
        req_data, response_queue = chat_queue.get()
        try:
            gen = initialize_generator()
            user_input = req_data.get("message", "")
            response = gen.process_user_input(user_input)
            response_queue.put({"status": "success", "response": response})
        except Exception as e:
            logger.error(f"Fehler bei der Verarbeitung der Chat-Anfrage: {e}", exc_info=True)  # Fügt Stack-Trace hinzu
            response_queue.put({"status": "error", "message": str(e)})
        chat_queue.task_done()
    is_processing = False

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    API-Endpunkt für den Chat mit dem Chatbot.
    
    Erwartet ein JSON mit einem "message"-Feld.
    Gibt die Antwort des Chatbots zurück.
    """
    global processing_thread, is_processing
    
    req_data = request.json
    if not req_data or "message" not in req_data:
        return jsonify({"status": "error", "message": "Nachricht fehlt"}), 400
    
    response_queue = queue.Queue()
    chat_queue.put((req_data, response_queue))
    
    # Starte den Verarbeitungs-Thread, falls noch nicht aktiv
    if not is_processing:
        processing_thread = threading.Thread(target=process_chat_queue)
        processing_thread.daemon = True
        processing_thread.start()
    
    # Warte auf die Antwort
    response = response_queue.get(timeout=120)
    return jsonify(response)

@app.route('/api/start', methods=['GET'])
def start_conversation():
    """
    Startet eine neue Konversation mit dem Chatbot.
    
    Gibt die erste Frage des Chatbots zurück.
    """
    try:
        gen = initialize_generator()
        first_question = gen.start_conversation()
        return jsonify({"status": "success", "response": first_question})
    except Exception as e:
        logger.error(f"Fehler beim Starten der Konversation: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/reset', methods=['POST'])
def reset_conversation():
    """
    Setzt die Konversation zurück, um einen neuen Kurs zu erstellen.
    """
    try:
        gen = initialize_generator()
        gen.reset_conversation()
        first_question = gen.start_conversation()
        return jsonify({"status": "success", "response": first_question})
    except Exception as e:
        logger.error(f"Fehler beim Zurücksetzen der Konversation: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/download', methods=['GET'])
def download_script():
    """
    Lädt das generierte Skript herunter.
    
    Unterstützt die Formate 'txt', 'json' und 'html' als Query-Parameter.
    """
    try:
        gen = initialize_generator()
        format_type = request.args.get('format', 'txt')
        if format_type not in ['txt', 'json', 'html']:
            return jsonify({"status": "error", "message": "Ungültiges Format. Erlaubt sind: txt, json, html"}), 400
        
        # Erstelle eine temporäre Datei
        with tempfile.NamedTemporaryFile(suffix=f'.{format_type}', delete=False) as temp_file:
            temp_path = temp_file.name
        
        # Speichere das Skript in der temporären Datei
        script_path = gen.save_generated_script(filename=os.path.basename(temp_path), format=format_type)
        
        return send_file(script_path, as_attachment=True)
    except Exception as e:
        logger.error(f"Fehler beim Herunterladen des Skripts: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/reindex', methods=['POST'])
def reindex():
    """
    Indiziert die Dokumente neu.
    """
    try:
        gen = initialize_generator()
        doc_count = gen.reindex_documents()
        return jsonify({"status": "success", "message": f"{doc_count} Dokumente erfolgreich neu indexiert."})
    except Exception as e:
        logger.error(f"Fehler bei der Neuindexierung: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Einfacher Health-Check-Endpunkt.
    """
    return jsonify({"status": "ok", "message": "Der E-Learning-Kurs-Generator ist bereit."})

@app.route('/', methods=['GET'])
def home():
    """
    Rendert die Chat-Benutzeroberfläche für den E-Learning-Kurs-Generator.
    """
    return render_template('index.html')

@app.route('/api-docs', methods=['GET'])
def api_docs():
    """
    Dokumentation der API-Endpunkte.
    """
    return render_template('api_docs.html')

# Hilfsfunktion zum Erstellen der UI-Dateien
def create_ui_files():
    """Erstellt die UI-Dateien, wenn sie nicht existieren"""
    # Erstelle index.html
    if not os.path.exists("templates/index.html"):
        with open("templates/index.html", "w", encoding="utf-8") as f:
            f.write("""<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>E-Learning-Kurs-Generator</title>
    <link rel="stylesheet" href="/static/css/style.css">
</head>
<body>
    <div class="container">
        <div class="chat-container">
            <h1>E-Learning-Kurs-Generator für Informationssicherheit</h1>
            <div class="chat-box" id="chatBox">
                <!-- Chat-Nachrichten werden hier dynamisch hinzugefügt -->
            </div>
            <div class="input-area">
                <input type="text" id="userInput" placeholder="Ihre Antwort..." autofocus>
                <button id="sendButton">Senden</button>
            </div>
            <div class="action-buttons">
                <button id="resetButton">Neue Konversation</button>
                <div class="download-options">
                    <span>Skript herunterladen als:</span>
                    <button id="downloadTxtButton" class="download-button">TXT</button>
                    <button id="downloadJsonButton" class="download-button">JSON</button>
                    <button id="downloadHtmlButton" class="download-button">HTML</button>
                </div>
            </div>
        </div>
        <div class="script-preview">
            <h2>Skript-Vorschau</h2>
            <div class="script-content" id="scriptContent">
                <p class="preview-placeholder">Hier wird die Vorschau des generierten Skripts angezeigt, sobald es erstellt wurde.</p>
            </div>
        </div>
    </div>
    <script src="/static/js/main.js"></script>
</body>
</html>""")
    
    # Erstelle style.css
    if not os.path.exists("static/css/style.css"):
        with open("static/css/style.css", "w", encoding="utf-8") as f:
            f.write("""* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body {
    font-family: Arial, sans-serif;
    line-height: 1.6;
    background-color: #f5f7f9;
    color: #333;
}

.container {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    max-width: 1400px;
    margin: 20px auto;
    padding: 0 20px;
}

h1 {
    color: #2c3e50;
    margin-bottom: 20px;
    font-size: 1.8rem;
    grid-column: span 2;
}

.chat-container {
    background-color: #fff;
    border-radius: 8px;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
    padding: 20px;
    display: flex;
    flex-direction: column;
    height: 85vh;
}

.chat-box {
    flex-grow: 1;
    overflow-y: auto;
    padding: 10px;
    border: 1px solid #e1e4e8;
    border-radius: 5px;
    margin-bottom: 15px;
    background-color: #fafafa;
}

.chat-message {
    margin-bottom: 10px;
    padding: 10px;
    border-radius: 5px;
    max-width: 90%;
    word-wrap: break-word;
}

.user-message {
    background-color: #dcf8c6;
    align-self: flex-end;
    margin-left: auto;
}

.assistant-message {
    background-color: #e9eaee;
    align-self: flex-start;
}

.input-area {
    display: flex;
    margin-bottom: 15px;
}

#userInput {
    flex-grow: 1;
    padding: 10px;
    border: 1px solid #ddd;
    border-radius: 5px 0 0 5px;
    font-size: 16px;
}

#sendButton {
    padding: 10px 15px;
    background-color: #4CAF50;
    color: white;
    border: none;
    border-radius: 0 5px 5px 0;
    cursor: pointer;
    font-size: 16px;
}

#sendButton:hover {
    background-color: #45a049;
}

.action-buttons {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

#resetButton {
    padding: 8px 15px;
    background-color: #f44336;
    color: white;
    border: none;
    border-radius: 5px;
    cursor: pointer;
}

#resetButton:hover {
    background-color: #d32f2f;
}

.download-options {
    display: flex;
    align-items: center;
    gap: 10px;
}

.download-button {
    padding: 8px 12px;
    background-color: #3498db;
    color: white;
    border: none;
    border-radius: 5px;
    cursor: pointer;
}

.download-button:hover {
    background-color: #2980b9;
}

.script-preview {
    background-color: #fff;
    border-radius: 8px;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
    padding: 20px;
    height: 85vh;
    display: flex;
    flex-direction: column;
}

.script-preview h2 {
    color: #2c3e50;
    margin-bottom: 15px;
    font-size: 1.5rem;
}

.script-content {
    overflow-y: auto;
    padding: 15px;
    border: 1px solid #e1e4e8;
    border-radius: 5px;
    background-color: #fafafa;
    flex-grow: 1;
    white-space: pre-wrap;
    font-size: 14px;
    line-height: 1.7;
}

.preview-placeholder {
    color: #6c757d;
    font-style: italic;
}

button:disabled {
    background-color: #cccccc;
    cursor: not-allowed;
}

@media (max-width: 900px) {
    .container {
        grid-template-columns: 1fr;
    }
    
    h1 {
        grid-column: span 1;
    }
    
    .chat-container, .script-preview {
        height: auto;
        max-height: 70vh;
    }
}""")
    
    # Erstelle main.js
    if not os.path.exists("static/js/main.js"):
        with open("static/js/main.js", "w", encoding="utf-8") as f:
            f.write("""document.addEventListener('DOMContentLoaded', function() {
    const chatBox = document.getElementById('chatBox');
    const userInput = document.getElementById('userInput');
    const sendButton = document.getElementById('sendButton');
    const resetButton = document.getElementById('resetButton');
    const downloadTxtButton = document.getElementById('downloadTxtButton');
    const downloadJsonButton = document.getElementById('downloadJsonButton');
    const downloadHtmlButton = document.getElementById('downloadHtmlButton');
    const scriptContent = document.getElementById('scriptContent');
    
    let scriptGenerated = false;
    
    // Setze Download-Buttons initial deaktiviert
    downloadTxtButton.disabled = true;
    downloadJsonButton.disabled = true;
    downloadHtmlButton.disabled = true;
    
    // Starte die Konversation beim Laden der Seite
    startConversation();
    
    // Event-Listener für Senden-Button und Enter-Taste
    sendButton.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
    
    // Event-Listener für Reset-Button
    resetButton.addEventListener('click', resetConversation);
    
    // Event-Listener für Download-Buttons
    downloadTxtButton.addEventListener('click', () => downloadScript('txt'));
    downloadJsonButton.addEventListener('click', () => downloadScript('json'));
    downloadHtmlButton.addEventListener('click', () => downloadScript('html'));
    
    // Funktion zum Starten einer neuen Konversation
    function startConversation() {
        fetch('/api/start')
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    addMessage(data.response, 'assistant');
                    scriptGenerated = false;
                    updateButtonStates();
                }
            })
            .catch(error => {
                console.error('Fehler beim Starten der Konversation:', error);
                addMessage('Es gab einen Fehler beim Verbinden mit dem Server. Bitte versuchen Sie es später erneut.', 'assistant');
            });
    }
    
    // Funktion zum Senden einer Nachricht
    function sendMessage() {
        const message = userInput.value.trim();
        if (message === '') return;
        
        // Nachricht des Benutzers anzeigen
        addMessage(message, 'user');
        
        // Eingabefeld leeren und Button deaktivieren
        userInput.value = '';
        sendButton.disabled = true;
        
        // Nachricht an den Server senden
        fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: message
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                addMessage(data.response, 'assistant');
                
                // Prüfe, ob das Skript generiert wurde
                if (data.response.includes('Hier ist der entworfene E-Learning-Kurs')) {
                    scriptGenerated = true;
                    updateScriptPreview(data.response);
                    updateButtonStates();
                }
            } else {
                addMessage('Es gab einen Fehler bei der Verarbeitung Ihrer Nachricht.', 'assistant');
            }
            sendButton.disabled = false;
        })
        .catch(error => {
            console.error('Fehler beim Senden der Nachricht:', error);
            addMessage('Es gab einen Fehler bei der Kommunikation mit dem Server.', 'assistant');
            sendButton.disabled = false;
        });
    }
    
    // Funktion zum Zurücksetzen der Konversation
    function resetConversation() {
        fetch('/api/reset', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                // Chat-Box leeren
                chatBox.innerHTML = '';
                // Skript-Vorschau zurücksetzen
                scriptContent.innerHTML = '<p class="preview-placeholder">Hier wird die Vorschau des generierten Skripts angezeigt, sobald es erstellt wurde.</p>';
                
                // Erste Nachricht anzeigen
                addMessage(data.response, 'assistant');
                
                scriptGenerated = false;
                updateButtonStates();
            }
        })
        .catch(error => {
            console.error('Fehler beim Zurücksetzen der Konversation:', error);
        });
    }
    
    // Funktion zum Herunterladen des Skripts
    function downloadScript(format) {
        if (!scriptGenerated) return;
        
        window.open(`/api/download?format=${format}`, '_blank');
    }
    
    // Funktion zum Hinzufügen einer Nachricht zur Chat-Box
    function addMessage(message, sender) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('chat-message', `${sender}-message`);
        
        // Verarbeite Markdown-ähnliche Formatierung
        let formattedMessage = message.replace(/\\n/g, '<br>');
        formattedMessage = formattedMessage.replace(/#{1,6} (.+?)\\n/g, '<h3>$1</h3>');
        formattedMessage = formattedMessage.replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>');
        
        messageElement.innerHTML = formattedMessage;
        chatBox.appendChild(messageElement);
        
        // Scrolle zum Ende der Chat-Box
        chatBox.scrollTop = chatBox.scrollHeight;
    }
    
    // Funktion zum Extrahieren und Anzeigen des Skripts in der Vorschau
    function updateScriptPreview(response) {
        const scriptStartMarker = 'Hier ist der entworfene E-Learning-Kurs';
        const startIndex = response.indexOf(scriptStartMarker);
        
        if (startIndex !== -1) {
            const scriptContent = response.substring(startIndex + scriptStartMarker.length).trim();
            document.getElementById('scriptContent').innerHTML = `<pre>${scriptContent}</pre>`;
        }
    }
    
    // Funktion zum Aktualisieren der Button-Zustände
    function updateButtonStates() {
        downloadTxtButton.disabled = !scriptGenerated;
        downloadJsonButton.disabled = !scriptGenerated;
        downloadHtmlButton.disabled = !scriptGenerated;
    }
});""")

    # Erstelle die API-Dokumentationsseite
    if not os.path.exists("templates/api_docs.html"):
        with open("templates/api_docs.html", "w", encoding="utf-8") as f:
            f.write("""<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>E-Learning-Kurs-Generator API Dokumentation</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        h1 { color: #2c3e50; }
        .endpoint { margin-bottom: 20px; background-color: #f5f5f5; padding: 15px; border-radius: 5px; }
        .method { font-weight: bold; color: #27ae60; }
        .url { font-family: monospace; }
        .description { margin-top: 10px; }
        .back-link { margin-top: 30px; display: block; }
    </style>
</head>
<body>
    <h1>E-Learning-Kurs-Generator API</h1>
    <p>Willkommen bei der API-Dokumentation des E-Learning-Kurs-Generators für Informationssicherheit.</p>
    
    <h2>Verfügbare Endpunkte:</h2>
    
    <div class="endpoint">
        <div><span class="method">GET</span> <span class="url">/api/start</span></div>
        <div class="description">Startet eine neue Konversation und gibt die erste Frage zurück.</div>
    </div>
    
    <div class="endpoint">
        <div><span class="method">POST</span> <span class="url">/api/chat</span></div>
        <div class="description">Sendet eine Nachricht an den Chatbot und erhält eine Antwort.</div>
    </div>
    
    <div class="endpoint">
        <div><span class="method">POST</span> <span class="url">/api/reset</span></div>
        <div class="description">Setzt die Konversation zurück, um einen neuen Kurs zu erstellen.</div>
    </div>
    
    <div class="endpoint">
        <div><span class="method">GET</span> <span class="url">/api/download?format=txt</span></div>
        <div class="description">Lädt das generierte Skript im gewünschten Format herunter (txt, json oder html).</div>
    </div>
    
    <div class="endpoint">
        <div><span class="method">POST</span> <span class="url">/api/reindex</span></div>
        <div class="description">Indiziert die Dokumente neu (z.B. nach Hinzufügen neuer Dokumente).</div>
    </div>
    
    <div class="endpoint">
        <div><span class="method">GET</span> <span class="url">/api/health</span></div>
        <div class="description">Überprüft, ob der Server läuft.</div>
    </div>
    
    <a href="/" class="back-link">Zurück zum Chat</a>
</body>
</html>""")

# Main-Funktion für den Start des Servers
def main():
    """
    Hauptfunktion zum Starten des E-Learning-Kurs-Generators.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="E-Learning-Kurs-Generator für Informationssicherheit")
    parser.add_argument("--config", type=str, default="./config.json", help="Pfad zur Konfigurationsdatei")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host-Adresse für den API-Server")
    parser.add_argument("--port", type=int, default=8000, help="Port für den API-Server")
    parser.add_argument("--reindex", action="store_true", help="Dokumente neu indexieren beim Start")
    
    args = parser.parse_args()
    
    global generator
    generator = ELearningCourseGenerator(config_path=args.config)
    
    # Stelle sicher, dass die Verzeichnisse und Dateien für die UI existieren
    ensure_ui_directories()
    
    # Neuindexierung, falls angefordert
    if args.reindex:
        doc_count = generator.reindex_documents()
        print(f"{doc_count} Dokumente erfolgreich neu indexiert.")
    
    # Initialisiere die Komponenten für schnelleren ersten Zugriff
    generator.setup()
    
    # Starte den API-Server
    print(f"Starte API-Server auf http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()