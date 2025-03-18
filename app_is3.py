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
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
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
    
    def __init__(self, persist_directory: str = "./data/chroma_db"):
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
        # Erstelle eine neue Vektordatenbank
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_metadata={"hnsw:space": "cosine"}
        )
        
        # Persistiere die Datenbank
        self.vectorstore.persist()
        
        logger.info(f"Vektordatenbank erstellt mit {len(documents)} Dokumenten")
    
    def load_vectorstore(self) -> bool:
        """
        Lädt eine bestehende Vektordatenbank.
        
        Returns:
            True, wenn die Datenbank geladen wurde, False sonst
        """
        try:
            # Versuche, die Vektordatenbank zu laden
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            
            logger.info("Bestehende Vektordatenbank geladen")
            return True
        except Exception as e:
            logger.error(f"Fehler beim Laden der Vektordatenbank: {e}")
            return False
    
    def get_retriever(self, search_type: str = "mmr", search_kwargs: Dict[str, Any] = None):
        """
        Gibt einen Retriever für die Vektordatenbank zurück.
        
        Args:
            search_type: Art der Suche (z.B. "mmr" für Maximum Marginal Relevance)
            search_kwargs: Zusätzliche Suchparameter
            
        Returns:
            Retriever-Objekt
        """
        if self.vectorstore is None:
            raise ValueError("Vektordatenbank wurde nicht initialisiert")
        
        if search_kwargs is None:
            search_kwargs = {"k": 5}
        
        return self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
    
    def retrieve_documents(self, query: str, filter: Dict[str, Any] = None, k: int = 5) -> List[Document]:
        """
        Führt eine Suche in der Vektordatenbank durch.
        
        Args:
            query: Suchanfrage
            filter: Filter für die Suche
            k: Anzahl der zurückzugebenden Dokumente
            
        Returns:
            Liste von gefundenen Document-Objekten
        """
        if self.vectorstore is None:
            raise ValueError("Vektordatenbank wurde nicht initialisiert")
        
        results = self.vectorstore.similarity_search(
            query=query,
            filter=filter,
            k=k
        )
        
        return results
    
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
        for section in self.template["sections"]:
            if section["id"] == section_id:
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
        for section in self.template["sections"]:
            if section["id"] not in completed_sections:
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
        
        # Finde den nächsten nicht bearbeiteten Abschnitt
        next_section = self.template_manager.get_next_section(self.conversation_state["completed_sections"])
        
        if next_section is None:
            # Alle Abschnitte wurden bearbeitet
            self.conversation_state["current_step"] = "review"
            return self.get_next_question()
        
        # Prüfe, ob wir zu einem neuen Abschnitt wechseln
        current_section = self.conversation_state.get("current_section")
        is_new_section = current_section != next_section["id"]
        
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
            logger.error(f"Fehler bei der Fragengenerierung für {section_title}: {e}")
            
            # Erhöhe Fehlerzähler
            self.conversation_state["question_error_count"] += 1
            
            # Bei zu vielen Fehlern zum nächsten Abschnitt wechseln
            if self.conversation_state["question_error_count"] > 2:
                logger.warning(f"Zu viele Fehler bei {section_title}, überspringe Abschnitt")
                self.conversation_state["completed_sections"].append(section_id)
                return self.get_next_template_question()
            
            # Verwende vordefinierte Frage als Fallback
            question = predefined_questions.get(section_type, f"Können Sie mir mehr über {section_title} in Ihrem Arbeitsalltag erzählen?")
        
        # Aktualisiere den Gesprächszustand
        self.conversation_state["current_section"] = section_id
        
        # Füge Übergangsinfo hinzu bei neuem Abschnitt
        if is_new_section:
            return f"Nun kommen wir zum Abschnitt '{section_title}'.\n\n{question}"
        
        return question
    
    def generate_retrieval_queries(self, section_title: str, section_id: str) -> List[str]:
        """
        Erzeugt Retrieval-Anfragen für einen Abschnitt des Templates.
        
        Args:
            section_title: Titel des Abschnitts
            section_id: ID des Abschnitts
            
        Returns:
            Liste von Retrieval-Anfragen
        """
        # Basis-Anfrage aus dem Abschnittstitel
        base_query = section_title
        
        # Kontextspezifische Anfragen
        organization = self.conversation_state["context_info"].get(
            "Für welche Art von Organisation erstellen wir den E-Learning-Kurs (z.B. Krankenhaus, Bank, Behörde)?", "")
        audience = self.conversation_state["context_info"].get(
            "Welche Mitarbeitergruppen sollen geschult werden?", "")
        compliance = self.conversation_state["context_info"].get(
            "Gibt es spezifische Compliance-Anforderungen oder Branchenstandards, die berücksichtigt werden müssen?", "")
        
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
        section_query = section_templates.get(section_id, "")
        
        # Compliance-spezifische Anfrage hinzufügen, falls vorhanden
        compliance_query = ""
        if compliance and compliance.strip():
            compliance_query = f"Informationssicherheit {compliance} {organization}"
        
        # Spezifische Anfragen für bestimmte Abschnitte
        specific_queries = []
        if section_id == "threat_awareness":
            specific_queries.append(f"Alltägliche Situationen Informationssicherheit {organization}")
        elif section_id == "threat_identification":
            specific_queries.append(f"Phishing Erkennung Merkmale {organization}")
        elif section_id == "threat_impact_assessment":
            specific_queries.append(f"Konsequenzen Datenverlust Cyberangriff {organization}")
        elif section_id == "tactic_choice":
            specific_queries.append(f"Schutzmaßnahmen Verdächtige E-Mails {organization}")
        elif section_id == "tactic_justification":
            specific_queries.append(f"Warum E-Mail-Sicherheit wichtig {organization}")
        elif section_id == "tactic_mastery":
            specific_queries.append(f"Schritte Überprüfung Verdächtige E-Mails {organization}")
        elif section_id == "tactic_check_follow_up":
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
            
            if max_questions_reached or self.is_response_adequate(response):
                # Speichere die Antwort für den aktuellen Abschnitt
                current_section = self.conversation_state["current_section"]
                self.conversation_state["section_responses"][current_section] = response
                
                # Generiere Inhalt für diesen Abschnitt
                self._generate_section_content(current_section)
                
                self.conversation_state["completed_sections"].append(current_section)
                
                # Setze den Fragenzähler für den nächsten Abschnitt zurück
                self.conversation_state["current_section_question_count"] = 0
            else:
                # Fordere eine detailliertere Antwort an
                return self.generate_followup_question(response)
        
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
            
            if "threat" in current_section_id:
                relevant_terms.extend(["gefahr", "risiko", "bedrohung", "sicherheit", "schaden"])
            if "tactic" in current_section_id:
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
        # Hole die Antwort des Nutzers
        user_response = self.conversation_state["section_responses"][section_id]
        section = self.template_manager.get_section_by_id(section_id)
        
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
        retrieval_queries.extend(self.generate_retrieval_queries(section["title"], section_id))
        
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
            "vectorstore_dir": "./data/vectorstore",
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
            self.config["vectorstore_dir"],
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
            os.makedirs(self.config["vectorstore_dir"], exist_ok=True)
            logger.info(f"Alte Vektordatenbank gelöscht: {self.config['vectorstore_dir']}")
        
        # Erstelle neue Vektordatenbank
        self.vector_store_manager.create_vectorstore(processed_docs)
        logger.info(f"Neuindexierung abgeschlossen. {len(processed_docs)} Dokumente indexiert.")
        
        return len(processed_docs)


# Streamlit-Benutzeroberfläche für den E-Learning-Kurs-Generator
def create_streamlit_app():
    """
    Erstellt eine Streamlit-Anwendung für den E-Learning-Kurs-Generator.
    """
    st.set_page_config(page_title="E-Learning-Kurs-Generator", page_icon="📚", layout="wide")
    
    st.title("E-Learning-Kurs-Generator für Informationssicherheit")
    st.markdown("### Erstellen Sie maßgeschneiderte Schulungsinhalte für Informationssicherheit")
    
    # Sidebar für Einstellungen und Aktionen
    with st.sidebar:
        st.header("Aktionen")
        
        # Download-Format auswählen
        format_options = {
            "txt": "Text (.txt)",
            "json": "JSON (.json)",
            "html": "HTML (.html)"
        }
        selected_format = st.selectbox(
            "Ausgabeformat wählen",
            options=list(format_options.keys()),
            format_func=lambda x: format_options[x],
            index=0
        )
        
        if st.button("Dokumente neu indexieren"):
            with st.spinner("Indexiere Dokumente..."):
                if "bot" not in st.session_state:
                    st.session_state.bot = ELearningCourseGenerator()
                doc_count = st.session_state.bot.reindex_documents()
                st.success(f"{doc_count} Dokumente erfolgreich neu indexiert!")
                
                # Starte die Konversation neu
                st.session_state.messages = [
                    {"role": "assistant", "content": st.session_state.bot.start_conversation()}
                ]
                st.session_state.script_generated = False
                st.rerun()
                
        # Zähler für generierte Skripte anzeigen
        if "bot" in st.session_state:
            st.metric("Generierte Skripte", st.session_state.bot.generated_scripts_count)
            
            # Prüfe, ob 5 oder mehr Skripte generiert wurden
            if st.session_state.bot.generated_scripts_count >= 5:
                st.success("✅ Anforderung erfüllt: 5 verschiedene Skripte wurden generiert!")
    
    # Hauptbereich aufteilen
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.subheader("Konversation mit dem Chatbot")
        
        # Initialisiere den Bot im Session State, falls nicht vorhanden
        if "bot" not in st.session_state:
            st.session_state.bot = ELearningCourseGenerator()
            st.session_state.messages = [
                {"role": "assistant", "content": st.session_state.bot.start_conversation()}
            ]
            st.session_state.script_generated = False
        
        # Container für den Chatverlauf mit fester Höhe
        chat_container = st.container(height=500)
        
        # Zeige den Chatverlauf im Container an
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
        
        # Eingabefeld für den Nutzer
        user_input = st.chat_input("Ihre Antwort")
        
        if user_input:
            # Füge die Nutzereingabe zum Chatverlauf hinzu
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            with st.chat_message("user"):
                st.write(user_input)
            
            # Verarbeite die Nutzereingabe
            with st.spinner("Verarbeite Eingabe..."):
                bot_response = st.session_state.bot.process_user_input(user_input)
            
            # Füge die Antwort des Bots zum Chatverlauf hinzu
            st.session_state.messages.append({"role": "assistant", "content": bot_response})
            
            with st.chat_message("assistant"):
                st.write(bot_response)
            
            # Prüfe, ob der Kurs generiert wurde
            if "Hier ist der entworfene E-Learning-Kurs" in bot_response:
                st.session_state.script_generated = True
                
                # Speichere den Kurs und biete ihn zum Download an
                script_path = st.session_state.bot.save_generated_script(format=selected_format)
                
                with open(script_path, "r", encoding="utf-8") as f:
                    script_content = f.read()
                
                st.download_button(
                    label=f"E-Learning-Kurs herunterladen ({format_options[selected_format]})",
                    data=script_content,
                    file_name=os.path.basename(script_path),
                    mime="text/plain"
                )
                
                # Button zum Erstellen eines neuen Kurses
                if st.button("Neuen E-Learning-Kurs erstellen"):
                    st.session_state.bot.reset_conversation()
                    st.session_state.messages = [
                        {"role": "assistant", "content": st.session_state.bot.start_conversation()}
                    ]
                    st.session_state.script_generated = False
                    st.rerun()
            
            st.rerun()
    
    with col2:
        st.subheader("Vorschau des generierten Skripts")
        
        # Zeige eine Vorschau des Skripts, wenn es generiert wurde
        if "script_generated" in st.session_state and st.session_state.script_generated:
            # Hole das Skript im gewünschten Format
            if selected_format == "html":
                html_script = st.session_state.bot.dialog_manager.generate_html_script()
                st.components.v1.html(html_script, height=600, scrolling=True)
            else:
                script_text = st.session_state.bot.dialog_manager.get_script_summary()
                st.text_area("Skript-Inhalt", script_text, height=600)
        else:
            st.info("Hier wird die Vorschau des generierten Skripts angezeigt, sobald es erstellt wurde.")


# Hauptfunktion
def main():
    """
    Hauptfunktion zum Starten des E-Learning-Kurs-Generators.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="E-Learning-Kurs-Generator für Informationssicherheit")
    parser.add_argument("--config", type=str, default="./config.json", help="Pfad zur Konfigurationsdatei")
    parser.add_argument("--mode", type=str, choices=["cli", "web"], default="web", help="Ausführungsmodus (CLI oder Web)")
    parser.add_argument("--reindex", action="store_true", help="Dokumente neu indexieren")
    parser.add_argument("--format", type=str, choices=["txt", "json", "html"], default="txt", help="Ausgabeformat (Text, JSON oder HTML)")
    
    args = parser.parse_args()
    
    # Neuindexierung, falls angefordert
    if args.reindex:
        bot = ELearningCourseGenerator(config_path=args.config)
        doc_count = bot.reindex_documents()
        print(f"{doc_count} Dokumente erfolgreich neu indexiert.")
        if args.mode == "cli" and not args.mode:  # Falls nur die Neuindexierung gewünscht ist
            return
    
    if args.mode == "web":
        # Starte die Streamlit-Anwendung
        create_streamlit_app()
    else:
        # Starte den CLI-Modus
        bot = ELearningCourseGenerator(config_path=args.config)
        bot.setup()
        
        print("\n=== E-Learning-Kurs-Generator für Informationssicherheit ===")
        print("Beenden Sie das Programm mit 'exit', 'quit' oder 'q'")
        print(bot.start_conversation())
        
        while True:
            user_input = input("\n> ")
            
            if user_input.lower() in ["exit", "quit", "q"]:
                break
            
            response = bot.process_user_input(user_input)
            print(response)
            
            if "Hier ist der entworfene E-Learning-Kurs" in response:
                script_path = bot.save_generated_script(format=args.format)
                print(f"\nE-Learning-Kurs gespeichert unter: {script_path}")
                
                # Frage, ob ein neuer Kurs erstellt werden soll
                new_course = input("\nMöchten Sie einen neuen E-Learning-Kurs erstellen? (j/n): ")
                if new_course.lower() in ["j", "ja", "y", "yes"]:
                    bot.reset_conversation()
                    print("\n=== Neuer E-Learning-Kurs ===")
                    print(bot.start_conversation())
                else:
                    break


if __name__ == "__main__":
    main()