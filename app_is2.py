import os
import json
import copy
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path
from datetime import datetime

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
        # Einfache Heuristik: Suche nach bestimmten Schlüsselwörtern
        content_lower = content.lower()
        
        if any(term in content_lower for term in ["lernziel", "learning objective", "ziel"]):
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
        # Initialisiere das LLM
        self.llm = Ollama(model=model_name)
        
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
        Du bist ein freundlicher Berater, der Unternehmen bei der Erstellung von E-Learning-Kursen zu Informationssicherheit unterstützt.
        Deine Aufgabe ist es, Fragen zu stellen, die einem nicht-technischen Kunden helfen, über die Prozesse und den Kontext seines Unternehmens 
        zu sprechen. Stelle KEINE direkten Fragen zu Informationssicherheitsmaßnahmen oder technischen Details, da der Kunde damit nicht vertraut ist.
        
        Formuliere eine freundliche, leicht verständliche Frage zu folgendem Thema:
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
        """
        
        return PromptTemplate(
            template=template,
            input_variables=["section_title", "section_description", "context_text", 
                             "organization", "audience"]
        )
    
    def _create_content_generation_prompt(self) -> PromptTemplate:
        """
        Erstellt ein Prompt-Template für die Inhaltsgenerierung.
        
        Returns:
            PromptTemplate-Objekt
        """
        template = """
        Erstelle den Inhalt für den Abschnitt "{section_title}" eines E-Learning-Kurses zur Informationssicherheit.
        
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
        
        Der Inhalt sollte didaktisch gut aufbereitet sein, relevante Beispiele enthalten und für die angegebene Zielgruppe verständlich sein.
        Vermeide unnötig komplizierte Fachbegriffe und konzentriere dich auf praktische, anwendbare Informationen.
        
        Gib nur den fertigen Inhalt zurück, keine zusätzlichen Erklärungen.
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
        response = self.chains["question_generation"].run({
            "section_title": section_title,
            "section_description": section_description,
            "context_text": context_text,
            "organization": organization,
            "audience": audience
        })
        
        return response.strip()
    
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


class TemplateManager:
    """
    Verwaltet das Template für den E-Learning-Kurs.
    """
    
    def __init__(self, template_path: str):
        """
        Initialisiert den TemplateManager.
        
        Args:
            template_path: Pfad zur Template-Datei
        """
        self.template_path = template_path
        self.template = self.load_template()
    
    def load_template(self) -> Dict[str, Any]:
        """
        Lädt das Template aus der angegebenen Datei.
        
        Returns:
            Template als Dictionary
        """
        try:
            with open(self.template_path, "r", encoding="utf-8") as f:
                template = json.load(f)
            
            logger.info(f"Template geladen: {self.template_path}")
            return template
        except Exception as e:
            logger.error(f"Fehler beim Laden des Templates: {e}")
            # Erstelle ein minimales Fallback-Template
            return {
                "title": "E-Learning-Kurs zur Informationssicherheit",
                "description": "Ein maßgeschneiderter Kurs zur Stärkung des Sicherheitsbewusstseins",
                "sections": [
                    {
                        "id": "context",
                        "title": "Unternehmenskontext",
                        "description": "Beschreibung des Unternehmens und seiner spezifischen Anforderungen",
                        "type": "context"
                    },
                    {
                        "id": "learning_objectives",
                        "title": "Lernziele",
                        "description": "Was die Teilnehmer nach Abschluss des Kurses wissen und können sollten",
                        "type": "learning_objectives"
                    },
                    {
                        "id": "process_security",
                        "title": "Sicherheit im Arbeitsalltag",
                        "description": "Sicherheitsrelevante Aspekte der täglichen Arbeitsprozesse",
                        "type": "process"
                    },
                    {
                        "id": "threats",
                        "title": "Potenzielle Risiken",
                        "description": "Sicherheitsrisiken, die für die Organisation relevant sind",
                        "type": "threats"
                    },
                    {
                        "id": "content",
                        "title": "Hauptinhalte",
                        "description": "Wesentliche Informationssicherheitskonzepte und -praktiken",
                        "type": "content"
                    },
                    {
                        "id": "practical_measures",
                        "title": "Praktische Maßnahmen",
                        "description": "Konkrete Schritte und Verhaltensweisen zum Schutz von Informationen",
                        "type": "controls"
                    },
                    {
                        "id": "methods",
                        "title": "Lehrmethoden",
                        "description": "Didaktische Ansätze zur effektiven Vermittlung der Inhalte",
                        "type": "methods"
                    },
                    {
                        "id": "assessment",
                        "title": "Erfolgskontrolle",
                        "description": "Methoden zur Überprüfung des Lernerfolgs",
                        "type": "assessment"
                    }
                ]
            }
    
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
    
    def create_script_from_responses(self, section_responses: Dict[str, str]) -> Dict[str, Any]:
        """
        Erstellt ein Skript aus den gegebenen Antworten.
        
        Args:
            section_responses: Dictionary mit Abschnitts-IDs als Schlüssel und Inhalten als Werte
            
        Returns:
            Vollständiges Skript als Dictionary
        """
        script = copy.deepcopy(self.template)
        
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
            "completed_sections": [],
            "current_section": None,
            "content_quality_checks": {}
        }
        
        # Liste der Kontextfragen
        self.context_questions = [
            "Für welche Art von Organisation erstellen wir den E-Learning-Kurs (z.B. Krankenhaus, Bank, Behörde)?",
            "Welche Mitarbeitergruppen sollen geschult werden?",
            "Wie lang sollte der E-Learning-Kurs maximal dauern?",
            "Gibt es spezifische Compliance-Anforderungen oder Branchenstandards, die berücksichtigt werden müssen?"
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
            return self.get_next_template_question()
        
        elif self.conversation_state["current_step"] == "template_navigation":
            return self.get_next_template_question()
        
        elif self.conversation_state["current_step"] == "review":
            return "Ich habe basierend auf Ihren Eingaben einen E-Learning-Kurs zur Informationssicherheit entworfen. Möchten Sie das Ergebnis sehen?"
        
        elif self.conversation_state["current_step"] == "completion":
            return "Vielen Dank für Ihre Mitwirkung! Der E-Learning-Kurs wurde erfolgreich erstellt und kann jetzt heruntergeladen werden."
    
    def get_next_template_question(self) -> str:
        """
        Bestimmt die nächste Frage basierend auf dem Template.
        
        Returns:
            Nächste Frage als String
        """
        # Finde den nächsten nicht bearbeiteten Abschnitt
        next_section = self.template_manager.get_next_section(self.conversation_state["completed_sections"])
        
        if next_section is None:
            # Alle Abschnitte wurden bearbeitet
            self.conversation_state["current_step"] = "review"
            return self.get_next_question()
        
        # Generiere eine Frage für diesen Abschnitt
        section_id = next_section["id"]
        section_title = next_section["title"]
        section_description = next_section["description"]
        
        # Hole relevante Dokumente für diesen Abschnitt
        retrieval_queries = self.generate_retrieval_queries(section_title, section_id)
        
        retrieved_docs = self.vector_store_manager.retrieve_with_multiple_queries(
            queries=retrieval_queries,
            filter={"section_type": next_section.get("type", "generic")},
            top_k=3
        )
        
        # Extrahiere Kontext aus den abgerufenen Dokumenten
        context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # Extrahiere Kontextinformationen für die Fragengenerierung
        organization = self.conversation_state["context_info"].get(
            "Für welche Art von Organisation erstellen wir den E-Learning-Kurs (z.B. Krankenhaus, Bank, Behörde)?", "")
        audience = self.conversation_state["context_info"].get(
            "Welche Mitarbeitergruppen sollen geschult werden?", "")
        
        # Generiere eine Frage mit dem LLM
        question = self.llm_manager.generate_question(
            section_title=section_title,
            section_description=section_description,
            context_text=context_text,
            organization=organization,
            audience=audience
        )
        
        # Aktualisiere den Gesprächszustand
        self.conversation_state["current_section"] = section_id
        
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
        
        # Abschnittsspezifische Anfragetemplates
        section_templates = {
            "context": f"Typische Informationssicherheitsherausforderungen für {organization}",
            "learning_objectives": f"Lernziele Informationssicherheit {organization} {audience}",
            "process_security": f"Sicherheitsmaßnahmen Arbeitsabläufe {organization}",
            "threats": f"Bedrohungen Sicherheitsrisiken {organization}",
            "content": f"Informationssicherheit Grundlagen {organization} {audience}",
            "practical_measures": f"Praktische Sicherheitsmaßnahmen {organization} Mitarbeiter",
            "methods": f"Didaktische Methoden Sicherheitsbewusstsein {audience}",
            "assessment": f"Lernerfolgsmessung Sicherheitsschulung {audience}"
        }
        
        # Verwende das entsprechende Template, falls vorhanden
        section_query = section_templates.get(section_id, "")
        
        # Compliance-spezifische Anfrage hinzufügen, falls vorhanden
        compliance_query = ""
        if compliance and compliance.strip():
            compliance_query = f"Informationssicherheit {compliance} {organization}"
        
        # Lerntheorie-Anfrage für Methoden-Abschnitt
        learning_theory_query = ""
        if section_id == "methods":
            learning_theory_query = f"Lerntheorien Erwachsenenbildung Sicherheitsschulung"
        
        # Erstelle die Liste der Anfragen
        queries = [base_query, industry_query]
        
        if section_query:
            queries.append(section_query)
        
        if compliance_query:
            queries.append(compliance_query)
        
        if learning_theory_query:
            queries.append(learning_theory_query)
        
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
            # Prüfe, ob die Antwort ausreichend detailliert ist
            if self.is_response_adequate(response):
                # Speichere die Antwort für den aktuellen Abschnitt
                current_section = self.conversation_state["current_section"]
                self.conversation_state["section_responses"][current_section] = response
                self.conversation_state["completed_sections"].append(current_section)
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
        # Einfache Heuristik: Prüfe die Länge der Antwort
        min_word_count = 10
        word_count = len(response.split())
        
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
        Die folgende Antwort des Kunden zu einer Frage über {section['title']} ist recht kurz:
        
        "{response}"
        
        Formuliere eine freundliche Nachfrage, die mehr Details zu ihren Prozessen oder ihrem Arbeitskontext erbittet.
        Die Nachfrage sollte:
        1. Wertschätzend für die bisherige Antwort sein
        2. Konkrete Aspekte ansprechen, zu denen mehr Details hilfreich wären
        3. Keine Fachbegriffe aus der Informationssicherheit verwenden
        4. Offen formuliert sein, um ausführlichere Antworten zu fördern
        
        Stelle nur die Nachfrage, keine Einleitung oder zusätzliche Erklärungen.
        """
        
        followup_question = self.llm_manager.llm(followup_prompt)
        
        return followup_question
    
    def generate_script(self) -> Dict[str, Any]:
        """
        Generiert den finalen E-Learning-Kurs.
        
        Returns:
            Kurs als Dictionary
        """
        # Erstelle eine Kopie des Templates
        script = copy.deepcopy(self.template_manager.template)
        
        # Organisation und Zielgruppe ermitteln für den Kurstitel
        organization = self.conversation_state["context_info"].get(
            "Für welche Art von Organisation erstellen wir den E-Learning-Kurs (z.B. Krankenhaus, Bank, Behörde)?", "")
        audience = self.conversation_state["context_info"].get(
            "Welche Mitarbeitergruppen sollen geschult werden?", "")
        
        # Titel und Beschreibung anpassen
        script["title"] = f"Informationssicherheit für {organization}"
        script["description"] = f"Ein maßgeschneiderter E-Learning-Kurs zur Stärkung des Sicherheitsbewusstseins für {audience} bei {organization}"
        
        # Für jeden Abschnitt des Templates
        for section in script["sections"]:
            section_id = section["id"]
            
            if section_id in self.conversation_state["section_responses"]:
                # Hole die Antwort des Nutzers
                user_response = self.conversation_state["section_responses"][section_id]
                
                # Extrahiere Schlüsselinformationen für das Retrieval
                key_concepts = self.llm_manager.extract_key_information(
                    section_type=section.get("type", "generic"),
                    user_response=user_response
                )
                
                # Generiere Retrieval-Anfragen basierend auf den Schlüsselkonzepten
                retrieval_queries = [
                    f"{concept} Informationssicherheit {organization}" 
                    for concept in key_concepts
                ]
                
                # Füge abschnittsspezifische Anfragen hinzu
                retrieval_queries.extend(self.generate_retrieval_queries(section["title"], section_id))
                
                # Hole relevante Dokumente
                retrieved_docs = self.vector_store_manager.retrieve_with_multiple_queries(
                    queries=retrieval_queries,
                    filter={"section_type": section.get("type", "generic")},
                    top_k=3
                )
                
                # Extrahiere Kontext aus den abgerufenen Dokumenten
                context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
                
                # Generiere Inhalt für diesen Abschnitt
                content = self.llm_manager.generate_content(
                    section_title=section["title"],
                    section_description=section.get("description", ""),
                    user_response=user_response,
                    organization=self.conversation_state["context_info"].get(
                        "Für welche Art von Organisation erstellen wir den E-Learning-Kurs (z.B. Krankenhaus, Bank, Behörde)?", ""),
                    audience=self.conversation_state["context_info"].get(
                        "Welche Mitarbeitergruppen sollen geschult werden?", ""),
                    duration=self.conversation_state["context_info"].get(
                        "Wie lang sollte der E-Learning-Kurs maximal dauern?", ""),
                    context_text=context_text
                )
                
                # Führe Qualitätsprüfung durch
                has_issues, verified_content = self.llm_manager.check_hallucinations(
                    content=content,
                    user_input=user_response,
                    context_text=context_text
                )
                
                # Speichere das Ergebnis der Qualitätsprüfung
                self.conversation_state["content_quality_checks"][section_id] = has_issues
                
                # Aktualisiere den Abschnitt mit dem geprüften Inhalt
                section["content"] = verified_content
        
        return script
    
    def get_script_summary(self) -> str:
        """
        Erstellt eine Zusammenfassung des generierten Kurses.
        
        Returns:
            Zusammenfassung als String
        """
        # Generiere den Kurs
        script = self.generate_script()
        
        # Erstelle eine Zusammenfassung
        summary = f"# {script.get('title', 'E-Learning-Kurs zur Informationssicherheit')}\n\n"
        
        if "description" in script:
            summary += f"{script['description']}\n\n"
        
        summary += "## Rahmenbedingungen\n\n"
        
        for question, answer in self.conversation_state["context_info"].items():
            summary += f"**{question}** {answer}\n\n"
        
        summary += "## Kursinhalt\n\n"
        
        for section in script["sections"]:
            summary += f"### {section['title']}\n\n"
            
            if "content" in section:
                summary += f"{section['content']}\n\n"
            else:
                summary += "Kein Inhalt verfügbar.\n\n"
            
            # Füge Hinweis zur Qualitätsprüfung hinzu, falls relevant
            section_id = section["id"]
            if section_id in self.conversation_state["content_quality_checks"]:
                has_issues = self.conversation_state["content_quality_checks"][section_id]
                if has_issues:
                    summary += "*Hinweis: Dieser Abschnitt wurde nach der Qualitätsprüfung überarbeitet.*\n\n"
        
        return summary
    
    def save_script(self, output_path: str) -> None:
        """
        Speichert den generierten Kurs in einer Datei.
        
        Args:
            output_path: Pfad zur Ausgabedatei
        """
        script = self.generate_script()
        
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(script, f, ensure_ascii=False, indent=2)
            
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
            template_path=self.config["template_path"]
        )
        
        self.dialog_manager = None
    
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
            "template_path": "./data/documents/templates/elearning_template.json",
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
        
        return self.dialog_manager.process_user_response(user_input)
    
    def save_generated_script(self, filename: str = None) -> str:
        """
        Speichert den generierten Kurs und gibt den Pfad zurück.
        
        Args:
            filename: Name der Ausgabedatei (optional)
            
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
            filename = f"elearning_{sanitized_organization}_{sanitized_audience}_{timestamp}.json"
        
        output_path = os.path.join(self.config["output_dir"], filename)
        self.dialog_manager.save_script(output_path)
        
        return output_path
    
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
    st.title("E-Learning-Kurs-Generator für Informationssicherheit")
    
    # Sidebar für Einstellungen und Aktionen
    with st.sidebar:
        st.header("Aktionen")
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
                st.experimental_rerun()
    
    # Initialisiere den Bot im Session State, falls nicht vorhanden
    if "bot" not in st.session_state:
        st.session_state.bot = ELearningCourseGenerator()
        st.session_state.messages = [
            {"role": "assistant", "content": st.session_state.bot.start_conversation()}
        ]
        st.session_state.script_generated = False
    
    # Zeige den Chatverlauf an
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Eingabefeld für den Nutzer
    if not st.session_state.script_generated:
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
                script_path = st.session_state.bot.save_generated_script()
                
                with open(script_path, "r", encoding="utf-8") as f:
                    script_content = f.read()
                
                st.download_button(
                    label="E-Learning-Kurs herunterladen",
                    data=script_content,
                    file_name=os.path.basename(script_path),
                    mime="application/json"
                )
    
    # Wenn der Kurs bereits generiert wurde, biete die Möglichkeit, einen neuen zu erstellen
    else:
        if st.button("Neuen E-Learning-Kurs erstellen"):
            st.session_state.bot = ELearningCourseGenerator()
            st.session_state.messages = [
                {"role": "assistant", "content": st.session_state.bot.start_conversation()}
            ]
            st.session_state.script_generated = False
            st.rerun()


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
        
        print(bot.start_conversation())
        
        while True:
            user_input = input("> ")
            
            if user_input.lower() in ["exit", "quit", "q"]:
                break
            
            response = bot.process_user_input(user_input)
            print(response)
            
            if "Hier ist der entworfene E-Learning-Kurs" in response:
                script_path = bot.save_generated_script()
                print(f"E-Learning-Kurs gespeichert unter: {script_path}")
                break


if __name__ == "__main__":
    from datetime import datetime
    main()