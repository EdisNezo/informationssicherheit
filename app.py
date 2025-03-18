import os
import json
import copy
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path

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
    für den RAG-basierten E-Learning-Skript-Generator.
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
                    
                except Exception as e:
                    logger.error(f"Fehler beim Laden von {file_path}: {e}")
        
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
        elif "kompetenzprofil" in path_str or "profile" in path_str:
            return "competence_profile"
        elif "beispiel" in path_str or "example" in path_str:
            return "example"
        else:
            # Versuche, anhand des Verzeichnisnamens zu bestimmen
            parent_dir = file_path.parent.name.lower()
            if parent_dir in ["templates", "profile", "examples"]:
                return parent_dir[:-1]  # Entferne das 's' am Ende
            
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
        
        if any(term in content_lower for term in ["lernziel", "learning objective"]):
            return "learning_objectives"
        elif any(term in content_lower for term in ["inhalt", "content"]):
            return "content"
        elif any(term in content_lower for term in ["methode", "didaktik", "method"]):
            return "methods"
        elif any(term in content_lower for term in ["prüfung", "assessment", "evaluation"]):
            return "assessment"
        
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
            persist_directory=self.persist_directory
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
        Basierend auf folgendem Kontext:
        - Abschnitt im E-Learning-Skript: {section_title}
        - Beschreibung: {section_description}
        - Relevanter Hintergrund: {context_text}
        - Einrichtung: {facility}
        - Zielgruppe: {audience}
        
        Formuliere eine präzise, offene Frage, um Informationen für diesen Abschnitt des E-Learning-Skripts zu erhalten.
        Die Frage sollte detaillierte und spezifische Antworten fördern.
        
        Gib nur die Frage zurück, keine Erklärungen oder Einleitungen.
        """
        
        return PromptTemplate(
            template=template,
            input_variables=["section_title", "section_description", "context_text", 
                             "facility", "audience"]
        )
    
    def _create_content_generation_prompt(self) -> PromptTemplate:
        """
        Erstellt ein Prompt-Template für die Inhaltsgenerierung.
        
        Returns:
            PromptTemplate-Objekt
        """
        template = """
        Erstelle den Inhalt für den Abschnitt "{section_title}" eines E-Learning-Skripts.
        
        Benutzerantwort: {user_response}
        
        Kontext:
        - Abschnittsbeschreibung: {section_description}
        - Einrichtung: {facility}
        - Zielgruppe: {audience}
        - Dauer: {duration}
        - Relevante Informationen: {context_text}
        
        Der Inhalt sollte:
        1. Die Antwort des Benutzers integrieren
        2. Fachlich korrekt und relevant sein
        3. Dem Niveau der Zielgruppe entsprechen
        4. Didaktisch sinnvoll strukturiert sein
        5. Keine Informationen enthalten, die nicht durch die Benutzerantwort oder den Kontext gestützt werden
        
        Gib nur den fertigen Inhalt zurück, keine zusätzlichen Erklärungen.
        """
        
        return PromptTemplate(
            template=template,
            input_variables=["section_title", "section_description", "user_response", 
                             "facility", "audience", "duration", "context_text"]
        )
    
    def _create_hallucination_check_prompt(self) -> PromptTemplate:
        """
        Erstellt ein Prompt-Template für die Halluzinationsprüfung.
        
        Returns:
            PromptTemplate-Objekt
        """
        template = """
        Überprüfe den folgenden Text auf mögliche Halluzinationen, d.h. Aussagen, die nicht durch die Benutzerantwort oder den Kontext gestützt werden.
        
        Text: {content}
        
        Benutzerantwort: {user_input}
        
        Verfügbarer Kontext:
        {context_text}
        
        Identifiziere alle Aussagen, die nicht durch die Benutzerantwort oder den Kontext gestützt werden.
        Für jede identifizierte Halluzination:
        1. Zitiere die problematische Passage
        2. Erkläre, warum es sich um eine Halluzination handelt
        3. Schlage eine Korrektur vor, die mit dem verfügbaren Kontext übereinstimmt
        
        Falls keine Halluzinationen gefunden wurden, antworte mit "KEINE_HALLUZINATIONEN".
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
        Extrahiere die wichtigsten Schlüsselbegriffe und -konzepte aus folgender Antwort 
        zu einer Frage über {section_type} im E-Learning-Bereich:
        
        "{user_response}"
        
        Gib nur eine Liste von 3-5 Schlüsselbegriffen oder Phrasen zurück, die für 
        eine Informationssuche verwendet werden könnten. Keine Erklärungen.
        """
        
        return PromptTemplate(
            template=template,
            input_variables=["section_type", "user_response"]
        )
    
    def generate_question(self, section_title: str, section_description: str, 
                         context_text: str, facility: str, audience: str) -> str:
        """
        Generiert eine Frage für einen Abschnitt des Templates.
        
        Args:
            section_title: Titel des Abschnitts
            section_description: Beschreibung des Abschnitts
            context_text: Kontextinformationen aus dem Retrieval
            facility: Art der medizinischen Einrichtung
            audience: Zielgruppe der Schulung
            
        Returns:
            Generierte Frage
        """
        response = self.chains["question_generation"].run({
            "section_title": section_title,
            "section_description": section_description,
            "context_text": context_text,
            "facility": facility,
            "audience": audience
        })
        
        return response.strip()
    
    def generate_content(self, section_title: str, section_description: str, 
                        user_response: str, facility: str, audience: str, 
                        duration: str, context_text: str) -> str:
        """
        Generiert Inhalt für einen Abschnitt des Skripts.
        
        Args:
            section_title: Titel des Abschnitts
            section_description: Beschreibung des Abschnitts
            user_response: Antwort des Nutzers
            facility: Art der medizinischen Einrichtung
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
            "facility": facility,
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
            Tuple aus (hat_halluzinationen, korrigierter_inhalt)
        """
        response = self.chains["hallucination_check"].run({
            "content": content,
            "user_input": user_input,
            "context_text": context_text
        })
        
        # Prüfe, ob Halluzinationen gefunden wurden
        if "KEINE_HALLUZINATIONEN" in response:
            return False, content
        
        # Korrigiere den Inhalt basierend auf dem Halluzinations-Check
        corrected_content = self.generate_content_with_corrections(content, response)
        
        return True, corrected_content
    
    def generate_content_with_corrections(self, original_content: str, correction_feedback: str) -> str:
        """
        Generiert korrigierten Inhalt basierend auf dem Halluzinations-Feedback.
        
        Args:
            original_content: Ursprünglicher Inhalt
            correction_feedback: Feedback zur Korrektur
            
        Returns:
            Korrigierter Inhalt
        """
        correction_prompt = f"""
        Korrigiere den folgenden Text basierend auf dem Feedback:
        
        Originaltext:
        {original_content}
        
        Feedback zu Korrekturen:
        {correction_feedback}
        
        Erstelle eine korrigierte Version des Textes, die die identifizierten Probleme behebt.
        Gib nur den korrigierten Text zurück, keine Erklärungen.
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
    Verwaltet das Template für das E-Learning-Skript.
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
                "title": "E-Learning-Skript",
                "description": "Ein Skript für eine E-Learning-Einheit",
                "sections": [
                    {
                        "id": "learning_objectives",
                        "title": "Lernziele",
                        "description": "Beschreibung der zu erreichenden Lernziele",
                        "type": "learning_objectives"
                    },
                    {
                        "id": "content",
                        "title": "Inhalte",
                        "description": "Fachliche Inhalte der Schulung",
                        "type": "content"
                    },
                    {
                        "id": "methods",
                        "title": "Methoden",
                        "description": "Didaktische Methoden zur Vermittlung der Inhalte",
                        "type": "methods"
                    },
                    {
                        "id": "assessment",
                        "title": "Erfolgsmessung",
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
    Implementiert den dialogbasierten Prozess zur Erstellung eines E-Learning-Skripts.
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
            "hallucination_checks": {}
        }
        
        # Liste der Kontextfragen
        self.context_questions = [
            "In welcher medizinischen Einrichtung sollen die Schulungen umgesetzt werden?",
            "Welche spezifischen Personengruppen sollen geschult werden?",
            "Wie lang darf die Schulung maximal sein?",
            "Gibt es spezifische Kompetenzprofile zu berücksichtigen?"
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
            return "Willkommen bei der E-Learning-Skript-Erstellung. Ich werde Sie durch den Prozess führen. Zunächst benötige ich einige Informationen für den Kontext. In welcher medizinischen Einrichtung sollen die Schulungen umgesetzt werden?"
        
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
            return "Ich habe basierend auf Ihren Antworten ein E-Learning-Skript erstellt. Möchten Sie das Ergebnis sehen?"
        
        elif self.conversation_state["current_step"] == "completion":
            return "Vielen Dank für Ihre Mitarbeit bei der Erstellung des E-Learning-Skripts. Das Skript wurde erfolgreich generiert."
    
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
        
        # Extrahiere den Kontext aus den abgerufenen Dokumenten
        context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # Extrahiere Kontextinformationen für die Fragengenerierung
        facility = self.conversation_state["context_info"].get(
            "In welcher medizinischen Einrichtung sollen die Schulungen umgesetzt werden?", "")
        audience = self.conversation_state["context_info"].get(
            "Welche spezifischen Personengruppen sollen geschult werden?", "")
        
        # Generiere eine Frage mit dem LLM
        question = self.llm_manager.generate_question(
            section_title=section_title,
            section_description=section_description,
            context_text=context_text,
            facility=facility,
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
        facility = self.conversation_state["context_info"].get(
            "In welcher medizinischen Einrichtung sollen die Schulungen umgesetzt werden?", "")
        audience = self.conversation_state["context_info"].get(
            "Welche spezifischen Personengruppen sollen geschult werden?", "")
        
        context_query = f"{section_title} für {facility} und {audience}"
        
        # Abschnittsspezifische Anfragetemplates
        section_templates = {
            "learning_objectives": f"Lernziele für {section_title} in {facility}",
            "content": f"Fachlicher Inhalt zu {section_title} für {audience}",
            "methods": f"Didaktische Methoden für {section_title} bei {audience}",
            "assessment": f"Prüfungsmethoden für {section_title} bei {audience}"
        }
        
        # Verwende das entsprechende Template, falls vorhanden
        section_query = section_templates.get(section_id, "")
        
        # Erstelle die Liste der Anfragen
        queries = [base_query, context_query]
        
        if section_query:
            queries.append(section_query)
        
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
                return "Hier ist das generierte E-Learning-Skript basierend auf Ihren Eingaben:\n\n" + self.get_script_summary()
            else:
                # Frage erneut, ob der Nutzer das Ergebnis sehen möchte
                return "Möchten Sie das generierte E-Learning-Skript ansehen? Bitte antworten Sie mit 'Ja' oder 'Nein'."
        
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
        min_word_count = 15
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
        Die folgende Antwort zu einer Frage über '{section['title']}' für ein E-Learning-Skript ist nicht ausreichend detailliert:
        
        "{response}"
        
        Formuliere eine freundliche Nachfrage, die nach konkreteren Details fragt 
        und dem Nutzer hilft, eine ausführlichere Antwort zu geben.
        """
        
        followup_question = self.llm_manager.llm(followup_prompt)
        return followup_question
    
    def generate_script(self) -> Dict[str, Any]:
        """
        Generiert das finale E-Learning-Skript.
        
        Returns:
            Skript als Dictionary
        """
        # Erstelle eine Kopie des Templates
        script = copy.deepcopy(self.template_manager.template)
        
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
                
                # Generiere Retrieval-Anfragen
                retrieval_queries = [
                    f"{concept} für {self.conversation_state['context_info'].get('In welcher medizinischen Einrichtung sollen die Schulungen umgesetzt werden?', '')}" 
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
                
                # Extrahiere den Kontext aus den abgerufenen Dokumenten
                context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
                
                # Generiere den Inhalt für diesen Abschnitt
                content = self.llm_manager.generate_content(
                    section_title=section["title"],
                    section_description=section.get("description", ""),
                    user_response=user_response,
                    facility=self.conversation_state["context_info"].get("In welcher medizinischen Einrichtung sollen die Schulungen umgesetzt werden?", ""),
                    audience=self.conversation_state["context_info"].get("Welche spezifischen Personengruppen sollen geschult werden?", ""),
                    duration=self.conversation_state["context_info"].get("Wie lang darf die Schulung maximal sein?", ""),
                    context_text=context_text
                )
                
                # Führe Halluzinationsprüfung durch
                has_hallucinations, verified_content = self.llm_manager.check_hallucinations(
                    content=content,
                    user_input=user_response,
                    context_text=context_text
                )
                
                # Speichere das Ergebnis der Halluzinationsprüfung
                self.conversation_state["hallucination_checks"][section_id] = has_hallucinations
                
                # Aktualisiere den Abschnitt mit dem verifizierten Inhalt
                section["content"] = verified_content
        
        return script
    
    def get_script_summary(self) -> str:
        """
        Erstellt eine Zusammenfassung des generierten Skripts.
        
        Returns:
            Zusammenfassung als String
        """
        # Generiere das Skript
        script = self.generate_script()
        
        # Erstelle eine Zusammenfassung
        summary = f"# {script.get('title', 'E-Learning-Skript')}\n\n"
        
        if "description" in script:
            summary += f"{script['description']}\n\n"
        
        summary += "## Kontextinformationen\n\n"
        
        for question, answer in self.conversation_state["context_info"].items():
            summary += f"**{question}** {answer}\n\n"
        
        summary += "## Inhalte\n\n"
        
        for section in script["sections"]:
            summary += f"### {section['title']}\n\n"
            
            if "content" in section:
                summary += f"{section['content']}\n\n"
            else:
                summary += "Kein Inhalt verfügbar.\n\n"
            
            # Füge Informationen zur Halluzinationsprüfung hinzu
            section_id = section["id"]
            if section_id in self.conversation_state["hallucination_checks"]:
                has_hallucinations = self.conversation_state["hallucination_checks"][section_id]
                if has_hallucinations:
                    summary += "*Hinweis: Für diesen Abschnitt wurden potenzielle Ungenauigkeiten korrigiert.*\n\n"
        
        return summary
    
    def save_script(self, output_path: str) -> None:
        """
        Speichert das generierte Skript in einer Datei.
        
        Args:
            output_path: Pfad zur Ausgabedatei
        """
        script = self.generate_script()
        
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(script, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Skript gespeichert: {output_path}")
        except Exception as e:
            logger.error(f"Fehler beim Speichern des Skripts: {e}")


class ELearningScriptBot:
    """
    Hauptklasse, die alle Komponenten des E-Learning-Skript-Generators integriert.
    """
    
    def __init__(self, config_path: str = "./config.json"):
        """
        Initialisiert den ELearningScriptBot.
        
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
            "template_path": "./data/templates/blank_template.json",
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
        Richtet den Bot ein, indem Dokumente geladen und die Vektordatenbank erstellt wird.
        """
        # Versuche, eine bestehende Vektordatenbank zu laden
        if not self.vector_store_manager.load_vectorstore():
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
        Speichert das generierte Skript und gibt den Pfad zurück.
        
        Args:
            filename: Name der Ausgabedatei (optional)
            
        Returns:
            Pfad zur gespeicherten Datei
        """
        if self.dialog_manager is None:
            raise ValueError("Der Dialog-Manager wurde nicht initialisiert")
        
        if filename is None:
            # Generiere einen Dateinamen basierend auf dem Kontext
            facility = self.dialog_manager.conversation_state["context_info"].get(
                "In welcher medizinischen Einrichtung sollen die Schulungen umgesetzt werden?", "")
            audience = self.dialog_manager.conversation_state["context_info"].get(
                "Welche spezifischen Personengruppen sollen geschult werden?", "")
            
            sanitized_facility = ''.join(c for c in facility if c.isalnum() or c.isspace()).strip().replace(' ', '_')
            sanitized_audience = ''.join(c for c in audience if c.isalnum() or c.isspace()).strip().replace(' ', '_')
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"script_{sanitized_facility}_{sanitized_audience}_{timestamp}.json"
        
        output_path = os.path.join(self.config["output_dir"], filename)
        self.dialog_manager.save_script(output_path)
        
        return output_path


# Streamlit-Benutzeroberfläche für den E-Learning-Skript-Generator
def create_streamlit_app():
    """
    Erstellt eine Streamlit-Anwendung für den E-Learning-Skript-Generator.
    """
    st.title("E-Learning-Skript-Generator")
    
    # Initialisiere den Bot im Session State, falls nicht vorhanden
    if "bot" not in st.session_state:
        st.session_state.bot = ELearningScriptBot()
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
            
            # Prüfe, ob das Skript generiert wurde
            if "Hier ist das generierte E-Learning-Skript" in bot_response:
                st.session_state.script_generated = True
                
                # Speichere das Skript und biete es zum Download an
                script_path = st.session_state.bot.save_generated_script()
                
                with open(script_path, "r", encoding="utf-8") as f:
                    script_content = f.read()
                
                st.download_button(
                    label="Skript herunterladen",
                    data=script_content,
                    file_name=os.path.basename(script_path),
                    mime="application/json"
                )
    
    # Wenn das Skript bereits generiert wurde, biete die Möglichkeit, ein neues zu erstellen
    else:
        if st.button("Neues Skript erstellen"):
            st.session_state.bot = ELearningScriptBot()
            st.session_state.messages = [
                {"role": "assistant", "content": st.session_state.bot.start_conversation()}
            ]
            st.session_state.script_generated = False
            st.rerun()

def reindex_all():
    """Löscht die Vektordatenbank und erzwingt eine Neuindexierung."""
    import shutil
    import os
    
    vectorstore_dir = "./data/vectorstore"
    if os.path.exists(vectorstore_dir):
        shutil.rmtree(vectorstore_dir)
        os.makedirs(vectorstore_dir, exist_ok=True)
        print(f"Vektordatenbank gelöscht: {vectorstore_dir}")
    
    # Starte die App mit Setup
    bot = ELearningScriptBot()
    bot.setup()
    print("Neuindexierung abgeschlossen!")

# Hauptfunktion
def main():
    """
    Hauptfunktion zum Starten des E-Learning-Skript-Generators.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="E-Learning-Skript-Generator")
    parser.add_argument("--config", type=str, default="./config.json", help="Pfad zur Konfigurationsdatei")
    parser.add_argument("--mode", type=str, choices=["cli", "web"], default="web", help="Ausführungsmodus (CLI oder Web)")
    parser.add_argument("--reindex", action="store_true", help="Dokumente neu indexieren")
    
    args = parser.parse_args()

    if args.reindex:
        reindex_all()
        return
    
    if args.mode == "web":
        # Starte die Streamlit-Anwendung
        create_streamlit_app()
    else:
        # Starte den CLI-Modus
        bot = ELearningScriptBot(config_path=args.config)
        bot.setup()
        
        print(bot.start_conversation())
        
        while True:
            user_input = input("> ")
            
            if user_input.lower() in ["exit", "quit", "q"]:
                break
            
            response = bot.process_user_input(user_input)
            print(response)
            
            if "Hier ist das generierte E-Learning-Skript" in response:
                script_path = bot.save_generated_script()
                print(f"Skript gespeichert unter: {script_path}")
                break


if __name__ == "__main__":
    import datetime
    from datetime import datetime
    main()