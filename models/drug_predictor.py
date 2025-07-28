import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import login
import logging
from typing import List, Dict, Optional
import threading
import time
import os

logger = logging.getLogger(__name__)

class DrugPredictor:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.drug_embeddings = None
        self.df_drugs = None
        self.initialization_status = {
            "status": "not_started",
            "error": None,
            "progress": 0,
            "message": "",
            "start_time": None,
            "end_time": None
        }
        self._lock = threading.Lock()
        self._initialization_thread = None
    
    def initialize(self, background=False) -> bool:
        """Initialize the model and load drug database"""
        if background:
            if self._initialization_thread and self._initialization_thread.is_alive():
                logger.info("Initialization already running in background")
                return True
            
            self._initialization_thread = threading.Thread(target=self._initialize_sync)
            self._initialization_thread.daemon = True
            self._initialization_thread.start()
            return True
        else:
            return self._initialize_sync()
    
    def _initialize_sync(self) -> bool:
        """Synchronous initialization implementation"""
        with self._lock:
            if self.initialization_status["status"] == "completed":
                return True
            
            if self.initialization_status["status"] == "in_progress":
                return False
            
            self.initialization_status["status"] = "in_progress"
            self.initialization_status["progress"] = 0
            self.initialization_status["message"] = "Starting initialization..."
            self.initialization_status["start_time"] = time.time()
            self.initialization_status["error"] = None
            
            try:
                # Step 1: Login to Hugging Face if token provided
                if self.config.HUGGINGFACE_TOKEN:
                    logger.info("Logging in to Hugging Face...")
                    login(token=self.config.HUGGINGFACE_TOKEN)
                    self.initialization_status["progress"] = 10
                    self.initialization_status["message"] = "Logged in to Hugging Face"
                
                # Step 2: Load model
                logger.info(f"Loading model: {self.config.MODEL_NAME}")
                self.initialization_status["progress"] = 20
                self.initialization_status["message"] = "Loading model from Hugging Face..."
                
                # FIX: Add trust_remote_code=True to allow custom code execution
                self.model = SentenceTransformer(
                    self.config.MODEL_NAME,
                    trust_remote_code=True
                )
                
                logger.info("Model loaded successfully!")
                self.initialization_status["progress"] = 50
                self.initialization_status["message"] = "Model loaded successfully"
                
                # Step 3: Load drug database
                logger.info("Loading drug database...")
                self.initialization_status["progress"] = 60
                self.initialization_status["message"] = "Loading drug database..."
                
                if not self._load_drug_database():
                    return False
                
                # Step 4: Generate embeddings
                logger.info("Generating drug embeddings...")
                self.initialization_status["progress"] = 80
                self.initialization_status["message"] = "Generating embeddings..."
                
                if not self._generate_embeddings():
                    return False
                
                # Completion
                self.initialization_status["status"] = "completed"
                self.initialization_status["progress"] = 100
                self.initialization_status["message"] = "Initialization completed successfully"
                self.initialization_status["error"] = None
                self.initialization_status["end_time"] = time.time()
                
                elapsed_time = self.initialization_status["end_time"] - self.initialization_status["start_time"]
                logger.info(f"Drug predictor initialization completed successfully in {elapsed_time:.2f} seconds!")
                return True
                
            except Exception as e:
                error_msg = f"Initialization failed: {str(e)}"
                logger.error(error_msg)
                self.initialization_status["status"] = "failed"
                self.initialization_status["error"] = error_msg
                self.initialization_status["message"] = error_msg
                self.initialization_status["end_time"] = time.time()
                return False
    
    def wait_for_initialization(self, timeout=None) -> bool:
        """Wait for initialization to complete"""
        timeout = timeout or self.config.STARTUP_TIMEOUT
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.initialization_status["status"]
            
            if status == "completed":
                return True
            elif status == "failed":
                return False
            elif status == "not_started":
                logger.warning("Initialization not started, starting now...")
                self.initialize(background=True)
            
            time.sleep(1)  # Check every second
        
        logger.error(f"Initialization timeout after {timeout} seconds")
        return False
    
    def _load_drug_database(self) -> bool:
        """Load and validate drug database"""
        try:
            if not os.path.exists(self.config.CSV_PATH):
                raise FileNotFoundError(f"CSV file not found: {self.config.CSV_PATH}")
            
            self.df_drugs = pd.read_csv(self.config.CSV_PATH)
            logger.info(f"CSV loaded with shape: {self.df_drugs.shape}")
            
            # Validate required columns
            required_columns = ['Nama', 'DeskripsiObat']
            missing_columns = [col for col in required_columns if col not in self.df_drugs.columns]
            
            if missing_columns:
                raise ValueError(f"Missing columns: {missing_columns}. Available: {self.df_drugs.columns.tolist()}")
            
            # Process data
            self.df_drugs['ObatLengkap'] = self.df_drugs['Nama'].astype(str) + ' - ' + self.df_drugs['DeskripsiObat'].astype(str)
            self.df_drugs = self.df_drugs.drop_duplicates(subset=['ObatLengkap']).reset_index(drop=True)
            
            logger.info(f"Drug database processed successfully! Total drugs: {len(self.df_drugs)}")
            return True
            
        except Exception as e:
            error_msg = f"Error loading drug database: {str(e)}"
            logger.error(error_msg)
            self.initialization_status["status"] = "failed"
            self.initialization_status["error"] = error_msg
            return False
    
    def _generate_embeddings(self) -> bool:
        """Generate embeddings for all drugs"""
        try:
            drug_texts = self.df_drugs['ObatLengkap'].tolist()
            self.drug_embeddings = self.model.encode(
                drug_texts, 
                show_progress_bar=True,
                batch_size=32
            )
            
            logger.info(f"Embeddings generated successfully! Shape: {self.drug_embeddings.shape}")
            return True
            
        except Exception as e:
            error_msg = f"Error generating embeddings: {str(e)}"
            logger.error(error_msg)
            self.initialization_status["status"] = "failed"
            self.initialization_status["error"] = error_msg
            return False
    
    def predict(self, keluhan: str, anamnesa: str, top_k: int = 5) -> List[Dict]:
        """Predict drugs based on symptoms and anamnesa"""
        if self.initialization_status["status"] != "completed":
            raise RuntimeError("Model not initialized")
        
        # Combine and clean query
        query = f"{keluhan} {anamnesa}".strip()
        if not query:
            raise ValueError("Query cannot be empty")
        
        # Generate query embedding
        query_embedding = self.model.encode([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.drug_embeddings)[0]
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_scores = similarities[top_indices]
        
        # Format results
        results = []
        for i, (idx, score) in enumerate(zip(top_indices, top_scores)):
            drug_info = {
                'rank': i + 1,
                'nama_obat': self.df_drugs.iloc[idx]['Nama'],
                'deskripsi_obat': self.df_drugs.iloc[idx]['DeskripsiObat'],
                'similarity_score': float(score),
                'confidence': self._get_confidence_level(float(score))
            }
            results.append(drug_info)
        
        return results
    
    def _get_confidence_level(self, score: float) -> str:
        """Convert similarity score to confidence level"""
        if score >= 0.8:
            return "high"
        elif score >= 0.6:
            return "medium"
        elif score >= 0.4:
            return "low"
        else:
            return "very_low"
    
    def is_ready(self) -> bool:
        """Check if model is ready for predictions"""
        return self.initialization_status["status"] == "completed"
    
    def get_status(self) -> Dict:
        """Get current initialization status"""
        status = {**self.initialization_status}
        
        # Add runtime information
        if status["start_time"]:
            if status["end_time"]:
                status["duration"] = status["end_time"] - status["start_time"]
            else:
                status["duration"] = time.time() - status["start_time"]
        
        # Add component status
        status.update({
            'model_loaded': self.model is not None,
            'embeddings_loaded': self.drug_embeddings is not None,
            'database_loaded': self.df_drugs is not None
        })
        
        return status
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        if self.df_drugs is None:
            return {"error": "Database not loaded"}
        
        return {
            'total_drugs': len(self.df_drugs),
            'columns': self.df_drugs.columns.tolist(),
            'embedding_shape': self.drug_embeddings.shape if self.drug_embeddings is not None else None,
            'model_name': self.config.MODEL_NAME,
            'csv_path': self.config.CSV_PATH
        }