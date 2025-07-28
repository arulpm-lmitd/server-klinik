from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import threading
from typing import Dict, Any
import time

logger = logging.getLogger(__name__)

class APIService:
    def __init__(self, config, drug_predictor):
        self.config = config
        self.drug_predictor = drug_predictor
        self.app = Flask(__name__)
        self._setup_flask()
        self._register_routes()
        
        # Auto-initialize if configured
        if self.config.AUTO_INITIALIZE:
            logger.info("Auto-initialization enabled, starting model loading...")
            self.drug_predictor.initialize(background=True)
    
    def _setup_flask(self):
        """Configure Flask application"""
        CORS(self.app)
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, self.config.LOG_LEVEL),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Add before_first_request equivalent for Flask 2.3+
        @self.app.before_request
        def before_first_request():
            if not hasattr(self, '_first_request_done'):
                self._first_request_done = True
                if self.config.AUTO_INITIALIZE and not self.drug_predictor.is_ready():
                    logger.info("First request received, ensuring initialization...")
    
    def _register_routes(self):
        """Register all API routes"""
        
        @self.app.route('/health', methods=['GET'])
        @self.app.route('/', methods=['GET'])
        def health_check():
            """Health check endpoint"""
            return jsonify({
                'status': 'success',
                'message': 'Drug Recommendation API is running!',
                'version': '2.0.0',
                'ready': self.drug_predictor.is_ready(),
                'initialization_status': self.drug_predictor.get_status(),
                'endpoints': {
                    'GET /': 'Health check',
                    'GET /health': 'Health check',
                    'GET /ready': 'Readiness check',
                    'POST /predict': 'Predict drugs based on symptoms',
                    'GET /stats': 'Get database statistics',
                    'GET /status': 'Get initialization status',
                    'POST /initialize': 'Force initialization'
                }
            })
        
        @self.app.route('/ready', methods=['GET'])
        def readiness_check():
            """Kubernetes-style readiness check"""
            if self.drug_predictor.is_ready():
                return jsonify({
                    'status': 'ready',
                    'message': 'API is ready to serve requests'
                }), 200
            else:
                status = self.drug_predictor.get_status()
                return jsonify({
                    'status': 'not_ready',
                    'message': f'Initialization in progress: {status["message"]}',
                    'progress': status["progress"]
                }), 503
        
        @self.app.route('/status', methods=['GET'])
        def get_status():
            """Get initialization status"""
            return jsonify({
                'status': 'success',
                'data': self.drug_predictor.get_status()
            })
        
        @self.app.route('/stats', methods=['GET'])
        def get_stats():
            """Get database statistics"""
            try:
                stats = self.drug_predictor.get_stats()
                if 'error' in stats:
                    return jsonify({
                        'status': 'error',
                        'message': stats['error']
                    }), 503
                
                return jsonify({
                    'status': 'success',
                    'data': stats
                })
            except Exception as e:
                logger.error(f"Error getting stats: {str(e)}")
                return jsonify({
                    'status': 'error',
                    'message': f'An error occurred: {str(e)}'
                }), 500
        
        @self.app.route('/initialize', methods=['POST'])
        def initialize():
            """Force initialization"""
            status = self.drug_predictor.get_status()
            
            if status["status"] == "in_progress":
                return jsonify({
                    'status': 'info',
                    'message': 'Initialization already in progress',
                    'progress': status["progress"]
                })
            
            if status["status"] == "completed":
                return jsonify({
                    'status': 'success',
                    'message': 'Model already initialized'
                })
            
            # Start initialization
            self.drug_predictor.initialize(background=True)
            
            return jsonify({
                'status': 'success',
                'message': 'Initialization started',
                'estimated_time': '2-5 minutes'
            })
        
        @self.app.route('/predict', methods=['POST'])
        def predict():
            """Main prediction endpoint"""
            try:
                # Check if model is ready
                if not self.drug_predictor.is_ready():
                    status = self.drug_predictor.get_status()
                    
                    if status["status"] == "not_started":
                        return jsonify({
                            'status': 'error',
                            'message': 'Model not initialized. Please wait for auto-initialization or call POST /initialize.',
                            'initialization_status': status
                        }), 503
                    
                    if status["status"] == "in_progress":
                        return jsonify({
                            'status': 'info',
                            'message': f'Model initialization in progress ({status["progress"]}%). Please wait...',
                            'initialization_status': status
                        }), 202
                    
                    if status["status"] == "failed":
                        return jsonify({
                            'status': 'error',
                            'message': 'Model initialization failed',
                            'error': status.get("error")
                        }), 503
                
                # Validate request
                data = request.get_json()
                if not data:
                    return jsonify({
                        'status': 'error',
                        'message': 'No JSON data provided'
                    }), 400
                
                # Extract and validate parameters
                keluhan = data.get('keluhan', '').strip()
                anamnesa = data.get('anamnesa', '').strip()
                top_k = data.get('top_k', self.config.DEFAULT_TOP_K)
                
                if not keluhan and not anamnesa:
                    return jsonify({
                        'status': 'error',
                        'message': 'Either keluhan or anamnesa must be provided'
                    }), 400
                
                if not isinstance(top_k, int) or top_k < 1 or top_k > self.config.MAX_TOP_K:
                    top_k = self.config.DEFAULT_TOP_K
                
                # Make prediction
                start_time = time.time()
                predictions = self.drug_predictor.predict(keluhan, anamnesa, top_k)
                processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
                
                return jsonify({
                    'status': 'success',
                    'query': {
                        'keluhan': keluhan,
                        'anamnesa': anamnesa,
                        'top_k': top_k
                    },
                    'predictions': predictions,
                    'total_results': len(predictions),
                    'processing_time_ms': round(processing_time, 2)
                })
                
            except ValueError as e:
                return jsonify({
                    'status': 'error',
                    'message': str(e)
                }), 400
            except RuntimeError as e:
                return jsonify({
                    'status': 'error',
                    'message': str(e)
                }), 503
            except Exception as e:
                logger.error(f"Error in predict endpoint: {str(e)}")
                return jsonify({
                    'status': 'error',
                    'message': 'An internal error occurred'
                }), 500
        
        @self.app.errorhandler(404)
        def not_found(error):
            return jsonify({
                'status': 'error',
                'message': 'Endpoint not found'
            }), 404
        
        @self.app.errorhandler(500)
        def internal_error(error):
            return jsonify({
                'status': 'error',
                'message': 'Internal server error'
            }), 500
    
    def run(self):
        """Run the Flask application"""
        logger.info(f"Starting Drug Recommendation API on {self.config.HOST}:{self.config.PORT}")
        
        # If auto-initialization is enabled, wait for it to complete in production
        if self.config.AUTO_INITIALIZE and not self.config.DEBUG:
            logger.info("Waiting for model initialization to complete before starting server...")
            if self.drug_predictor.wait_for_initialization():
                logger.info("✅ Model initialization completed! API is ready to serve requests.")
            else:
                logger.error("❌ Model initialization failed! API will start but predictions will not work.")
        
        self.app.run(
            host=self.config.HOST,
            port=self.config.PORT,
            debug=self.config.DEBUG,
            threaded=True
        )
