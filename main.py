import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from models.drug_predictor import DrugPredictor
from services.api_service import APIService
import logging

# For Gunicorn compatibility
def create_app():
    """Application factory for Gunicorn"""
    # Initialize configuration
    config = Config()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Validate configuration
    if not config.HUGGINGFACE_TOKEN:
        logger.warning("HUGGINGFACE_TOKEN not set. Public models only.")
    
    if not os.path.exists(config.CSV_PATH):
        logger.error(f"Drug database file not found: {config.CSV_PATH}")
        raise FileNotFoundError(f"Required file missing: {config.CSV_PATH}")
    
    # Initialize components
    logger.info("Initializing Drug Predictor...")
    drug_predictor = DrugPredictor(config)
    
    logger.info("Initializing API Service...")
    api_service = APIService(config, drug_predictor)
    
    return api_service.app

def main():
    """Main application entry point for direct execution"""
    try:
        app = create_app()
        
        # Get config for direct run
        config = Config()
        logger = logging.getLogger(__name__)
        
        logger.info("Starting application...")
        logger.info(f"API will be available at: http://{config.HOST}:{config.PORT}")
        
        if config.AUTO_INITIALIZE:
            logger.info("🚀 Auto-initialization enabled - model will load automatically")
        else:
            logger.info("🔧 Manual initialization - call POST /initialize to load model")
        
        app.run(
            host=config.HOST,
            port=config.PORT,
            debug=config.DEBUG,
            threaded=True
        )
        
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Application failed to start: {str(e)}")
        sys.exit(1)

# For Gunicorn
app = create_app()

if __name__ == '__main__':
    main()