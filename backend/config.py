import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Flask application configuration."""
    
    # Flask settings
    PORT = int(os.getenv('PORT', 5000))
    HOST = os.getenv('HOST', '0.0.0.0')
    DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'
    
    # Project Paths (Relative to project root, making them absolute here)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_model.pkl')
    # If best_model.pkl is missing or small, we might want to fallback to xgboost_model.pkl
    # but the requirement is specific.
    
    SCALER_PATH = os.path.join(BASE_DIR, 'models', 'feature_scaler.json')
    FEATURE_NAMES_PATH = os.path.join(BASE_DIR, 'feature_names.txt')
    
    # Database - Supabase
    SUPABASE_URL = os.getenv('SUPABASE_URL')
    SUPABASE_KEY = os.getenv('SUPABASE_KEY')
    
    # CORS
    ALLOWED_ORIGINS = ["http://localhost:3000"]
    
    # Model Metadata
    MODEL_TYPE = "xgboost"
    UNIT = "kg/hectare"
