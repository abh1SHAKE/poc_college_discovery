import os

class Config:
    # LLM Settings
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

    # Scraping Settings
    REQUEST_DELAY = float(os.getenv("REQUEST_DELAY", "1.5"))
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
    MAX_COLLEGES_PER_QUERY = int(os.getenv("MAX_COLLEGES_PER_QUERY", "20"))

    # Confidence Thresholds
    MIN_CONFIDENCE_THRESHOLD = float(os.getenv("MIN_CONFIDENCE_THRESHOLD", "0.3"))
    VERIFICATION_THRESHOLD = float(os.getenv("VERIFICATION_THRESHOLD", "0.5"))

    # Output Settings
    OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")
    DEFAULT_OUTPUT_FORMAT = os.getenv("DEFAULT_OUTPUT_FORMAT", "both")