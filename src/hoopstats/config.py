import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Roboflow Configuration
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")

def validate_config():
    """
    Validates that essential configuration variables are present.
    Exits the program if critical keys are missing.
    """
    if not ROBOFLOW_API_KEY:
        print("Error: ROBOFLOW_API_KEY not found in environment variables.")
        print("Please create a .env file with your ROBOFLOW_API_KEY.")
        print("See .env.example for reference.")
        sys.exit(1)

# Run validation on import to ensure we fail fast
# Optional: we can make this explicit if preferred, but fail-fast is usually good for scripts.
# For now, we'll just expose the variable and let the consumer decide when to validate,
# or we can warn.
if not ROBOFLOW_API_KEY:
    # Use a warning instead of exit on import, so importing config doesn't crash 
    # if we are just running unit tests or similar that might mock it.
    pass 

