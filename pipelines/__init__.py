# Add 'data_prep' to the path so Python can find 'modules'
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()