import os
from dotenv import load_dotenv

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    print("Hello from llm-evaluations!")
    print(f"Log level: {os.getenv('LOG_LEVEL', 'INFO')}")
    print(f"Results directory: {os.getenv('RESULTS_DIR', 'results')}")


if __name__ == "__main__":
    main()
