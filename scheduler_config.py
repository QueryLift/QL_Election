"""
Election Response Logger Scheduler Configuration
"""

# Schedule settings - when to run the comprehensive logging
SCHEDULE_TIMES = [
    "02:00",  # Early morning
    "14:00",  # Afternoon
    "22:00"   # Evening
]

# Log settings
LOG_LEVEL = "INFO"
LOG_FILE = "election_scheduler.log"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

# Execution interval settings (seconds)
CHECK_INTERVAL = 60  # Check schedule every minute

# Error handling settings
MAX_RETRIES = 3
RETRY_DELAY = 300  # 5 minutes

# Debug mode
DEBUG_MODE = False

# Execution settings
ENABLE_PARTIES = True      # Run party response logging
ENABLE_CANDIDATES = True   # Run candidate response logging  
ENABLE_OPEN_QUESTIONS = True  # Run open question response logging

# Rate limiting (seconds between different logging phases)
PHASE_DELAY = 30  # 30 seconds between party/candidate/open question logging