#!/usr/bin/env python3
"""
Election Response Logger Scheduler
Runs comprehensive response logging on a schedule
"""

import schedule
import time
import logging
import sys
from datetime import datetime
from log_response import log_all_responses, log_responses_for_all_parties, log_responses_for_all_candidates, log_responses_for_open_questions
from scheduler_config import *

# Log configuration
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

def run():
    """Main execution function"""
    try:
        logging.info("=== Election Response Logger Execution Started ===")
        start_time = datetime.now()
        
        # Run party response logging
        if ENABLE_PARTIES:
            try:
                logging.info("== Starting Party Response Logging ==")
                log_responses_for_all_parties()
                logging.info("== Party Response Logging Completed ==")
                if ENABLE_CANDIDATES or ENABLE_OPEN_QUESTIONS:
                    time.sleep(PHASE_DELAY)
            except Exception as e:
                logging.error(f"Error in party response logging: {str(e)}")
                if DEBUG_MODE:
                    import traceback
                    logging.error(traceback.format_exc())
        
        # Run candidate response logging
        if ENABLE_CANDIDATES:
            try:
                logging.info("== Starting Candidate Response Logging ==")
                log_responses_for_all_candidates()
                logging.info("== Candidate Response Logging Completed ==")
                if ENABLE_OPEN_QUESTIONS:
                    time.sleep(PHASE_DELAY)
            except Exception as e:
                logging.error(f"Error in candidate response logging: {str(e)}")
                if DEBUG_MODE:
                    import traceback
                    logging.error(traceback.format_exc())
        
        # Run open question response logging
        if ENABLE_OPEN_QUESTIONS:
            try:
                logging.info("== Starting Open Question Response Logging ==")
                log_responses_for_open_questions()
                logging.info("== Open Question Response Logging Completed ==")
            except Exception as e:
                logging.error(f"Error in open question response logging: {str(e)}")
                if DEBUG_MODE:
                    import traceback
                    logging.error(traceback.format_exc())
        
        end_time = datetime.now()
        duration = end_time - start_time
        logging.info(f"=== Election Response Logger Execution Completed in {duration} ===")
        
    except Exception as e:
        logging.error(f"Critical error during scheduler execution: {str(e)}")
        if DEBUG_MODE:
            import traceback
            logging.error(traceback.format_exc())

def run_with_retry():
    """Run with retry mechanism"""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            run()
            return  # Success, exit retry loop
        except Exception as e:
            logging.error(f"Attempt {attempt}/{MAX_RETRIES} failed: {str(e)}")
            if attempt < MAX_RETRIES:
                logging.info(f"Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                logging.error("All retry attempts failed")

def schedule_jobs():
    """Configure scheduled jobs"""
    for time_str in SCHEDULE_TIMES:
        schedule.every().day.at(time_str).do(run_with_retry)
    
    logging.info("Scheduled jobs configured:")
    for time_str in SCHEDULE_TIMES:
        logging.info(f"- Daily at {time_str}")

def run_scheduler():
    """Start the scheduler"""
    logging.info("=== Starting Election Response Logger Scheduler ===")
    logging.info(f"Configuration: Check interval={CHECK_INTERVAL}s, Log level={LOG_LEVEL}")
    logging.info(f"Enabled modules: Parties={ENABLE_PARTIES}, Candidates={ENABLE_CANDIDATES}, Open Questions={ENABLE_OPEN_QUESTIONS}")
    
    schedule_jobs()
    
    while True:
        try:
            schedule.run_pending()
            time.sleep(CHECK_INTERVAL)
        except KeyboardInterrupt:
            logging.info("=== Stopping Election Response Logger Scheduler ===")
            break
        except Exception as e:
            logging.error(f"Error in scheduler loop: {str(e)}")
            if DEBUG_MODE:
                import traceback
                logging.error(traceback.format_exc())
            time.sleep(CHECK_INTERVAL)

def show_schedule():
    """Display current schedule"""
    print("=== Current Schedule ===")
    for job in schedule.jobs:
        print(f"- {job}")

def show_status():
    """Display current configuration status"""
    print("=== Election Response Logger Status ===")
    print(f"Schedule times: {', '.join(SCHEDULE_TIMES)}")
    print(f"Check interval: {CHECK_INTERVAL} seconds")
    print(f"Log level: {LOG_LEVEL}")
    print(f"Log file: {LOG_FILE}")
    print(f"Debug mode: {DEBUG_MODE}")
    print(f"Max retries: {MAX_RETRIES}")
    print(f"Retry delay: {RETRY_DELAY} seconds")
    print("Enabled modules:")
    print(f"- Parties: {ENABLE_PARTIES}")
    print(f"- Candidates: {ENABLE_CANDIDATES}")
    print(f"- Open Questions: {ENABLE_OPEN_QUESTIONS}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--once":
            # Run once immediately
            print("Running election response logging once...")
            run()
        elif sys.argv[1] == "--show":
            # Show schedule
            schedule_jobs()
            show_schedule()
        elif sys.argv[1] == "--status":
            # Show configuration status
            show_status()
        elif sys.argv[1] == "--test":
            # Test run with detailed output
            print("Running test execution...")
            DEBUG_MODE = True
            run()
        elif sys.argv[1] == "--help":
            # Show help
            print("Election Response Logger Scheduler")
            print("Usage:")
            print("  python run.py              # Run in scheduler mode")
            print("  python run.py --once       # Run once immediately")
            print("  python run.py --show       # Show current schedule")
            print("  python run.py --status     # Show configuration status")
            print("  python run.py --test       # Run once with debug output")
            print("  python run.py --help       # Show this help")
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Use 'python run.py --help' for usage information")
    else:
        # Run in scheduler mode
        run_scheduler()