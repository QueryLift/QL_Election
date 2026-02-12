import os
import logging
from datetime import datetime
from sqlalchemy import create_engine, and_, or_, func, desc
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from create_db import Base, Model, Prompt, Response, Party, Candidate

load_dotenv()

class ErrorChecker:
    def __init__(self):
        self.engine = create_engine(os.getenv("POSTGRE_SERVER_ADRESS"))
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('error_checker.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def check_model_errors(self, model_name=None):
        """
        Check for errors in model responses by analyzing response patterns and database integrity
        """
        self.logger.info(f"Starting model error check for: {model_name or 'all models'}")
        
        query = self.session.query(Response).join(Model).join(Prompt)
        
        if model_name:
            query = query.filter(Model.name == model_name)
        
        responses = query.all()
        
        errors = []
        
        for response in responses:
            # Check for empty or invalid responses
            if not response.content or response.content.strip() == "":
                errors.append({
                    "type": "empty_response",
                    "response_id": response.id,
                    "model": response.model.name,
                    "prompt_id": response.prompt_id,
                    "prompt_content": response.prompt.content[:100] + "..." if len(response.prompt.content) > 100 else response.prompt.content,
                    "error_message": "Response content is empty or only whitespace"
                })
            
            # Check for responses that are too short (likely errors)
            if response.content and len(response.content.strip()) < 20:
                errors.append({
                    "type": "short_response",
                    "response_id": response.id,
                    "model": response.model.name,
                    "prompt_id": response.prompt_id,
                    "prompt_content": response.prompt.content[:100] + "..." if len(response.prompt.content) > 100 else response.prompt.content,
                    "response_content": response.content,
                    "error_message": f"Response too short ({len(response.content)} chars)"
                })
            
            # Check for common error patterns in responses
            if response.content:
                error_patterns = [
                    "error", "Error", "ERROR",
                    "exception", "Exception", "EXCEPTION", 
                    "failed", "Failed", "FAILED",
                    "timeout", "Timeout", "TIMEOUT",
                    "invalid", "Invalid", "INVALID",
                    "cannot", "Cannot", "CANNOT",
                    "unable", "Unable", "UNABLE"
                ]
                
                for pattern in error_patterns:
                    if pattern in response.content:
                        errors.append({
                            "type": "error_pattern",
                            "response_id": response.id,
                            "model": response.model.name,
                            "prompt_id": response.prompt_id,
                            "prompt_content": response.prompt.content[:100] + "..." if len(response.prompt.content) > 100 else response.prompt.content,
                            "response_content": response.content[:200] + "..." if len(response.content) > 200 else response.content,
                            "error_message": f"Response contains error pattern: '{pattern}'"
                        })
                        break  # Only report first error pattern found
            
            # Check for null/missing sentiment scores (indicates analysis errors)
            if response.sentiment is None:
                errors.append({
                    "type": "missing_sentiment",
                    "response_id": response.id,
                    "model": response.model.name,
                    "prompt_id": response.prompt_id,
                    "prompt_content": response.prompt.content[:100] + "..." if len(response.prompt.content) > 100 else response.prompt.content,
                    "error_message": "Sentiment analysis failed or missing"
                })
            
            # Check for extreme sentiment values (potential analysis errors)
            if response.sentiment is not None and (response.sentiment < -1.0 or response.sentiment > 1.0):
                errors.append({
                    "type": "invalid_sentiment",
                    "response_id": response.id,
                    "model": response.model.name,
                    "prompt_id": response.prompt_id,
                    "prompt_content": response.prompt.content[:100] + "..." if len(response.prompt.content) > 100 else response.prompt.content,
                    "error_message": f"Invalid sentiment value: {response.sentiment}"
                })
        
        self.logger.info(f"Found {len(errors)} errors in model responses")
        return errors
    
    def check_prompt_errors(self, prompt_type=None):
        """
        Check for errors in prompts by analyzing database integrity and content
        """
        self.logger.info(f"Starting prompt error check for type: {prompt_type or 'all types'}")
        
        query = self.session.query(Prompt)
        
        if prompt_type:
            query = query.filter(Prompt.prompt_type == prompt_type)
        
        prompts = query.all()
        
        errors = []
        
        for prompt in prompts:
            # Check for empty or invalid prompts
            if not prompt.content or prompt.content.strip() == "":
                errors.append({
                    "type": "empty_prompt",
                    "prompt_id": prompt.id,
                    "prompt_type": prompt.prompt_type,
                    "party_id": prompt.party_id,
                    "candidate_id": prompt.candidate_id,
                    "error_message": "Prompt content is empty or only whitespace"
                })
            
            # Check for unresolved placeholders in prompts
            if prompt.content:
                placeholders = ["【政党名】", "【党首名】", "［候補者名］"]
                for placeholder in placeholders:
                    if placeholder in prompt.content:
                        errors.append({
                            "type": "unresolved_placeholder",
                            "prompt_id": prompt.id,
                            "prompt_type": prompt.prompt_type,
                            "party_id": prompt.party_id,
                            "candidate_id": prompt.candidate_id,
                            "prompt_content": prompt.content[:100] + "..." if len(prompt.content) > 100 else prompt.content,
                            "error_message": f"Unresolved placeholder: '{placeholder}'"
                        })
            
            # Check for orphaned prompts (missing party/candidate references)
            if prompt.prompt_type == 1 and not prompt.party_id:  # Party-specific prompt without party
                errors.append({
                    "type": "orphaned_prompt",
                    "prompt_id": prompt.id,
                    "prompt_type": prompt.prompt_type,
                    "error_message": "Party-specific prompt missing party_id"
                })
            
            if prompt.prompt_type == 2 and not prompt.candidate_id:  # Candidate-specific prompt without candidate
                errors.append({
                    "type": "orphaned_prompt",
                    "prompt_id": prompt.id,
                    "prompt_type": prompt.prompt_type,
                    "error_message": "Candidate-specific prompt missing candidate_id"
                })
        
        self.logger.info(f"Found {len(errors)} errors in prompts")
        return errors
    
    def check_database_integrity(self):
        """
        Check database integrity and consistency
        """
        self.logger.info("Starting database integrity check")
        
        errors = []
        
        # Check for responses without valid models
        invalid_model_responses = self.session.query(Response).filter(
            ~Response.ai_model_id.in_(
                self.session.query(Model.id)
            )
        ).all()
        
        for response in invalid_model_responses:
            errors.append({
                "type": "invalid_model_reference",
                "response_id": response.id,
                "model_id": response.ai_model_id,
                "error_message": f"Response references non-existent model ID: {response.ai_model_id}"
            })
        
        # Check for responses without valid prompts
        invalid_prompt_responses = self.session.query(Response).filter(
            ~Response.prompt_id.in_(
                self.session.query(Prompt.id)
            )
        ).all()
        
        for response in invalid_prompt_responses:
            errors.append({
                "type": "invalid_prompt_reference",
                "response_id": response.id,
                "prompt_id": response.prompt_id,
                "error_message": f"Response references non-existent prompt ID: {response.prompt_id}"
            })
        
        # Check for prompts with invalid party references
        invalid_party_prompts = self.session.query(Prompt).filter(
            and_(
                Prompt.party_id.isnot(None),
                ~Prompt.party_id.in_(
                    self.session.query(Party.id)
                )
            )
        ).all()
        
        for prompt in invalid_party_prompts:
            errors.append({
                "type": "invalid_party_reference",
                "prompt_id": prompt.id,
                "party_id": prompt.party_id,
                "error_message": f"Prompt references non-existent party ID: {prompt.party_id}"
            })
        
        # Check for prompts with invalid candidate references
        invalid_candidate_prompts = self.session.query(Prompt).filter(
            and_(
                Prompt.candidate_id.isnot(None),
                ~Prompt.candidate_id.in_(
                    self.session.query(Candidate.id)
                )
            )
        ).all()
        
        for prompt in invalid_candidate_prompts:
            errors.append({
                "type": "invalid_candidate_reference",
                "prompt_id": prompt.id,
                "candidate_id": prompt.candidate_id,
                "error_message": f"Prompt references non-existent candidate ID: {prompt.candidate_id}"
            })
        
        self.logger.info(f"Found {len(errors)} database integrity errors")
        return errors
    
    def get_model_performance_stats(self):
        """
        Get performance statistics for each model to identify problematic models
        """
        self.logger.info("Generating model performance statistics")
        
        stats = []
        
        models = self.session.query(Model).all()
        
        for model in models:
            # Count total responses
            total_responses = self.session.query(Response).filter(
                Response.ai_model_id == model.id
            ).count()
            
            # Count responses with errors (empty or very short)
            error_responses = self.session.query(Response).filter(
                and_(
                    Response.ai_model_id == model.id,
                    or_(
                        Response.content.is_(None),
                        Response.content == "",
                        func.length(Response.content) < 20
                    )
                )
            ).count()
            
            # Count responses with missing sentiment
            missing_sentiment = self.session.query(Response).filter(
                and_(
                    Response.ai_model_id == model.id,
                    Response.sentiment.is_(None)
                )
            ).count()
            
            # Calculate average sentiment
            avg_sentiment = self.session.query(func.avg(Response.sentiment)).filter(
                and_(
                    Response.ai_model_id == model.id,
                    Response.sentiment.isnot(None)
                )
            ).scalar()
            
            # Calculate average response length
            avg_length = self.session.query(func.avg(func.length(Response.content))).filter(
                and_(
                    Response.ai_model_id == model.id,
                    Response.content.isnot(None)
                )
            ).scalar()
            
            stats.append({
                "model_name": model.name,
                "total_responses": total_responses,
                "error_responses": error_responses,
                "error_rate": (error_responses / total_responses * 100) if total_responses > 0 else 0,
                "missing_sentiment": missing_sentiment,
                "avg_sentiment": float(avg_sentiment) if avg_sentiment else None,
                "avg_response_length": float(avg_length) if avg_length else None
            })
        
        return stats
    
    def generate_error_report(self, output_file="error_report.txt"):
        """
        Generate a comprehensive error report
        """
        self.logger.info(f"Generating comprehensive error report: {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("ELECTION DATABASE ERROR REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            # Model errors
            f.write("MODEL ERRORS\n")
            f.write("-" * 40 + "\n")
            model_errors = self.check_model_errors()
            if model_errors:
                for error in model_errors:
                    f.write(f"Type: {error['type']}\n")
                    f.write(f"Response ID: {error['response_id']}\n")
                    f.write(f"Model: {error['model']}\n")
                    f.write(f"Error: {error['error_message']}\n")
                    if 'response_content' in error:
                        f.write(f"Content: {error['response_content']}\n")
                    f.write("-" * 40 + "\n")
            else:
                f.write("No model errors found.\n")
            
            f.write("\n")
            
            # Prompt errors
            f.write("PROMPT ERRORS\n")
            f.write("-" * 40 + "\n")
            prompt_errors = self.check_prompt_errors()
            if prompt_errors:
                for error in prompt_errors:
                    f.write(f"Type: {error['type']}\n")
                    f.write(f"Prompt ID: {error['prompt_id']}\n")
                    f.write(f"Error: {error['error_message']}\n")
                    if 'prompt_content' in error:
                        f.write(f"Content: {error['prompt_content']}\n")
                    f.write("-" * 40 + "\n")
            else:
                f.write("No prompt errors found.\n")
            
            f.write("\n")
            
            # Database integrity errors
            f.write("DATABASE INTEGRITY ERRORS\n")
            f.write("-" * 40 + "\n")
            db_errors = self.check_database_integrity()
            if db_errors:
                for error in db_errors:
                    f.write(f"Type: {error['type']}\n")
                    f.write(f"Error: {error['error_message']}\n")
                    f.write("-" * 40 + "\n")
            else:
                f.write("No database integrity errors found.\n")
            
            f.write("\n")
            
            # Model performance stats
            f.write("MODEL PERFORMANCE STATISTICS\n")
            f.write("-" * 40 + "\n")
            stats = self.get_model_performance_stats()
            for stat in stats:
                f.write(f"Model: {stat['model_name']}\n")
                f.write(f"Total Responses: {stat['total_responses']}\n")
                f.write(f"Error Responses: {stat['error_responses']}\n")
                f.write(f"Error Rate: {stat['error_rate']:.2f}%\n")
                f.write(f"Missing Sentiment: {stat['missing_sentiment']}\n")
                avg_sentiment_str = f"{stat['avg_sentiment']:.3f}" if stat['avg_sentiment'] is not None else "N/A"
                avg_length_str = f"{stat['avg_response_length']:.1f}" if stat['avg_response_length'] is not None else "N/A"
                f.write(f"Avg Sentiment: {avg_sentiment_str}\n")
                f.write(f"Avg Response Length: {avg_length_str} chars\n")
                f.write("-" * 40 + "\n")
        
        self.logger.info(f"Error report generated: {output_file}")
        print(f"Error report generated: {output_file}")
        
        return {
            "model_errors": len(model_errors),
            "prompt_errors": len(prompt_errors),
            "db_errors": len(db_errors),
            "total_errors": len(model_errors) + len(prompt_errors) + len(db_errors)
        }

def main():
    """
    Main function to run error checking
    """
    checker = ErrorChecker()
    
    print("Election Database Error Checker")
    print("1. Check model errors")
    print("2. Check prompt errors")
    print("3. Check database integrity")
    print("4. Generate full error report")
    print("5. Show model performance stats")
    
    choice = input("Select option (1-5): ")
    
    if choice == "1":
        model_name = input("Enter model name (or press Enter for all): ").strip()
        if not model_name:
            model_name = None
        errors = checker.check_model_errors(model_name)
        print(f"Found {len(errors)} model errors")
        for error in errors[:10]:  # Show first 10 errors
            print(f"- {error['type']}: {error['error_message']}")
    
    elif choice == "2":
        errors = checker.check_prompt_errors()
        print(f"Found {len(errors)} prompt errors")
        for error in errors[:10]:  # Show first 10 errors
            print(f"- {error['type']}: {error['error_message']}")
    
    elif choice == "3":
        errors = checker.check_database_integrity()
        print(f"Found {len(errors)} database integrity errors")
        for error in errors[:10]:  # Show first 10 errors
            print(f"- {error['type']}: {error['error_message']}")
    
    elif choice == "4":
        summary = checker.generate_error_report()
        print(f"Error report generated with {summary['total_errors']} total errors")
    
    elif choice == "5":
        stats = checker.get_model_performance_stats()
        print("\nModel Performance Statistics:")
        print("-" * 80)
        for stat in stats:
            print(f"Model: {stat['model_name']}")
            print(f"  Total Responses: {stat['total_responses']}")
            print(f"  Error Rate: {stat['error_rate']:.2f}%")
            avg_sentiment_str = f"{stat['avg_sentiment']:.3f}" if stat['avg_sentiment'] is not None else "N/A"
            avg_length_str = f"{stat['avg_response_length']:.1f}" if stat['avg_response_length'] is not None else "N/A"
            print(f"  Avg Sentiment: {avg_sentiment_str}")
            print(f"  Avg Length: {avg_length_str} chars")
            print("-" * 40)
    
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()