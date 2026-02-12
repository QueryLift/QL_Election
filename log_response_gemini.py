import os
import time
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from create_db import Base, Party, Candidate, Model, Prompt, Response, ResponseSource, ResponseCitation, PartyResponseMention, PartyCitationMention, Category
from generative_search import GenManager
from analysis import sentiment_analysis, PMR, citation_rate, check_citation_mentions

load_dotenv()

class DBManager:
    def __init__(self):
        self.engine = create_engine(os.getenv("POSTGRE_SERVER_ADRESS"))
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
    
    def get_party_info(self, party_id):
        return self.session.query(Party).filter(Party.id == party_id).first()
    
    def get_candidate_info(self, candidate_id):
        return self.session.query(Candidate).filter(Candidate.id == candidate_id).first()
    
    def get_all_parties(self):
        return self.session.query(Party).filter(Party.is_active == True).all()
    
    def get_all_candidates(self):
        return self.session.query(Candidate).all()
    
    def get_all_models(self):
        return [self.session.query(Model).all()[1]]
    
    def get_active_prompts_by_party(self, party_id):
        return self.session.query(Prompt).filter(
            Prompt.party_id == party_id,
            Prompt.is_active == True
        ).all()
    
    def get_active_prompts_by_candidate(self, candidate_id):
        return self.session.query(Prompt).filter(
            Prompt.candidate_id == candidate_id,
            Prompt.is_active == True
        ).all()
    
    def get_prompts_by_type(self, prompt_type_id, party_id=None, candidate_id=None):
        """Get all active prompts filtered by prompt_type with optional party/candidate filtering"""
        query = self.session.query(Prompt).filter(
            Prompt.prompt_type == prompt_type_id,
            Prompt.is_active == True
        )
        
        # For prompt_type_id=1 (party-specific), filter by party_id if provided
        if prompt_type_id == 1 and party_id is not None:
            query = query.filter(Prompt.party_id == party_id)
        
        # For prompt_type_id=2 (candidate-specific), filter by candidate_id if provided
        elif prompt_type_id == 2 and candidate_id is not None:
            query = query.filter(Prompt.candidate_id == candidate_id)
        
        return query.all()
    
    
    
    def create_response(self, prompt_id, content, model_id, usage=None, search_query=None, 
                      sentiment=None, party_mention_rate=None, semantic_negentropy=None,
                      noncontradiction=None, exact_match=None, cosine_sim=None, 
                      bert_score=None, bleurt=None):
        response = Response(
            prompt_id=prompt_id,
            content=content,
            ai_model_id=model_id,
            usage=usage,
            search_query=search_query,
            sentiment=sentiment,
            party_mention_rate=party_mention_rate,
            semantic_negentropy=semantic_negentropy,
            noncontradiction=noncontradiction,
            exact_match=exact_match,
            cosine_sim=cosine_sim,
            bert_score=bert_score,
            bleurt=bleurt
        )
        self.session.add(response)
        self.session.commit()
        return response
    
    def create_response_source(self, response_id, url, is_cited=False):
        source = ResponseSource(
            response_id=response_id,
            url=url,
            is_cited=is_cited
        )
        self.session.add(source)
        self.session.commit()
        return source
    
    def create_response_citation(self, response_id, url, citation_ratio=None, 
                               text_w_citations=None, sentiment=None):
        citation = ResponseCitation(
            response_id=response_id,
            url=url,
            citation_ratio=citation_ratio,
            text_w_citations=text_w_citations,
            sentiment=sentiment
        )
        self.session.add(citation)
        self.session.commit()
        return citation
    
    def create_party_response_mention(self, response_id, party_id):
        mention = PartyResponseMention(
            response_id=response_id,
            party_id=party_id
        )
        self.session.add(mention)
        self.session.commit()
        return mention
    
    def create_party_citation_mention(self, citation_id, party_id):
        mention = PartyCitationMention(
            citation_id=citation_id,
            party_id=party_id
        )
        self.session.add(mention)
        self.session.commit()
        return mention


class AnalysisManager:
    def __init__(self):
        pass
    
    def analyze_sentiment(self, text):
        """Use Google Cloud sentiment analysis"""
        try:
            return sentiment_analysis(text)
        except Exception as e:
            print(f"Sentiment analysis failed: {e}")
            return 0.0  # Neutral sentiment fallback
    
    def calculate_party_mention_rate(self, text, party_names):
        """Calculate party mention rate using PMR function"""
        try:
            # Convert party names list to compatible format for BMR function
            total_mentions = 0
            text_lower = text.lower()
            for party_name in party_names:
                if party_name.lower() in text_lower:
                    total_mentions += 1
            return total_mentions / len(party_names) if party_names else 0.0
        except Exception as e:
            print(f"BMR calculation failed: {e}")
            return 0.0
    
    def detect_party_mentions(self, text, parties):
        """Detect which parties are mentioned in the text"""
        mentioned_parties = []
        text_lower = text.lower()
        for party in parties:
            if party.name.lower() in text_lower:
                mentioned_parties.append(party)
        return mentioned_parties
    
    def process_citations_with_analysis(self, citations, response_text, all_parties):
        """Process citations using analysis.py citation_rate function"""
        try:
            if not citations:
                return []
            
            # Convert citations to the format expected by citation_rate
            citation_list = []
            for citation in citations:
                citation_list.append({
                    "url": citation.get("url", ""),
                    "start_index": citation.get("start_index", 0),
                    "end_index": citation.get("end_index", 0),
                    "text_w_citations": citation.get("text", "")
                })
            
            # Calculate citation rates
            citation_results = citation_rate(response_text, citation_list)
            
            # Check for party mentions in citations
            for result in citation_results:
                party_names = [p.name for p in all_parties]
                mentioned_brands = check_citation_mentions(
                    result["text_w_citations"], 
                    "", 
                    party_names
                )
                result["mentioned_parties"] = mentioned_brands
            
            return citation_results
        except Exception as e:
            print(f"Citation analysis failed: {e}")
            return []

def log_response_for_party(party_id):
    """
    Lightweight version of party response logging without UQ processing
    """
    print(f"Starting lite response logging for party ID: {party_id}")
    start_time = time.time()
    
    # Initialize managers
    dbmgr = DBManager()
    genmgr = GenManager()
    analysis_mgr = AnalysisManager()
    
    # Get party information
    party = dbmgr.get_party_info(party_id)
    if not party:
        print(f"Party with ID {party_id} not found")
        return
    
    print(f"Processing party: {party.name}")
    
    # Get all parties for mention detection
    all_parties = dbmgr.get_all_parties()
    party_names = [p.name for p in all_parties]
    
    # Get all available models
    models = dbmgr.get_all_models()
    
    # Get or create active prompts for this party
    prompts = dbmgr.get_active_prompts_by_party(party_id=party_id)
    if not prompts:
        print(f"No prompts could be created for party {party.name}")
        return
    
    print(f"Found {len(prompts)} active prompts and {len(models)} models")
    
    # Process each prompt with each model
    for prompt in prompts:
        print(f"Processing prompt: {prompt.content[:50]}...")
        
        # Replace placeholder with actual party name
        prompt_content = prompt.content.replace("【政党名】", party.name)
        
        for model in models:
            print(f"  Generating response with model: {model.name}")
            
            try:
                # Generate AI response
                ai_response = genmgr.generate_response(
                    prompt_content,
                    model.name
                )
                
                # Analyze sentiment
                sentiment = analysis_mgr.analyze_sentiment(ai_response["response_text"])
                
                # Calculate brand mention rate
                bmr = analysis_mgr.calculate_party_mention_rate(
                    ai_response["response_text"], 
                    party_names
                )
                
                # Create response record WITHOUT UQ processing - set all UQ fields to None
                response = dbmgr.create_response(
                    prompt_id=prompt.id,
                    content=ai_response["response_text"],
                    model_id=model.id,
                    usage=ai_response.get("usage"),
                    search_query=ai_response.get("search_query"),
                    sentiment=sentiment,
                    party_mention_rate=bmr,
                    semantic_negentropy=None,  # Skip UQ processing
                    noncontradiction=None,     # Skip UQ processing
                    exact_match=None,          # Skip UQ processing
                    cosine_sim=None,           # Skip UQ processing
                    bert_score=None,           # Skip UQ processing
                    bleurt=None,               # Skip UQ processing
                )
                
                # Process sources
                sources = ai_response.get("source", [])
                if sources:
                    for source_data in sources:
                        source_url = source_data.get("url") if isinstance(source_data, dict) else source_data
                        if source_url:
                            dbmgr.create_response_source(
                                response_id=response.id,
                                url=source_url,
                                is_cited=True
                            )
                
                # Process citations with enhanced analysis
                citations = ai_response.get("citations", [])
                if citations:
                    # Use advanced citation analysis
                    citation_results = analysis_mgr.process_citations_with_analysis(
                        citations, 
                        ai_response["response_text"], 
                        all_parties
                    )
                    
                    for citation_data, citation_result in zip(citations, citation_results):
                        citation_url = citation_data.get("url") if isinstance(citation_data, dict) else citation_data
                        citation_text = citation_data.get("text", "") if isinstance(citation_data, dict) else ""
                        
                        # Use citation ratio from analysis
                        citation_ratio = citation_result.get("citation_ratio", 0.0) if citation_result else 0.0
                        
                        citation_record = dbmgr.create_response_citation(
                            response_id=response.id,
                            url=citation_url,
                            citation_ratio=citation_ratio,
                            text_w_citations=citation_text,
                            sentiment=analysis_mgr.analyze_sentiment(citation_text) if citation_text else None
                        )
                        
                        # Detect party mentions in citations using enhanced method
                        if citation_text:
                            mentioned_parties = analysis_mgr.detect_party_mentions(
                                citation_text, 
                                all_parties
                            )
                            
                            for mentioned_party in mentioned_parties:
                                dbmgr.create_party_citation_mention(
                                    citation_id=citation_record.id,
                                    party_id=mentioned_party.id
                                )
                
                # Detect party mentions in main response
                mentioned_parties = analysis_mgr.detect_party_mentions(
                    ai_response["response_text"], 
                    all_parties
                )
                
                for mentioned_party in mentioned_parties:
                    dbmgr.create_party_response_mention(
                        response_id=response.id,
                        party_id=mentioned_party.id
                    )
                
                print(f"    Response logged successfully (ID: {response.id}) [LITE - No UQ]")
                
            except Exception as e:
                print(f"    Error generating response: {str(e)}")
                continue
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"Lite response logging completed for party {party.name} in {duration:.2f} seconds")

def log_response_for_candidate(candidate_id):
    """
    Lightweight version of candidate response logging without UQ processing
    """
    print(f"Starting lite response logging for candidate ID: {candidate_id}")
    start_time = time.time()
    
    # Initialize managers
    dbmgr = DBManager()
    genmgr = GenManager()
    analysis_mgr = AnalysisManager()
    
    # Get candidate information
    candidate = dbmgr.get_candidate_info(candidate_id)
    if not candidate:
        print(f"Candidate with ID {candidate_id} not found")
        return
    
    print(f"Processing candidate: {candidate.name}")
    
    # Get all parties for mention detection
    all_parties = dbmgr.get_all_parties()
    party_names = [p.name for p in all_parties]
    
    # Get all available models
    models = dbmgr.get_all_models()
    
    # Get or create active prompts for this candidate
    prompts = dbmgr.get_active_prompts_by_candidate(candidate_id=candidate_id)
    
    if not prompts:
        print(f"No prompts could be created for candidate {candidate.name}")
        return
    
    print(f"Found {len(prompts)} active prompts and {len(models)} models")
    
    # Process each prompt with each model
    for prompt in prompts:
        print(f"Processing prompt: {prompt.content[:50]}...")
        
        # Replace placeholder with actual candidate name
        prompt_content = prompt.content.replace("［候補者名］", candidate.name)
        
        for model in models:
            print(f"  Generating response with model: {model.name}")
            
            try:
                # Generate AI response
                ai_response = genmgr.generate_response(
                    prompt_content,
                    model.name
                )
                
                # Analyze sentiment
                sentiment = analysis_mgr.analyze_sentiment(ai_response["response_text"])
                
                # Calculate brand mention rate (for parties mentioned)
                bmr = analysis_mgr.calculate_party_mention_rate(
                    ai_response["response_text"], 
                    party_names
                )
                
                # Create response record WITHOUT UQ processing - set all UQ fields to None
                response = dbmgr.create_response(
                    prompt_id=prompt.id,
                    content=ai_response["response_text"],
                    model_id=model.id,
                    usage=ai_response.get("usage"),
                    search_query=ai_response.get("search_query"),
                    sentiment=sentiment,
                    party_mention_rate=bmr,
                    semantic_negentropy=None,  # Skip UQ processing
                    noncontradiction=None,     # Skip UQ processing
                    exact_match=None,          # Skip UQ processing
                    cosine_sim=None,           # Skip UQ processing
                    bert_score=None,           # Skip UQ processing
                    bleurt=None,               # Skip UQ processing
                )
                
                # Process sources with enhanced handling
                sources = ai_response.get("source", [])
                if sources:
                    for source_data in sources:
                        source_url = source_data.get("url") if isinstance(source_data, dict) else source_data
                        if source_url:
                            dbmgr.create_response_source(
                                response_id=response.id,
                                url=source_url,
                                is_cited=True
                            )
                
                # Process citations with enhanced analysis
                citations = ai_response.get("citations", [])
                if citations:
                    # Use advanced citation analysis
                    citation_results = analysis_mgr.process_citations_with_analysis(
                        citations, 
                        ai_response["response_text"], 
                        all_parties
                    )
                    
                    for citation_data, citation_result in zip(citations, citation_results):
                        citation_url = citation_data.get("url") if isinstance(citation_data, dict) else citation_data
                        citation_text = citation_data.get("text", "") if isinstance(citation_data, dict) else ""
                        
                        # Use citation ratio from analysis
                        citation_ratio = citation_result.get("citation_ratio", 0.0) if citation_result else 0.0
                        
                        citation_record = dbmgr.create_response_citation(
                            response_id=response.id,
                            url=citation_url,
                            citation_ratio=citation_ratio,
                            text_w_citations=citation_text,
                            sentiment=analysis_mgr.analyze_sentiment(citation_text) if citation_text else None
                        )
                        
                        # Detect party mentions in citations using enhanced method
                        if citation_text:
                            mentioned_parties = analysis_mgr.detect_party_mentions(
                                citation_text, 
                                all_parties
                            )
                            
                            for mentioned_party in mentioned_parties:
                                dbmgr.create_party_citation_mention(
                                    citation_id=citation_record.id,
                                    party_id=mentioned_party.id
                                )
                
                print(f"    Response logged successfully (ID: {response.id}) [LITE - No UQ]")
                
            except Exception as e:
                print(f"    Error generating response: {str(e)}")
                continue
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"Lite response logging completed for candidate {candidate.name} in {duration:.2f} seconds")

def log_responses_for_open_questions():
    """
    Lightweight version of open question response logging without UQ processing
    """
    print("Starting lite response logging for open questions")
    start_time = time.time()
    
    # Initialize managers
    dbmgr = DBManager()
    genmgr = GenManager()
    analysis_mgr = AnalysisManager()
    
    # Get all parties for mention detection
    all_parties = dbmgr.get_all_parties()
    party_names = [p.name for p in all_parties]
    
    # Get all available models
    models = dbmgr.get_all_models()
    
    # Get all open question prompts (prompt_type_id=3)
    prompts = dbmgr.get_prompts_by_type(prompt_type_id=3)
    
    if not prompts:
        print("No open question prompts found")
        return
    
    print(f"Found {len(prompts)} open question prompts and {len(models)} models")
    
    # Process each prompt with each model
    for prompt in prompts:
        print(f"Processing prompt: {prompt.content[:50]}...")
        
        for model in models:
            print(f"  Generating response with model: {model.name}")
            
            try:
                # Generate AI response
                ai_response = genmgr.generate_response(
                    prompt.content,
                    model.name
                )
                
                # Analyze sentiment
                sentiment = analysis_mgr.analyze_sentiment(ai_response["response_text"])
                
                # Calculate brand mention rate
                bmr = analysis_mgr.calculate_party_mention_rate(
                    ai_response["response_text"], 
                    party_names
                )
                
                # Create response record WITHOUT UQ processing - set all UQ fields to None
                response = dbmgr.create_response(
                    prompt_id=prompt.id,
                    content=ai_response["response_text"],
                    model_id=model.id,
                    usage=ai_response.get("usage"),
                    search_query=ai_response.get("search_query"),
                    sentiment=sentiment,
                    party_mention_rate=bmr,
                    semantic_negentropy=None,  # Skip UQ processing
                    noncontradiction=None,     # Skip UQ processing
                    exact_match=None,          # Skip UQ processing
                    cosine_sim=None,           # Skip UQ processing
                    bert_score=None,           # Skip UQ processing
                    bleurt=None,               # Skip UQ processing
                )
                
                # Process sources
                sources = ai_response.get("source", [])
                if sources:
                    for source_data in sources:
                        source_url = source_data.get("url") if isinstance(source_data, dict) else source_data
                        if source_url:
                            dbmgr.create_response_source(
                                response_id=response.id,
                                url=source_url,
                                is_cited=True
                            )
                
                # Process citations with enhanced analysis
                citations = ai_response.get("citations", [])
                if citations:
                    # Use advanced citation analysis
                    citation_results = analysis_mgr.process_citations_with_analysis(
                        citations, 
                        ai_response["response_text"], 
                        all_parties
                    )
                    
                    for citation_data, citation_result in zip(citations, citation_results):
                        citation_url = citation_data.get("url") if isinstance(citation_data, dict) else citation_data
                        citation_text = citation_data.get("text", "") if isinstance(citation_data, dict) else ""
                        
                        # Use citation ratio from analysis
                        citation_ratio = citation_result.get("citation_ratio", 0.0) if citation_result else 0.0
                        
                        citation_record = dbmgr.create_response_citation(
                            response_id=response.id,
                            url=citation_url,
                            citation_ratio=citation_ratio,
                            text_w_citations=citation_text,
                            sentiment=analysis_mgr.analyze_sentiment(citation_text) if citation_text else None
                        )
                        
                        # Detect party mentions in citations using enhanced method
                        if citation_text:
                            mentioned_parties = analysis_mgr.detect_party_mentions(
                                citation_text, 
                                all_parties
                            )
                            
                            for mentioned_party in mentioned_parties:
                                dbmgr.create_party_citation_mention(
                                    citation_id=citation_record.id,
                                    party_id=mentioned_party.id
                                )
                
                # Detect party mentions in main response
                mentioned_parties = analysis_mgr.detect_party_mentions(
                    ai_response["response_text"], 
                    all_parties
                )
                
                for mentioned_party in mentioned_parties:
                    dbmgr.create_party_response_mention(
                        response_id=response.id,
                        party_id=mentioned_party.id
                    )
                
                print(f"    Response logged successfully (ID: {response.id}) [LITE - No UQ]")
                
            except Exception as e:
                print(f"    Error generating response: {str(e)}")
                continue
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"Lite open question response logging completed in {duration:.2f} seconds")

def log_responses_for_all_parties():
    """
    Log responses for all active parties (lite version)
    """
    dbmgr = DBManager()
    parties = dbmgr.get_all_parties()
    
    print(f"Starting lite response logging for {len(parties)} parties")
    
    for party in parties:
        log_response_for_party(party.id)
        time.sleep(1)  # Small delay between parties

def log_responses_for_all_candidates():
    """
    Log responses for all candidates (lite version)
    """
    dbmgr = DBManager()
    candidates = dbmgr.get_all_candidates()
    
    print(f"Starting lite response logging for {len(candidates)} candidates")
    
    for candidate in candidates:
        log_response_for_candidate(candidate.id)
        time.sleep(1)  # Small delay between candidates

def log_all_responses():
    """
    Log responses for all types: parties, candidates, and open questions (lite version)
    """
    print("Starting comprehensive lite response logging for all types")
    total_start_time = time.time()
    
    # Log responses for all parties
    print("\n" + "="*60)
    print("PHASE 1: LOGGING PARTY RESPONSES (LITE)")
    print("="*60)
    log_responses_for_all_parties()
    
    # Log responses for all candidates
    print("\n" + "="*60)
    print("PHASE 2: LOGGING CANDIDATE RESPONSES (LITE)")
    print("="*60)
    log_responses_for_all_candidates()
    
    # Log responses for open questions
    print("\n" + "="*60)
    print("PHASE 3: LOGGING OPEN QUESTION RESPONSES (LITE)")
    print("="*60)
    log_responses_for_open_questions()
    
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    print("\n" + "="*60)
    print("ALL LITE RESPONSE LOGGING COMPLETED")
    print("="*60)
    print(f"Total time: {total_duration:.2f} seconds")

if __name__ == "__main__":
    # Example usage
    print("Election Response Logger - LITE VERSION (No UQ Processing)")
    print("1. Log responses for all parties (lite)")
    print("2. Log responses for all candidates (lite)")
    print("3. Log responses for specific party (lite)")
    print("4. Log responses for specific candidate (lite)")
    print("5. Log responses for open questions (lite)")
    print("6. Log all responses - parties + candidates + open questions (lite)")
    
    choice = input("Select option (1-6): ")
    
    if choice == "1":
        log_responses_for_all_parties()
    elif choice == "2":
        log_responses_for_all_candidates()
    elif choice == "3":
        party_id = int(input("Enter party ID: "))
        log_response_for_party(party_id)
    elif choice == "4":
        candidate_id = int(input("Enter candidate ID: "))
        log_response_for_candidate(candidate_id)
    elif choice == "5":
        log_responses_for_open_questions()
    elif choice == "6":
        log_all_responses()
    else:
        print("Invalid choice")