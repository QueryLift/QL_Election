#!/usr/bin/env python3
"""
Complete test script for the election response logging system
Tests both basic functionality and API integration with proper error handling
"""

import os
from log_response import DBManager, AnalysisManager, log_response_for_party
from generative_search import GenManager
from create_db import create_database

def test_database_connection():
    """Test database connection and basic operations"""
    print("Testing database connection...")
    
    try:
        dbmgr = DBManager()
        parties = dbmgr.get_all_parties()
        candidates = dbmgr.get_all_candidates()
        models = dbmgr.get_all_models()
        
        print(f"✓ Found {len(parties)} parties")
        print(f"✓ Found {len(candidates)} candidates")
        print(f"✓ Found {len(models)} models")
        
        return True
    except Exception as e:
        print(f"✗ Database connection failed: {str(e)}")
        return False

def test_gen_manager_initialization():
    """Test GenerativeManager initialization"""
    print("\nTesting GenerativeManager initialization...")
    
    try:
        genmgr = GenManager()
        print("✓ GenManager initialized successfully")
        
        # Check if API keys are configured
        api_keys = {
            "OpenAI": os.getenv("OPENAI_KEY"),
            "Grok": os.getenv("GROK_API_KEY"),
            "Gemini": os.getenv("GEMINI_API_KEY"),
            "Anthropic": os.getenv("ANTHROPIC_API_KEY")
        }
        
        configured_apis = [name for name, key in api_keys.items() if key]
        print(f"✓ Configured APIs: {', '.join(configured_apis) if configured_apis else 'None'}")
        
        if not configured_apis:
            print("⚠️  No API keys configured - API tests will be skipped")
        
        return True
    except Exception as e:
        print(f"✗ GenerativeManager initialization failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_gen_manager_api_call():
    """Test GenerativeManager API call (if keys available)"""
    print("\nTesting GenerativeManager API call...")
    
    # Check if any API key is available
    if not (os.getenv("OPENAI_KEY") or os.getenv("GROK_API_KEY") or 
            os.getenv("GEMINI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
        print("⚠️  No API keys configured - skipping API test")
        return True
    
    try:
        genmgr = GenManager()
        test_prompt = "自由民主党は、物価高への対策として消費税率をどう扱うと公約していますか？"
        
        # Try different models based on available keys
        models_to_try = []
        if os.getenv("OPENAI_KEY"):
            models_to_try.append("gpt-4o-search-preview")
        if os.getenv("GROK_API_KEY"):
            models_to_try.append("grok-2-latest")
        if os.getenv("GEMINI_API_KEY"):
            models_to_try.append("gemini-2.0-flash")
        if os.getenv("ANTHROPIC_API_KEY"):
            models_to_try.append("claude-3-7-sonnet-20240620")
        
        success = False
        for model in models_to_try:
            try:
                print(f"  Trying {model}...")
                response = genmgr.generate_response(test_prompt, model)
                
                print(f"✓ Generated response with {model}")
                print(f"  - Response length: {len(response.get('response_text', ''))} characters")
                print(f"  - Sources: {len(response.get('source', []))} items")
                print(f"  - Citations: {len(response.get('citations', []))} items")
                print(f"  - Usage cost: ${response.get('usage', 0):.6f}")
                
                success = True
                break
            except Exception as model_error:
                print(f"  ✗ {model} failed: {str(model_error)}")
                continue
        
        if success:
            return True
        else:
            print("✗ All configured models failed")
            return False
            
    except Exception as e:
        print(f"✗ GenerativeManager API test failed: {str(e)}")
        return False

def test_analysis_manager():
    """Test AnalysisManager functionality"""
    print("\nTesting AnalysisManager...")
    
    try:
        analysis_mgr = AnalysisManager()
        dbmgr = DBManager()
        parties = dbmgr.get_all_parties()
        
        test_text = "自由民主党と立憲民主党の政策について議論します。公明党も重要な役割を果たす。"
        
        # Test sentiment analysis (may fail if Google Cloud credentials not configured)
        try:
            sentiment = analysis_mgr.analyze_sentiment(test_text)
            print(f"✓ Sentiment analysis: {sentiment}")
        except Exception as sentiment_error:
            print(f"⚠️  Sentiment analysis failed (Google Cloud not configured): {str(sentiment_error)}")
            sentiment = 0.0
        
        # Test party mention rate calculation
        party_names = [p.name for p in parties]
        bmr = analysis_mgr.calculate_party_mention_rate(test_text, party_names)
        mentions = analysis_mgr.detect_party_mentions(test_text, parties)
        
        print(f"✓ Party mention rate: {bmr}")
        print(f"✓ Detected party mentions: {len(mentions)} parties")
        for party in mentions:
            print(f"  - {party.name}")
        
        return True
    except Exception as e:
        print(f"✗ AnalysisManager test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_prompt_creation():
    """Test prompt creation functionality"""
    print("\nTesting prompt creation...")
    
    try:
        dbmgr = DBManager()
        parties = dbmgr.get_all_parties()
        
        if parties:
            party = parties[0]
            prompts = dbmgr.create_party_prompts(party.id)
            print(f"✓ Created {len(prompts)} prompts for party: {party.name}")
            
            # Test candidate prompts
            candidates = dbmgr.get_all_candidates()
            if candidates:
                candidate = candidates[0]
                candidate_prompts = dbmgr.create_candidate_prompts(candidate.id)
                print(f"✓ Created {len(candidate_prompts)} prompts for candidate: {candidate.name}")
        
        return True
    except Exception as e:
        print(f"✗ Prompt creation test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_full_response_logging():
    """Test the complete response logging process (if API keys available)"""
    print("\nTesting full response logging...")
    
    # Check if any API key is available
    if not (os.getenv("OPENAI_KEY") or os.getenv("GROK_API_KEY") or 
            os.getenv("GEMINI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
        print("⚠️  No API keys configured - skipping full response logging test")
        return True
    
    try:
        dbmgr = DBManager()
        parties = dbmgr.get_all_parties()
        
        if parties:
            party = parties[0]  # Test with first party
            print(f"Testing response logging for party: {party.name}")
            
            # This will make actual API calls, so we limit it to one party
            log_response_for_party(party.id)
            print("✓ Response logging completed successfully")
        
        return True
    except Exception as e:
        print(f"✗ Full response logging test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def run_complete_tests():
    """Run all tests with proper error handling"""
    print("=" * 60)
    print("COMPLETE ELECTION RESPONSE LOGGING SYSTEM TESTS")
    print("=" * 60)
    
    tests = [
        test_database_connection,
        test_gen_manager_initialization,
        test_analysis_manager,
        test_prompt_creation,
        test_gen_manager_api_call,
        # test_full_response_logging  # Commented out to avoid long API calls during testing
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()  # Add spacing between tests
    
    print("=" * 60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        print("\nTo test full response logging, uncomment the test_full_response_logging line")
        print("and ensure API keys are configured in your .env file.")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
    
    return passed == total

if __name__ == "__main__":
    run_complete_tests()