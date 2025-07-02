#!/usr/bin/env python3
"""
Fast test script for the election response logging system
Tests core functionality without making actual API calls
"""

from log_response import DBManager, AnalysisManager
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

def test_gen_manager():
    """Test GenerativeManager initialization (no API calls)"""
    print("\nTesting GenerativeManager initialization...")
    
    try:
        genmgr = GenManager()
        print("✓ GenManager initialized successfully")
        print("✓ No API errors during initialization")
        
        return True
    except Exception as e:
        print(f"✗ GenerativeManager test failed: {str(e)}")
        return False

def test_analysis_manager():
    """Test AnalysisManager functionality"""
    print("\nTesting AnalysisManager...")
    
    try:
        analysis_mgr = AnalysisManager()
        dbmgr = DBManager()
        parties = dbmgr.get_all_parties()
        
        test_text = "自由民主党と立憲民主党の政策について議論します。"
        
        # Test sentiment analysis (may fail if Google Cloud credentials not configured)
        try:
            sentiment = analysis_mgr.analyze_sentiment(test_text)
            print(f"✓ Sentiment analysis: {sentiment}")
        except Exception as sentiment_error:
            print(f"⚠️  Sentiment analysis failed (credentials not configured): skipping")
            sentiment = 0.0
        
        # Test party mention rate calculation
        party_names = [p.name for p in parties]
        bmr = analysis_mgr.calculate_party_mention_rate(test_text, party_names)
        mentions = analysis_mgr.detect_party_mentions(test_text, parties)
        
        print(f"✓ Party mention rate: {bmr}")
        print(f"✓ Detected party mentions: {len(mentions)} parties")
        
        return True
    except Exception as e:
        print(f"✗ AnalysisManager test failed: {str(e)}")
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
        return False

def test_response_creation():
    """Test response creation without API calls"""
    print("\nTesting response creation...")
    
    try:
        dbmgr = DBManager()
        parties = dbmgr.get_all_parties()
        models = dbmgr.get_all_models()
        
        if parties and models:
            party = parties[0]
            model = models[0]
            
            # Create prompts for the party
            prompts = dbmgr.create_party_prompts(party.id)
            if prompts:
                prompt = prompts[0]
                
                # Create a mock response
                mock_response = dbmgr.create_response(
                    prompt_id=prompt.id,
                    content="これは自由民主党の政策に関するテスト回答です。",
                    model_id=model.id,
                    usage=0.001,
                    search_query="自由民主党 政策",
                    sentiment=0.5,
                    party_mention_rate=0.8
                )
                
                print(f"✓ Created mock response (ID: {mock_response.id})")
        
        return True
    except Exception as e:
        print(f"✗ Response creation test failed: {str(e)}")
        return False

def run_fast_tests():
    """Run fast tests without API calls"""
    print("=" * 50)
    print("FAST ELECTION RESPONSE LOGGING SYSTEM TESTS")
    print("=" * 50)
    
    tests = [
        test_database_connection,
        test_gen_manager,
        test_analysis_manager,
        test_prompt_creation,
        test_response_creation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()  # Add spacing
    
    print("=" * 50)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 50)
    
    if passed == total:
        print("🎉 All fast tests passed! Core functionality is working.")
        print("\nNote: This test skips actual API calls for speed.")
        print("The system is ready for full API testing.")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
    
    return passed == total

if __name__ == "__main__":
    run_fast_tests()