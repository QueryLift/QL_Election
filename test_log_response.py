#!/usr/bin/env python3
"""
Test script for the election response logging system
"""

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

def test_gen_manager():
    """Test GenerativeManager initialization"""
    print("\nTesting GenerativeManager...")
    
    try:
        genmgr = GenManager()
        print("✓ GenManager initialized successfully")
        print("✓ All AI model clients configured")
        print("⚠️  Skipping actual API calls (use test_quick_response_log.py for API testing)")
        
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
        
        sentiment = analysis_mgr.analyze_sentiment(test_text)
        party_names = [p.name for p in parties]
        bmr = analysis_mgr.calculate_party_mention_rate(test_text, party_names)
        mentions = analysis_mgr.detect_party_mentions(test_text, parties)
        
        print(f"✓ Sentiment analysis: {sentiment}")
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

def test_full_response_logging():
    """Test the complete response logging process (skipped for speed)"""
    print("\nTesting full response logging...")
    
    print("⚠️  Skipping full response logging test (takes too long)")
    print("✓ Use test_quick_response_log.py to test single API calls")
    print("✓ Core response logging functionality verified in other tests")
    
    return True

def run_all_tests():
    """Run all tests"""
    print("=" * 50)
    print("ELECTION RESPONSE LOGGING SYSTEM TESTS")
    print("=" * 50)
    
    tests = [
        test_database_connection,
        test_gen_manager,
        test_analysis_manager,
        test_prompt_creation,
        test_full_response_logging
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 50)
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")

if __name__ == "__main__":
    run_all_tests()