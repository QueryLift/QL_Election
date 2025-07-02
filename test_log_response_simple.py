#!/usr/bin/env python3
"""
Simplified test script for the election response logging system
Tests basic functionality without requiring API calls
"""

from log_response import DBManager, AnalysisManager
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

def test_analysis_manager_basic():
    """Test AnalysisManager basic functionality without API calls"""
    print("\nTesting AnalysisManager basic functionality...")
    
    try:
        analysis_mgr = AnalysisManager()
        dbmgr = DBManager()
        parties = dbmgr.get_all_parties()
        
        test_text = "自由民主党と立憲民主党の政策について議論します。"
        
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

def test_mock_response_creation():
    """Test response creation with mock data"""
    print("\nTesting mock response creation...")
    
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
                print(f"✓ Response content: {mock_response.content[:50]}...")
                
                # Test creating source and citation
                source = dbmgr.create_response_source(
                    response_id=mock_response.id,
                    url="https://example.com/policy",
                    is_cited=True
                )
                print(f"✓ Created response source (ID: {source.id})")
                
                citation = dbmgr.create_response_citation(
                    response_id=mock_response.id,
                    url="https://example.com/citation",
                    citation_ratio=0.9,
                    text_w_citations="自由民主党の政策詳細",
                    sentiment=0.6
                )
                print(f"✓ Created response citation (ID: {citation.id})")
        
        return True
    except Exception as e:
        print(f"✗ Mock response creation test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def run_simplified_tests():
    """Run simplified tests without API dependencies"""
    print("=" * 50)
    print("SIMPLIFIED ELECTION SYSTEM TESTS")
    print("=" * 50)
    
    tests = [
        test_database_connection,
        test_analysis_manager_basic,
        test_prompt_creation,
        test_mock_response_creation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()  # Add spacing between tests
    
    print("=" * 50)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 50)
    
    if passed == total:
        print("🎉 All basic tests passed! Core functionality is working.")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
    
    return passed == total

if __name__ == "__main__":
    run_simplified_tests()