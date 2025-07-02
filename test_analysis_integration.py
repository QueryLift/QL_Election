#!/usr/bin/env python3
"""
Test script for the enhanced analysis integration in the election response logging system
"""

from log_response import DBManager, AnalysisManager
from generative_search import GenManager

def test_analysis_integration():
    """Test the analysis functions integration"""
    print("=" * 60)
    print("TESTING ENHANCED ANALYSIS INTEGRATION")
    print("=" * 60)
    
    # Initialize managers
    try:
        dbmgr = DBManager()
        analysis_mgr = AnalysisManager()
        genmgr = GenManager()
        print("✓ All managers initialized successfully")
    except Exception as e:
        print(f"✗ Manager initialization failed: {e}")
        return False
    
    # Test sentiment analysis
    print("\n1. Testing Sentiment Analysis...")
    test_text = "自由民主党の経済政策は非常に優れており、日本の将来に希望をもたらします。"
    try:
        sentiment = analysis_mgr.analyze_sentiment(test_text)
        print(f"✓ Sentiment analysis result: {sentiment}")
    except Exception as e:
        print(f"✗ Sentiment analysis failed: {e}")
    
    # Test brand mention rate calculation
    print("\n2. Testing Brand Mention Rate...")
    try:
        parties = dbmgr.get_all_parties()
        party_names = [p.name for p in parties]
        test_text = "自由民主党と立憲民主党の政策について議論する。公明党も重要な役割を果たす。"
        pmr = analysis_mgr.calculate_party_mention_rate(test_text, party_names)
        print(f"✓ Party mention rate: {pmr}")
        print(f"✓ Party names tested: {party_names[:3]}...")
    except Exception as e:
        print(f"✗ PMR calculation failed: {e}")
    
    # Test party mention detection
    print("\n3. Testing Party Mention Detection...")
    try:
        mentioned_parties = analysis_mgr.detect_party_mentions(test_text, parties)
        print(f"✓ Detected {len(mentioned_parties)} party mentions:")
        for party in mentioned_parties:
            print(f"  - {party.name}")
    except Exception as e:
        print(f"✗ Party mention detection failed: {e}")
    
    # Test citation processing with analysis
    print("\n4. Testing Citation Analysis...")
    try:
        mock_citations = [
            {
                "url": "https://example.com/policy1",
                "text": "自由民主党は消費税減税を検討している",
                "start_index": 0,
                "end_index": 20
            },
            {
                "url": "https://example.com/policy2", 
                "text": "立憲民主党は社会保障制度の充実を目指す",
                "start_index": 21,
                "end_index": 40
            }
        ]
        
        mock_response_text = "自由民主党は消費税減税を検討している。立憲民主党は社会保障制度の充実を目指す。"
        citation_results = analysis_mgr.process_citations_with_analysis(
            mock_citations, 
            mock_response_text, 
            parties
        )
        
        print(f"✓ Processed {len(citation_results)} citations:")
        for result in citation_results:
            print(f"  - URL: {result.get('url', 'N/A')}")
            print(f"    Citation ratio: {result.get('citation_ratio', 0)}")
            print(f"    Mentioned parties: {len(result.get('mentioned_parties', []))}")
            
    except Exception as e:
        print(f"✗ Citation analysis failed: {e}")
    
    # Test full AI response generation with analysis
    print("\n5. Testing Full AI Response with Analysis...")
    try:
        test_prompt = "自由民主党は、物価高への対策として消費税率をどう扱うと公約していますか？"
        ai_response = genmgr.generate_response(test_prompt, "gpt-4o-search-preview")
        
        print(f"✓ AI response generated:")
        print(f"  - Response length: {len(ai_response.get('response_text', ''))} characters")
        print(f"  - Sources: {len(ai_response.get('source', []))} items")
        print(f"  - Citations: {len(ai_response.get('citations', []))} items")
        print(f"  - Usage cost: ${ai_response.get('usage', 0):.6f}")
        
        # Analyze the response
        sentiment = analysis_mgr.analyze_sentiment(ai_response["response_text"])
        pmr = analysis_mgr.calculate_party_mention_rate(ai_response["response_text"], party_names)
        
        print(f"✓ Analysis results:")
        print(f"  - Sentiment score: {sentiment}")
        print(f"  - Party mention rate: {pmr}")
        
    except Exception as e:
        print(f"✗ Full AI response test failed: {e}")
    
    print("\n" + "=" * 60)
    print("ENHANCED ANALYSIS INTEGRATION TEST COMPLETED")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    test_analysis_integration()