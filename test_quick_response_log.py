#!/usr/bin/env python3
"""
Quick test for full response logging with just one prompt
"""

import os
from log_response import DBManager, AnalysisManager
from generative_search import GenManager

def test_single_response_log():
    """Test logging a single response"""
    print("Testing single response logging...")
    
    # Check if any API key is available
    if not (os.getenv("OPENAI_KEY") or os.getenv("GROK_API_KEY") or 
            os.getenv("GEMINI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
        print("⚠️  No API keys configured - cannot test response logging")
        return False
    
    try:
        # Initialize managers
        dbmgr = DBManager()
        genmgr = GenManager()
        analysis_mgr = AnalysisManager()
        
        # Get first party and create prompts
        parties = dbmgr.get_all_parties()
        if not parties:
            print("✗ No parties found in database")
            return False
            
        party = parties[0]
        prompts = dbmgr.create_party_prompts(party.id)
        
        if not prompts:
            print("✗ No prompts created")
            return False
            
        # Get first model
        models = dbmgr.get_all_models()
        if not models:
            print("✗ No models found")
            return False
            
        model = models[1]
        prompt = prompts[0]
        
        print(f"Testing with party: {party.name}")
        print(f"Model: {model.name}")
        print(f"Prompt: {prompt.content[:50]}...")
        
        # Replace placeholder and generate response
        prompt_content = prompt.content.replace("【政党名】", party.name)
        
        # Get all parties for mention detection
        all_parties = dbmgr.get_all_parties()
        party_names = [p.name for p in all_parties]
        
        # Generate AI response
        ai_response = genmgr.generate_response(prompt_content, model.name)
        
        print(f"✓ Generated response: {len(ai_response['response_text'])} characters")
        
        # Analyze sentiment
        sentiment = analysis_mgr.analyze_sentiment(ai_response["response_text"])
        
        # Calculate party mention rate
        bmr = analysis_mgr.calculate_party_mention_rate(
            ai_response["response_text"], 
            party_names
        )
        
        print(f"✓ Sentiment: {sentiment}")
        print(f"✓ Party mention rate: {bmr}")
        
        # Create response record
        response = dbmgr.create_response(
            prompt_id=prompt.id,
            content=ai_response["response_text"],
            model_id=model.id,
            usage=ai_response.get("usage"),
            search_query=ai_response.get("search_query"),
            sentiment=sentiment,
            party_mention_rate=bmr
        )
        
        print(f"✓ Created response record (ID: {response.id})")
        
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
            print(f"✓ Processed {len(sources)} sources")
        
        # Process citations
        citations = ai_response.get("citations", [])
        if citations:
            for citation_data in citations:
                citation_url = citation_data.get("url") if isinstance(citation_data, dict) else citation_data
                citation_text = citation_data.get("text", "") if isinstance(citation_data, dict) else ""
                
                citation_record = dbmgr.create_response_citation(
                    response_id=response.id,
                    url=citation_url,
                    citation_ratio=0.8,  # Mock ratio
                    text_w_citations=citation_text,
                    sentiment=analysis_mgr.analyze_sentiment(citation_text) if citation_text else None
                )
            print(f"✓ Processed {len(citations)} citations")
        
        print("✓ Single response logging test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Single response logging test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("QUICK RESPONSE LOGGING TEST")
    print("=" * 50)
    
    success = test_single_response_log()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Quick test passed! Full response logging is working.")
    else:
        print("⚠️  Quick test failed.")
    print("=" * 50)