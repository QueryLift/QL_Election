#!/usr/bin/env python3
"""
Test all available AI models to ensure they work correctly
"""

import os
from log_response import DBManager
from generative_search import GenManager

def test_all_models():
    """Test all available AI models"""
    print("=" * 60)
    print("TESTING ALL AI MODELS")
    print("=" * 60)
    
    # Check available API keys
    api_keys = {
        "OpenAI": os.getenv("OPENAI_API_KEY"),
        "Grok": os.getenv("XAI_API_KEY"), 
        "Gemini": os.getenv("GOOGLE_API_KEY"),
        "Anthropic": os.getenv("ANTHROPIC_API_KEY")
    }
    
    print("API Key Status:")
    for name, key in api_keys.items():
        status = "✓ Configured" if key else "✗ Missing"
        print(f"  {name}: {status}")
    
    if not any(api_keys.values()):
        print("\n⚠️  No API keys configured - cannot test models")
        return False
    
    try:
        # Initialize managers
        dbmgr = DBManager()
        genmgr = GenManager()
        
        # Get available models from database
        models = dbmgr.get_all_models()
        test_prompt = "自由民主党の主要政策は何ですか？"
        
        print(f"\nTesting {len(models)} models with prompt: {test_prompt}")
        print("-" * 60)
        
        successful_models = []
        failed_models = []
        
        for model in models:
            print(f"\nTesting {model.name}...")
            
            try:
                response = genmgr.generate_response(test_prompt, model.name)
                
                print(f"✓ {model.name} - Success")
                print(f"  Response length: {len(response.get('response_text', ''))} chars")
                sources = response.get('source', []) or []
                citations = response.get('citations', []) or []
                print(f"  Sources: {len(sources)} items")
                print(f"  Citations: {len(citations)} items")
                print(f"  Usage: ${response.get('usage', 0):.6f}")
                
                successful_models.append(model.name)
                
            except Exception as e:
                print(f"✗ {model.name} - Failed: {str(e)}")
                failed_models.append(model.name)
        
        print("\n" + "=" * 60)
        print("MODEL TEST SUMMARY")
        print("=" * 60)
        print(f"✓ Successful: {len(successful_models)}/{len(models)} models")
        print(f"✗ Failed: {len(failed_models)}/{len(models)} models")
        
        if successful_models:
            print(f"\nWorking models: {', '.join(successful_models)}")
        
        if failed_models:
            print(f"\nFailed models: {', '.join(failed_models)}")
        
        return len(successful_models) > 0
        
    except Exception as e:
        print(f"✗ Model testing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_all_models()
    if success:
        print("\n🎉 At least one model is working! System is operational.")
    else:
        print("\n⚠️  No models are working. Check API keys and configuration.")