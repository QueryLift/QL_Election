#!/usr/bin/env python3
"""
Debug script to examine Gemini response structure
"""

import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

def debug_gemini_response():
    """Examine the structure of a Gemini response to understand citations/sources"""
    
    # Configure Gemini
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    
    # Create model
    model = genai.GenerativeModel("gemini-2.0-flash-exp")
    
    # Test prompt
    prompt = "自由民主党の主要政策について教えてください。信頼できるソースからの情報を含めてください。"
    
    print("=" * 60)
    print("GEMINI RESPONSE STRUCTURE DEBUG")
    print("=" * 60)
    print(f"Prompt: {prompt}")
    print("-" * 60)
    
    try:
        response = model.generate_content(prompt)
        
        print("Response attributes:")
        print(f"  - dir(response): {[attr for attr in dir(response) if not attr.startswith('_')]}")
        
        print(f"\nResponse text length: {len(response.text)} characters")
        print(f"Response text preview: {response.text[:200]}...")
        
        print(f"\nCandidates: {hasattr(response, 'candidates')}")
        if hasattr(response, 'candidates') and response.candidates:
            print(f"Number of candidates: {len(response.candidates)}")
            candidate = response.candidates[0]
            
            print(f"\nCandidate attributes:")
            print(f"  - dir(candidate): {[attr for attr in dir(candidate) if not attr.startswith('_')]}")
            
            # Check for grounding metadata
            print(f"\nGrounding metadata: {hasattr(candidate, 'grounding_metadata')}")
            if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                gm = candidate.grounding_metadata
                print(f"  - dir(grounding_metadata): {[attr for attr in dir(gm) if not attr.startswith('_')]}")
                
                # Check for grounding chunks
                print(f"  - grounding_chunks: {hasattr(gm, 'grounding_chunks')}")
                if hasattr(gm, 'grounding_chunks') and gm.grounding_chunks:
                    print(f"    Number of chunks: {len(gm.grounding_chunks)}")
                    for i, chunk in enumerate(gm.grounding_chunks):
                        print(f"    Chunk {i} attributes: {[attr for attr in dir(chunk) if not attr.startswith('_')]}")
                        if hasattr(chunk, 'web') and chunk.web:
                            print(f"      Web URI: {chunk.web.uri}")
                
                # Check for web search queries
                print(f"  - web_search_queries: {hasattr(gm, 'web_search_queries')}")
                if hasattr(gm, 'web_search_queries') and gm.web_search_queries:
                    print(f"    Queries: {list(gm.web_search_queries)}")
                
                # Check for grounding supports
                print(f"  - grounding_supports: {hasattr(gm, 'grounding_supports')}")
                if hasattr(gm, 'grounding_supports') and gm.grounding_supports:
                    print(f"    Number of supports: {len(gm.grounding_supports)}")
                    for i, support in enumerate(gm.grounding_supports):
                        print(f"    Support {i} attributes: {[attr for attr in dir(support) if not attr.startswith('_')]}")
        
        # Check usage metadata
        print(f"\nUsage metadata: {hasattr(response, 'usage_metadata')}")
        if hasattr(response, 'usage_metadata'):
            usage = response.usage_metadata
            print(f"  - dir(usage_metadata): {[attr for attr in dir(usage) if not attr.startswith('_')]}")
            print(f"  - prompt_token_count: {getattr(usage, 'prompt_token_count', 'N/A')}")
            print(f"  - candidates_token_count: {getattr(usage, 'candidates_token_count', 'N/A')}")
            print(f"  - total_token_count: {getattr(usage, 'total_token_count', 'N/A')}")
        
        print("\n" + "=" * 60)
        print("DEBUG COMPLETE")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_gemini_response()