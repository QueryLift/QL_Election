#!/usr/bin/env python3
"""
Initialize prompts for all parties and candidates in the database
"""

from log_response import DBManager

def initialize_all_prompts():
    """Initialize prompts for all parties and candidates"""
    print("Initializing prompts for all entities...")
    
    dbmgr = DBManager()
    
    # Initialize prompts for all parties
    parties = dbmgr.get_all_parties()
    print(f"Found {len(parties)} parties")
    
    for party in parties:
        print(f"Creating prompts for party: {party.name}")
        prompts = dbmgr.create_party_prompts(party.id)
        print(f"  Created {len(prompts)} prompts")
    
    # Initialize prompts for all candidates
    candidates = dbmgr.get_all_candidates()
    print(f"Found {len(candidates)} candidates")
    
    for candidate in candidates:
        print(f"Creating prompts for candidate: {candidate.name}")
        prompts = dbmgr.create_candidate_prompts(candidate.id)
        print(f"  Created {len(prompts)} prompts")
    
    print("Prompt initialization completed!")

if __name__ == "__main__":
    initialize_all_prompts()