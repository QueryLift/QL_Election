#!/usr/bin/env python3
"""
Test script to verify SQLAlchemy relationships are working correctly
"""

import os
import sys
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Add the current directory to the path so we can import create_db
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from create_db import Base, Party, Candidate, Model, Prompt, Response, Category, PromptType

def test_relationships():
    """Test that all SQLAlchemy relationships are properly defined"""
    print("Testing SQLAlchemy relationships...")
    
    try:
        load_dotenv()
        
        # Create an in-memory SQLite database for testing
        engine = create_engine("sqlite:///:memory:", echo=False)
        
        # Create all tables
        Base.metadata.create_all(engine)
        print("✓ All tables created successfully")
        
        # Create session
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Test creating basic objects
        party = Party(name="Test Party", url="https://test.com")
        candidate = Candidate(name="Test Candidate")
        model = Model(name="test-model")
        category = Category(name="test-category")
        prompt_type = PromptType(name="test-type")
        
        # Add and commit basic objects first
        session.add_all([party, candidate, model, category, prompt_type])
        session.commit()
        print("✓ Basic objects created and committed")
        
        # Test creating a prompt with relationships
        prompt = Prompt(
            party_id=party.id,
            content="Test prompt content",
            category_id=category.id,
            prompt_type=prompt_type.id,
            is_active=True
        )
        session.add(prompt)
        session.commit()
        print("✓ Prompt created with relationships")
        
        # Test creating a response
        response = Response(
            prompt_id=prompt.id,
            content="Test response content",
            ai_model_id=model.id,
            sentiment=0.5,
            party_mention_rate=0.3
        )
        session.add(response)
        session.commit()
        print("✓ Response created with relationships")
        
        # Test accessing relationships
        print(f"✓ Party name: {prompt.party.name}")
        print(f"✓ Category name: {prompt.category.name}")
        print(f"✓ Response content: {response.prompt.content[:20]}...")
        print(f"✓ Model name: {response.model.name}")
        
        print("\n🎉 ALL RELATIONSHIPS WORKING CORRECTLY!")
        return True
        
    except Exception as e:
        print(f"✗ Relationship test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        if 'session' in locals():
            session.close()

if __name__ == "__main__":
    success = test_relationships()
    if not success:
        sys.exit(1)