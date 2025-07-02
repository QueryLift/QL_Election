#!/usr/bin/env python3
"""
Test database operations to verify cascade configuration fixes
"""

import os
import sys
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from create_db import Base, Party, Candidate, Model, Prompt, Response, ResponseSource, ResponseCitation, PartyResponseMention, PartyCitationMention, Category, PromptType

load_dotenv()

def test_database_operations():
    """Test basic database operations including cascade behavior"""
    print("=" * 60)
    print("TESTING DATABASE OPERATIONS")
    print("=" * 60)
    
    try:
        # Create test engine (using SQLite for testing)
        test_db_url = "sqlite:///test_election.db"
        engine = create_engine(test_db_url, echo=False)
        
        # Create all tables
        Base.metadata.create_all(engine)
        print("✓ Database tables created successfully")
        
        # Create session
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Test 1: Create basic entities
        print("\nTest 1: Creating basic entities...")
        
        # Create party
        party = Party(name="テスト政党", url="https://test-party.jp/")
        session.add(party)
        session.commit()
        print("✓ Party created")
        
        # Create candidate
        candidate = Candidate(name="テスト候補者")
        session.add(candidate)
        session.commit()
        print("✓ Candidate created")
        
        # Create model
        model = Model(name="test-model")
        session.add(model)
        session.commit()
        print("✓ Model created")
        
        # Create category
        category = Category(name="test-category")
        session.add(category)
        session.commit()
        print("✓ Category created")
        
        # Create prompt type
        prompt_type = PromptType(name="test-prompt-type")
        session.add(prompt_type)
        session.commit()
        print("✓ PromptType created")
        
        # Test 2: Create prompt with relationships
        print("\nTest 2: Creating prompt with relationships...")
        
        prompt = Prompt(
            party_id=party.id,
            candidate_id=candidate.id,
            content="テストプロンプト",
            category_id=category.id,
            prompt_type=prompt_type.id,
            is_active=True
        )
        session.add(prompt)
        session.commit()
        print("✓ Prompt created with relationships")
        
        # Test 3: Create response with cascade relationships
        print("\nTest 3: Creating response with cascade relationships...")
        
        response = Response(
            prompt_id=prompt.id,
            content="テストレスポンス",
            ai_model_id=model.id,
            usage=0.001,
            search_query="テストクエリ",
            sentiment=0.5,
            party_mention_rate=0.3,
            semantic_negentropy=0.8,
            noncontradiction=0.9,
            exact_match=0.7,
            cosine_sim=0.85,
            bert_score=0.9,
            bleurt=0.88
        )
        session.add(response)
        session.commit()
        print("✓ Response created")
        
        # Test 4: Create sources and citations
        print("\nTest 4: Creating sources and citations...")
        
        # Create response source
        source = ResponseSource(
            response_id=response.id,
            url="https://test-source.com",
            is_cited=True
        )
        session.add(source)
        session.commit()
        print("✓ Response source created")
        
        # Create response citation
        citation = ResponseCitation(
            response_id=response.id,
            url="https://test-citation.com",
            citation_ratio=0.75,
            text_w_citations="テスト引用文",
            sentiment=0.6
        )
        session.add(citation)
        session.commit()
        print("✓ Response citation created")
        
        # Test 5: Create party mentions
        print("\nTest 5: Creating party mentions...")
        
        # Create party response mention
        party_response_mention = PartyResponseMention(
            response_id=response.id,
            party_id=party.id
        )
        session.add(party_response_mention)
        session.commit()
        print("✓ Party response mention created")
        
        # Create party citation mention
        party_citation_mention = PartyCitationMention(
            citation_id=citation.id,
            party_id=party.id
        )
        session.add(party_citation_mention)
        session.commit()
        print("✓ Party citation mention created")
        
        # Test 6: Test cascade deletion behavior
        print("\nTest 6: Testing cascade deletion behavior...")
        
        # Get counts before deletion
        initial_response_count = session.query(Response).count()
        initial_source_count = session.query(ResponseSource).count()
        initial_citation_count = session.query(ResponseCitation).count()
        initial_party_response_mention_count = session.query(PartyResponseMention).count()
        initial_party_citation_mention_count = session.query(PartyCitationMention).count()
        
        print(f"Before deletion - Responses: {initial_response_count}, Sources: {initial_source_count}, Citations: {initial_citation_count}")
        print(f"Before deletion - Party Response Mentions: {initial_party_response_mention_count}, Party Citation Mentions: {initial_party_citation_mention_count}")
        
        # Delete prompt (should cascade to response and its related records)
        session.delete(prompt)
        session.commit()
        
        # Check counts after deletion
        final_response_count = session.query(Response).count()
        final_source_count = session.query(ResponseSource).count()
        final_citation_count = session.query(ResponseCitation).count()
        final_party_response_mention_count = session.query(PartyResponseMention).count()
        final_party_citation_mention_count = session.query(PartyCitationMention).count()
        
        print(f"After deletion - Responses: {final_response_count}, Sources: {final_source_count}, Citations: {final_citation_count}")
        print(f"After deletion - Party Response Mentions: {final_party_response_mention_count}, Party Citation Mentions: {final_party_citation_mention_count}")
        
        # Verify cascade worked correctly
        if (final_response_count == 0 and final_source_count == 0 and 
            final_citation_count == 0 and final_party_response_mention_count == 0 and 
            final_party_citation_mention_count == 0):
            print("✓ Cascade deletion working correctly")
        else:
            print("✗ Cascade deletion not working as expected")
            return False
        
        session.close()
        
        # Clean up test database
        os.remove("test_election.db")
        print("✓ Test database cleaned up")
        
        print("\n" + "=" * 60)
        print("DATABASE TEST SUMMARY")
        print("=" * 60)
        print("✓ All database operations completed successfully")
        print("✓ Cascade configuration is working correctly")
        print("✓ No SQLAlchemy relationship errors detected")
        
        return True
        
    except Exception as e:
        print(f"✗ Database test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Clean up test database if it exists
        try:
            if os.path.exists("test_election.db"):
                os.remove("test_election.db")
        except:
            pass
            
        return False

if __name__ == "__main__":
    success = test_database_operations()
    if success:
        print("\n🎉 Database is working correctly!")
    else:
        print("\n⚠️  Database has issues that need to be fixed.")
        sys.exit(1)