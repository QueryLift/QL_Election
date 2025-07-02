from sqlalchemy import Column, Integer, String, ForeignKey, Boolean, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql.sqltypes import TIMESTAMP
from sqlalchemy.sql.expression import func

Base = declarative_base()

class Model(Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)

    responses = relationship("Response", back_populates="model")

class PromptType(Base):
    __tablename__ = "prompt_types"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)

class Product(Base):
    __tablename__ = "products"

    id        = Column(Integer, primary_key=True)
    url  = Column(String, nullable=False)
    name = Column(String, nullable=False)
    description = Column(String)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(
        TIMESTAMP, nullable=False,
        server_default=func.now()
    )
    updated_at = Column(
        TIMESTAMP, nullable=False,
        server_default=func.now(),
        onupdate=func.now()
    )
    prompts = relationship("Prompt", back_populates="product")
    related_url = relationship("RelatedUrl", back_populates="product")

class RelatedUrl(Base):
    __tablename__ = "related_urls"
    id = Column(Integer, primary_key=True)
    url = Column(String, nullable=False)
    created_at = Column(
        TIMESTAMP, nullable=False,
        server_default=func.now()
    )
    product_id = Column(Integer, ForeignKey("products.id", ondelete="CASCADE"))
    product = relationship("Product", back_populates="related_url")


class Prompt(Base):
    __tablename__ = "prompts"

    id        = Column(Integer, primary_key=True)
    product_id = Column(
        Integer,
        ForeignKey("products.id", ondelete="CASCADE"),
        nullable=False
    )
    content    = Column(String, nullable=False)
    created_at = Column(
        TIMESTAMP, nullable=False,
        server_default=func.now()
    )
    category_id = Column(Integer, ForeignKey("categories.id", ondelete="CASCADE"))
    prompt_type = Column(Integer, ForeignKey("prompt_types.id"))
    is_active = Column(Boolean, nullable=False, default=True)

    product   = relationship("Product", back_populates="prompts")
    response  = relationship(
        "Response", back_populates="prompt",
        uselist=False, cascade="all, delete-orphan"
    )
    categories = relationship("Category", back_populates="prompts")

class Response(Base):
    __tablename__ = "responses"

    id         = Column(Integer, primary_key=True)
    prompt_id  = Column(
        Integer,
        ForeignKey("prompts.id", ondelete="CASCADE"),
        nullable=False
    )
    content     = Column(String, nullable=False)
    created_at = Column(
        TIMESTAMP, nullable=False,
        server_default=func.now()
    )
    ai_model_id = Column(Integer, ForeignKey("models.id"))
    usage = Column(Float, nullable=True)
    search_query = Column(String, nullable=True)
    sentntiment = Column(Float, nullable=True)
    brand_mention_rate = Column(Float, nullable=True)

    prompt = relationship("Prompt", back_populates="response")
    model  = relationship("Model", back_populates="responses")

    sources   = relationship(
        "ResponseSource", back_populates="response",
        cascade="all, delete-orphan"
    )
    citations = relationship(
        "ResponseCitation", back_populates="response",
        cascade="all, delete-orphan"
    )
    product_mentions = relationship(
        "ProductResponseMention", back_populates="response",
        cascade="all, delete-orphan"
    )
    

class ResponseSource(Base):
    __tablename__ = "response_sources"

    id          = Column(Integer, primary_key=True)
    response_id = Column(
        Integer,
        ForeignKey("responses.id", ondelete="CASCADE"),
        nullable=False
    )
    url = Column(String, nullable=False)
    is_cited = Column(Boolean, nullable=False, default=False)

    response = relationship("Response", back_populates="sources")

class ResponseCitation(Base):
    __tablename__ = "response_citations"

    id          = Column(Integer, primary_key=True)
    response_id = Column(
        Integer,
        ForeignKey("responses.id", ondelete="CASCADE"),
        nullable=False
    )
    url = Column(String, nullable=False)
    citaition_ratio = Column(Float, nullable=True)
    text_w_citations = Column(String, nullable=True)
    sentiment = Column(Float, nullable=True)
    response = relationship("Response", back_populates="citations")
    product_mentions = relationship(
        "ProductCitationMention", back_populates="citation",
        cascade="all, delete-orphan"
    )
    

class ProductCitationMention(Base):
    __tablename__ = "product_citation_mentions"

    id          = Column(Integer, primary_key=True)
    citation_id = Column(
        Integer,
        ForeignKey("response_citations.id", ondelete="CASCADE"),
        nullable=False
    )
    product_id = Column(
        Integer,
        ForeignKey("products.id", ondelete="CASCADE"),
        nullable=False
    )
    
    citation = relationship("ResponseCitation", back_populates="product_mentions")
    product = relationship("Product")


class ProductResponseMention(Base):
    __tablename__ = "product_response_mentions"

    id          = Column(Integer, primary_key=True)
    response_id = Column(
        Integer,
        ForeignKey("responses.id", ondelete="CASCADE"),
        nullable=False
    )
    product_id = Column(
        Integer,
        ForeignKey("products.id", ondelete="CASCADE"),
        nullable=False
    )
    
    response = relationship("Response", back_populates="product_mentions")
    product = relationship("Product")

class Category(Base):
    __tablename__ = "categories"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    prompts = relationship("Prompt", back_populates="category")

def create_database():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from dotenv import load_dotenv
    import os
    
    load_dotenv()
    
    # Create engine - replace with your database URL
    engine = create_engine("sqlite:///mock_database.db")  # Using SQLite for mock
    
    # Create all tables
    Base.metadata.create_all(engine)
    
    # Create session factory
    Session = sessionmaker(bind=engine)
    session = Session()
    
    print("Mock database tables created successfully!")
    
    return session

if __name__ == "__main__":
    session = create_database()