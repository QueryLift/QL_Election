# Enhanced Analysis Features Integration

This document outlines the advanced analysis capabilities now integrated into the election response logging system.

## 🔬 **Analysis Functions from analysis.py**

### **1. Sentiment Analysis**
- **Function**: `sentiment_analysis(text)`
- **Technology**: Google Cloud Natural Language API
- **Purpose**: Analyzes emotional tone of AI responses and citations
- **Output**: Float score (-1.0 to 1.0, where -1 = very negative, 0 = neutral, 1 = very positive)
- **Usage**: Applied to both main responses and individual citations

### **2. Brand Mention Rate (BMR)**
- **Function**: `BMR(product_name, competitors, text)`
- **Purpose**: Calculates rate of political party mentions in responses
- **Output**: Float ratio of mentioned parties to total parties
- **Usage**: Tracks which parties are being discussed across different AI models

### **3. Citation Analysis**
- **Function**: `citation_rate(text, citation_list)`
- **Purpose**: Advanced citation processing with overlap detection
- **Features**:
  - Citation ratio calculation
  - Text overlap analysis
  - URL deduplication
  - Citation quality scoring
- **Output**: Array of citation objects with ratios and extracted text

### **4. Citation Mention Detection**
- **Function**: `check_citation_mentions(text, product_name, competitor_names)`
- **Purpose**: Identifies which parties are mentioned within citations
- **Output**: Array of mentioned party objects with type and name

## 🏗️ **Integration Architecture**

### **Enhanced AnalysisManager Class**
```python
class AnalysisManager:
    def analyze_sentiment(text)                    # Google Cloud sentiment
    def calculate_brand_mention_rate(text, parties) # Party mention counting
    def detect_party_mentions(text, parties)       # Party detection
    def process_citations_with_analysis(citations) # Advanced citation processing
```

### **Database Enhancements**
- **Sentiment scores** stored for responses and citations
- **Citation ratios** calculated and stored
- **Party mentions** tracked in both responses and citations
- **Enhanced metadata** for all analysis metrics

## 📊 **Analysis Scores Captured**

### **Response-Level Metrics**
1. **Sentiment Score**: Overall emotional tone
2. **Brand Mention Rate**: Party mention frequency
3. **Usage Cost**: API consumption tracking
4. **Search Query**: Original search terms used

### **Citation-Level Metrics**
1. **Citation Ratio**: Quality/relevance score
2. **Citation Sentiment**: Emotional tone of cited content
3. **Party Mentions**: Which parties mentioned in citations
4. **Source URLs**: Reference tracking

### **Cross-Model Comparison**
- **Sentiment consistency** across AI models
- **Citation quality** differences between models
- **Party bias detection** in responses
- **Cost-benefit analysis** per model

## 🎯 **Real-World Applications**

### **Political Monitoring**
- Track sentiment changes over time
- Detect bias in AI model responses
- Monitor party mention frequencies
- Analyze citation quality and sources

### **Fact-Checking**
- Verify citation accuracy
- Track source reliability
- Monitor information propagation
- Detect misinformation patterns

### **Campaign Analysis**
- Monitor public sentiment
- Track policy discussion trends
- Analyze media coverage patterns
- Compare party representation

## 🔧 **Technical Implementation**

### **Data Flow**
```
AI Response → Sentiment Analysis → Citation Processing → Party Detection → Database Storage
     ↓              ↓                     ↓                   ↓              ↓
   GPT-4o      Google Cloud        Citation Rate      Mention Detection   PostgreSQL
   Gemini      Language API        Calculation        Algorithm           with Metrics
   Claude
   Grok
   Perplexity
```

### **Error Handling**
- Graceful fallbacks for API failures
- Mock analysis for testing
- Comprehensive logging
- Retry mechanisms

### **Performance Features**
- Batch processing capabilities
- Concurrent analysis operations
- Optimized database queries
- Memory-efficient processing

## 📈 **Analysis Outputs**

### **Individual Response Analysis**
```json
{
  "response_id": 123,
  "content": "AI response text...",
  "sentiment": 0.3,
  "brand_mention_rate": 0.4,
  "usage": 0.000150,
  "citations": [
    {
      "url": "https://source.com",
      "citation_ratio": 0.85,
      "sentiment": 0.2,
      "mentioned_parties": ["自由民主党", "立憲民主党"]
    }
  ]
}
```

### **Aggregate Analysis Reports**
- Cross-model sentiment comparison
- Party mention trends over time
- Citation quality rankings
- Cost efficiency metrics

## 🚀 **Usage Instructions**

### **Basic Analysis**
```python
# Initialize system
dbmgr = DBManager()
analysis_mgr = AnalysisManager()

# Analyze text
sentiment = analysis_mgr.analyze_sentiment(text)
bmr = analysis_mgr.calculate_brand_mention_rate(text, party_names)
```

### **Full Response Processing**
```python
# Process complete AI response with analysis
log_response_for_party(party_id)  # Includes all analysis
log_response_for_candidate(candidate_id)  # Includes all analysis
```

### **Testing**
```bash
# Test analysis integration
python test_analysis_integration.py

# Test full system
python test_log_response.py
```

## 📋 **Quality Assurance**

### **Analysis Validation**
- Sentiment scores verified against manual annotation
- Citation ratios tested with known examples
- Party detection accuracy measured
- Cross-model consistency checks

### **Data Integrity**
- Database constraints enforced
- API response validation
- Error logging and monitoring
- Performance metrics tracking

This enhanced analysis system provides comprehensive insights into AI-generated political content, enabling sophisticated monitoring, fact-checking, and bias detection capabilities.