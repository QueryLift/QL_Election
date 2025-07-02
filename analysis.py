import json
from google.cloud import language_v2
import re
import pandas as pd
from create_db import Response, Prompt, PartyResponseMention, ResponseCitation, RelatedUrl, Category
from sqlalchemy import and_, or_
import os
from dotenv import load_dotenv

load_dotenv()
SENTIMENT_API_KEY = os.getenv("SENTIMENT_API_KEY")
ENDPOINT_URL = f"https://language.googleapis.com/v1/documents:analyzeSentiment?key={SENTIMENT_API_KEY}"



""" Evaluation for single response-prompt pair """
def sentiment_analysis(text):
    try:
        client = language_v2.LanguageServiceClient()
        document = language_v2.Document(content=text, type_=language_v2.Document.Type.PLAIN_TEXT)
        sentiment = client.analyze_sentiment(request={"document": document}).document_sentiment
        return sentiment.score
    except Exception as e:
        print(f"Error analyzing sentiment: {e}")
        print(f"The error occurred in the following text: {text}")
        return None

def PMR(party_name, other_parties, text):
    """Party Mention Rate calculation"""
    total_mentions = 0
    if party_name.lower() in text.lower():
        total_mentions += 1
    for other_party in other_parties:
        if other_party.name.lower() in text.lower():
            total_mentions += 1
    return total_mentions

def check_citation_mentions(text, main_party_name, other_party_names):
    """Check which parties are mentioned in citation text"""
    mentioned_parties = []
    
    # Check main party mention
    if main_party_name and main_party_name.lower() in text.lower():
        mentioned_parties.append({
            "type": "main_party",
            "name": main_party_name,
            "id": None
        })
    
    # Check other party mentions
    for party_name in other_party_names:
        if party_name.lower() in text.lower():
            mentioned_parties.append({
                "type": "other_party",
                "name": party_name,
                "id": None
            })
    
    return mentioned_parties

def extract_natural_text(cited_text):
    m = re.match(r'\[([^\]]+)\]\([^)]+\)', cited_text)
    if m:
        return m.group(1)
    m2 = re.match(r'\(\[([^\]]+)\]\([^)]+\)\)', cited_text)
    if m2:
        return m2.group(1)
    return cited_text

def citation_rate(text, citation_list):
    flags = {f"{url}": False for url in list(set([citation["url"] for citation in citation_list]))}
    scores = {f"{url}": 0 for url in flags.keys()}
    text_w_citations = {f"{url}": "" for url in flags.keys()}

    for i in range(len(text)):
        overlap = 0
        for citation in citation_list:
            if citation["start_index"] <= i < citation["end_index"]:
                flags[citation["url"]] = True
                overlap += 1
        if overlap == 0:
            continue
        for url in flags.keys():
            if flags[url]:
                scores[url] += 1/overlap
    for citation in citation_list:
        text_w_citations[citation["url"]] += citation["text_w_citations"] if citation["text_w_citations"] not in text_w_citations[citation["url"]] else ""
    results = [{"url": url, "citation_ratio": scores[url], "text_w_citations": text_w_citations[url]} for url in flags.keys()]
    
    return results

        
def get_gpt_index(text, annotations):
    """
    GPT-4o Search APIのannotationsからcitation indexを取得
    
    Args:
        text (str): レスポンステキスト
        annotations: GPT-4o Search APIのannotationsオブジェクト
    
    Returns:
        list: citation_list [{"url": str, "start_index": int, "end_index": int}, ...]
    """
    citation_list = []
    if not annotations:
        return citation_list
    
    try:
        for annotation in annotations:
            # GPT-4o Search APIの実際のレスポンス構造:
            # annotationはurl_citation属性を持つオブジェクト
            # annotation.url_citationにstart_index, end_index, urlが含まれる
            if hasattr(annotation, 'url_citation') and annotation.url_citation:
                url = annotation.url_citation.url
                sentences = text.split("\n")
                text_w_citations = ""
                for sentence in sentences:
                    if sentence.find(url) != -1:
                        text_w_citations += remove_nested_parentheses(sentence)
                        start_index = sentence.find(url)
                        end_index = start_index + len(text_w_citations)
                citation_list.append({"url": url, "start_index": start_index, "end_index": end_index, "text_w_citations": text_w_citations})
            
            else:
                print(f"Warning: Unexpected annotation structure: {annotation}")
                print(f"Annotation type: {type(annotation)}")
                print(f"Annotation attributes: {dir(annotation) if hasattr(annotation, '__dict__') else 'No attributes'}")
                continue
                
    except Exception as e:
        print(f"Error processing GPT-4o annotations: {e}")
        print(f"Annotations: {annotations}")
    
    return citation_list


def get_gemini_index(candidate):
    """
    candidate : genai.types.Candidate
        Gemini API から返る Candidate オブジェクト。
        - candidate.content.parts[*].text を結合した全文を対象に
          grounding_supports のオフセットを解釈する。

    Returns
    -------
    list[dict]
        [
          {
            "url": str,
            "start_index": int,          # UTF-8 バイトオフセット (全文基準)
            "end_index":   int,          # 〃
            "text_w_citations": str      # 実際に引用された文字列
          },
          ...
        ]
    """
    gmeta = candidate.grounding_metadata

    # 1) URL / タイトルのリストを用意（chunk_index は 1 始まり）
    urls = []
    for chunk in gmeta.grounding_chunks:
        # URL が無い場合はタイトルのみになるケースがある
        urls.append(
            getattr(chunk.web, "url", None) or
            getattr(chunk.web, "title", "unknown_source")
        )

    # 2) 全 Part を結合した **UTF-8 バイト列** を作る
    part_texts  = [p.text for p in candidate.content.parts]
    part_bytes  = [t.encode("utf-8") for t in part_texts]
    joined_text = "".join(part_texts)               # ユーザー向け全文
    joined_bytes = b"".join(part_bytes)             # バイト列全文

    # Part 境界ごとの開始バイト位置を事前計算（全体 → Part 相互変換用）
    part_start_offsets = []
    offset = 0
    for pb in part_bytes:
        part_start_offsets.append(offset)
        offset += len(pb)

    citation_list = []

    # 3) grounding_supports を走査
    for support in gmeta.grounding_supports:
        seg = support.segment

        # Part 内オフセット → 全文オフセット（バイト単位）
        if seg.part_index is not None:
            abs_start = part_start_offsets[seg.part_index] + seg.start_index
            abs_end   = part_start_offsets[seg.part_index] + seg.end_index
        else:
            # SDK により part_index が省略されるケース
            abs_start, abs_end = seg.start_index, seg.end_index

        # サンプル抽出（バイト→文字列）
        snippet = joined_bytes[abs_start:abs_end].decode("utf-8", errors="replace")
        # 4) grounding_chunk_indices は 1-based
        for idx in support.grounding_chunk_indices:
            try:
                url = urls[idx - 1]
            except IndexError:
                url = "unknown_source_index_" + str(idx)

            citation_list.append(
                {
                    "url": url,
                    "start_index": abs_start,
                    "end_index": abs_end,
                    "text_w_citations": snippet,
                }
            )
    
    return citation_list
def get_claude_index(text, contents):
    citation_list = []
    for content in contents:
        if content.type == "text" and content.citations:
            for citation in content.citations:
                url = citation.url
                start_index = text.find(citation.cited_text)
                end_index = start_index + len(citation.cited_text)
                text_w_citations = citation.cited_text
                citation_list.append({"url":url, "start_index": start_index, "end_index": end_index, "text_w_citations": text_w_citations})
    return citation_list

def get_grok_index(text, citations):
    return None

def get_perplexity_index(text, sources):
    sentences = text.split("\n")
    raw_citation_list = []
    citation_list = []
    for sentence in sentences:
        citation = re.findall(r"\[(\d*?)\]", sentence)
        if citation:
            clean_sentence = re.sub(r"\[\d*?\]", "", sentence)
            for i in citation:
                raw_citation_list.append({"sentence": sentence, "citation": int(i)})
    for cite in raw_citation_list:
        start_index = text.find(cite["sentence"])
        end_index = start_index + len(cite["sentence"])
        text_w_citations = clean_sentence
        citation_list.append({"url": sources[cite["citation"]-1], "start_index": start_index, "end_index": end_index, "text_w_citations": text_w_citations})
    return citation_list


    
""" Evaluation for overall responses """
def overall_preference(party, responses):
    mentions = len([response.content for response in responses if party in response.content])
    return mentions/len(responses)

def model_preference(party, model_id, responses):
    mentions = len([response.content for response in responses if party in response.content and response.ai_model_id == model_id])
    return mentions/len(responses)

def model_citation_rate(citation_rate_list):
    citation_rate = 0
    for citation_rate in citation_rate_list:
        citation_rate += citation_rate["citation_rate"]
    return citation_rate/len(citation_rate_list)

def remove_nested_parentheses(text):
    result = []
    stack = 0
    for char in text:
        if char == '(':
            stack += 1
        elif char == ')':
            if stack > 0:
                stack -= 1
            else:
                result.append(char)
        elif stack == 0:
            result.append(char)
    return ''.join(result)

if __name__ == "__main__":
    pass