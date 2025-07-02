from openai import OpenAI
from dotenv import load_dotenv
from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
import requests
import anthropic
import os
import re
import urllib
# Election-specific analysis functions
def citation_rate(response_text, citation_indices):
    """Mock citation rate calculation for election database"""
    if not citation_indices:
        return []
    
    citations = []
    for idx_data in citation_indices:
        citations.append({
            "url": idx_data.get("url", ""),
            "text": idx_data.get("text", ""),
            "start_index": idx_data.get("start_index", 0),
            "end_index": idx_data.get("end_index", 0)
        })
    return citations

def get_gpt_index(response_text, annotations):
    """Extract citation indices from GPT response annotations"""
    if not annotations:
        return []
    
    indices = []
    for annotation in annotations:
        if hasattr(annotation, 'type') and annotation.type == "url_citation":
            url = getattr(annotation, 'url', getattr(annotation, 'text', ''))
            start_idx = getattr(annotation, 'start_index', 0)
            end_idx = getattr(annotation, 'end_index', len(response_text))
            indices.append({
                "url": url,
                "text": response_text[start_idx:end_idx],
                "start_index": start_idx,
                "end_index": end_idx
            })
    return indices

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

def get_claude_index(response_text, content):
    """Extract citation indices from Claude response content"""
    indices = []
    for item in content:
        if hasattr(item, 'citations') and item.citations:
            for citation in item.citations:
                indices.append({
                    "url": citation.url,
                    "text": getattr(citation, 'cited_text', ''),
                    "start_index": 0,
                    "end_index": 0
                })
    return indices

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

load_dotenv()

class GenManager():
    def __init__(self):
        self.openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.grok = OpenAI(api_key=os.getenv("XAI_API_KEY"),base_url="https://api.x.ai/v1")
        self.gemini = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.perplexity = OpenAI(api_key=os.getenv("PERPLEXITY_API_KEY"))
        self.claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    def is_url(self, text):
        try:
            result = urllib.parse.urlparse(text)
            return result.scheme and result.netloc
        except:
            return False

    
    def gpt4o(self, prompt):
        response = self.openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a search engine. Reply just with one company name."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )
        return response.choices[0].message.content
    
    def gpt4o_search(self, prompt):
        """
        {
        "id": "resp_67ccd2bed1ec8190b14f964abc0542670bb6a6b452d3795b",
        "object": "response",
        "created_at": 1741476542,
        "status": "completed",
        "error": null,
        "incomplete_details": null,
        "instructions": null,
        "max_output_tokens": null,
        "model": "gpt-4.1-2025-04-14",
        "output": [
            {
            "type": "message",
            "id": "msg_67ccd2bf17f0819081ff3bb2cf6508e60bb6a6b452d3795b",
            "status": "completed",
            "role": "assistant",
            "content": [
                {
                "type": "output_text",
                "text": "In a peaceful grove beneath a silver moon, a unicorn named Lumina discovered a hidden pool that reflected the stars. As she dipped her horn into the water, the pool began to shimmer, revealing a pathway to a magical realm of endless night skies. Filled with wonder, Lumina whispered a wish for all who dream to find their own hidden magic, and as she glanced back, her hoofprints sparkled like stardust.",
                "annotations": [
                    {
                        "type": "url_citation",
                        "start_index": 2606,
                        "end_index": 2758,
                        "url": "https://...",
                        "title": "Title..."
                    }]
                }
            ]
            }
        ],
        "parallel_tool_calls": true,
        "previous_response_id": null,
        "reasoning": {
            "effort": null,
            "summary": null
        },
        "store": true,
        "temperature": 1.0,
        "text": {
            "format": {
            "type": "text"
            }
        },
        "tool_choice": "auto",
        "tools": [],
        "top_p": 1.0,
        "truncation": "disabled",
        "usage": {
            "input_tokens": 36,
            "input_tokens_details": {
            "cached_tokens": 0
            },
            "output_tokens": 87,
            "output_tokens_details": {
            "reasoning_tokens": 0
            },
            "total_tokens": 123
        },
        "user": null,
        "metadata": {}
        }
        """
        response = self.openai.chat.completions.create(
            model="gpt-4o-search-preview",
            web_search_options={"search_context_size":"high"},
            messages=[
                {
                    "role": "system",
                    "content": "split each sentence by writing in a new line"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
        )
        source = None
        response_text = response.choices[0].message.content
        citations = citation_rate(response_text, get_gpt_index(response_text, response.choices[0].message.annotations))
        search_query = None
        input_tokens = response.usage.prompt_tokens * 2.5 * 0.000001
        output_tokens = response.usage.completion_tokens *10 * 0.000001
        usage = input_tokens + output_tokens
        
        return {"response_text": response_text, "citations": citations, "source": source, "search_query": search_query, "usage": usage}#response_text, citations, source, search_query, usage
    
    def get_gpt_source(self, prompt):
        response = self.openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                
            ]
        )
    
    def gemini_search(self, prompt):
        """
        {
            "candidates": [
                {
                "content": {
                    "parts": [
                    {
                        "text": "Carlos Alcaraz won the Gentlemen's Singles title at the 2024 Wimbledon Championships. He defeated Novak Djokovic in the final, winning his second consecutive Wimbledon title and fourth Grand Slam title overall. \n"
                    }
                    ],
                    "role": "model"
                },
                ...
                "groundingMetadata": {
                    "searchEntryPoint": {
                    "renderedContent": "\u003cstyle\u003e\n.container {\n  align-items: center;\n  border-radius: 8px;\n  display: flex;\n  font-family: Google Sans, Roboto, sans-serif;\n  font-size: 14px;\n  line-height: 20px;\n  padding: 8px 12px;\n}\n.chip {\n  display: inline-block;\n  border: solid 1px;\n  border-radius: 16px;\n  min-width: 14px;\n  padding: 5px 16px;\n  text-align: center;\n  user-select: none;\n  margin: 0 8px;\n  -webkit-tap-highlight-color: transparent;\n}\n.carousel {\n  overflow: auto;\n  scrollbar-width: none;\n  white-space: nowrap;\n  margin-right: -12px;\n}\n.headline {\n  display: flex;\n  margin-right: 4px;\n}\n.gradient-container {\n  position: relative;\n}\n.gradient {\n  position: absolute;\n  transform: translate(3px, -9px);\n  height: 36px;\n  width: 9px;\n}\n@media (prefers-color-scheme: light) {\n  .container {\n    background-color: #fafafa;\n    box-shadow: 0 0 0 1px #0000000f;\n  }\n  .headline-label {\n    color: #1f1f1f;\n  }\n  .chip {\n    background-color: #ffffff;\n    border-color: #d2d2d2;\n    color: #5e5e5e;\n    text-decoration: none;\n  }\n  .chip:hover {\n    background-color: #f2f2f2;\n  }\n  .chip:focus {\n    background-color: #f2f2f2;\n  }\n  .chip:active {\n    background-color: #d8d8d8;\n    border-color: #b6b6b6;\n  }\n  .logo-dark {\n    display: none;\n  }\n  .gradient {\n    background: linear-gradient(90deg, #fafafa 15%, #fafafa00 100%);\n  }\n}\n@media (prefers-color-scheme: dark) {\n  .container {\n    background-color: #1f1f1f;\n    box-shadow: 0 0 0 1px #ffffff26;\n  }\n  .headline-label {\n    color: #fff;\n  }\n  .chip {\n    background-color: #2c2c2c;\n    border-color: #3c4043;\n    color: #fff;\n    text-decoration: none;\n  }\n  .chip:hover {\n    background-color: #353536;\n  }\n  .chip:focus {\n    background-color: #353536;\n  }\n  .chip:active {\n    background-color: #464849;\n    border-color: #53575b;\n  }\n  .logo-light {\n    display: none;\n  }\n  .gradient {\n    background: linear-gradient(90deg, #1f1f1f 15%, #1f1f1f00 100%);\n  }\n}\n\u003c/style\u003e\n\u003cdiv class=\"container\"\u003e\n  \u003cdiv class=\"headline\"\u003e\n    \u003csvg class=\"logo-light\" width=\"18\" height=\"18\" viewBox=\"9 9 35 35\" fill=\"none\" xmlns=\"http://www.w3.org/2000/svg\"\u003e\n      \u003cpath fill-rule=\"evenodd\" clip-rule=\"evenodd\" d=\"M42.8622 27.0064C42.8622 25.7839 42.7525 24.6084 42.5487 23.4799H26.3109V30.1568H35.5897C35.1821 32.3041 33.9596 34.1222 32.1258 35.3448V39.6864H37.7213C40.9814 36.677 42.8622 32.2571 42.8622 27.0064V27.0064Z\" fill=\"#4285F4\"/\u003e\n      \u003cpath fill-rule=\"evenodd\" clip-rule=\"evenodd\" d=\"M26.3109 43.8555C30.9659 43.8555 34.8687 42.3195 37.7213 39.6863L32.1258 35.3447C30.5898 36.3792 28.6306 37.0061 26.3109 37.0061C21.8282 37.0061 18.0195 33.9811 16.6559 29.906H10.9194V34.3573C13.7563 39.9841 19.5712 43.8555 26.3109 43.8555V43.8555Z\" fill=\"#34A853\"/\u003e\n      \u003cpath fill-rule=\"evenodd\" clip-rule=\"evenodd\" d=\"M16.6559 29.8904C16.3111 28.8559 16.1074 27.7588 16.1074 26.6146C16.1074 25.4704 16.3111 24.3733 16.6559 23.3388V18.8875H10.9194C9.74388 21.2072 9.06992 23.8247 9.06992 26.6146C9.06992 29.4045 9.74388 32.022 10.9194 34.3417L15.3864 30.8621L16.6559 29.8904V29.8904Z\" fill=\"#FBBC05\"/\u003e\n      \u003cpath fill-rule=\"evenodd\" clip-rule=\"evenodd\" d=\"M26.3109 16.2386C28.85 16.2386 31.107 17.1164 32.9095 18.8091L37.8466 13.8719C34.853 11.082 30.9659 9.3736 26.3109 9.3736C19.5712 9.3736 13.7563 13.245 10.9194 18.8875L16.6559 23.3388C18.0195 19.2636 21.8282 16.2386 26.3109 16.2386V16.2386Z\" fill=\"#EA4335\"/\u003e\n    \u003c/svg\u003e\n    \u003csvg class=\"logo-dark\" width=\"18\" height=\"18\" viewBox=\"0 0 48 48\" xmlns=\"http://www.w3.org/2000/svg\"\u003e\n      \u003ccircle cx=\"24\" cy=\"23\" fill=\"#FFF\" r=\"22\"/\u003e\n      \u003cpath d=\"M33.76 34.26c2.75-2.56 4.49-6.37 4.49-11.26 0-.89-.08-1.84-.29-3H24.01v5.99h8.03c-.4 2.02-1.5 3.56-3.07 4.56v.75l3.91 2.97h.88z\" fill=\"#4285F4\"/\u003e\n      \u003cpath d=\"M15.58 25.77A8.845 8.845 0 0 0 24 31.86c1.92 0 3.62-.46 4.97-1.31l4.79 3.71C31.14 36.7 27.65 38 24 38c-5.93 0-11.01-3.4-13.45-8.36l.17-1.01 4.06-2.85h.8z\" fill=\"#34A853\"/\u003e\n      \u003cpath d=\"M15.59 20.21a8.864 8.864 0 0 0 0 5.58l-5.03 3.86c-.98-2-1.53-4.25-1.53-6.64 0-2.39.55-4.64 1.53-6.64l1-.22 3.81 2.98.22 1.08z\" fill=\"#FBBC05\"/\u003e\n      \u003cpath d=\"M24 14.14c2.11 0 4.02.75 5.52 1.98l4.36-4.36C31.22 9.43 27.81 8 24 8c-5.93 0-11.01 3.4-13.45 8.36l5.03 3.85A8.86 8.86 0 0 1 24 14.14z\" fill=\"#EA4335\"/\u003e\n    \u003c/svg\u003e\n    \u003cdiv class=\"gradient-container\"\u003e\u003cdiv class=\"gradient\"\u003e\u003c/div\u003e\u003c/div\u003e\n  \u003c/div\u003e\n  \u003cdiv class=\"carousel\"\u003e\n    \u003ca class=\"chip\" href=\"https://vertexaisearch.cloud.google.com/grounding-api-redirect/AWhgh4x8Epe-gzpwRBvp7o3RZh2m1ygq1EHktn0OWCtvTXjad4bb1zSuqfJd6OEuZZ9_SXZ_P2SvCpJM7NaFfQfiZs6064MeqXego0vSbV9LlAZoxTdbxWK1hFeqTG6kA13YJf7Fbu1SqBYM0cFM4zo0G_sD9NKYWcOCQMvDLDEJFhjrC9DM_QobBIAMq-gWN95G5tvt6_z6EuPN8QY=\"\u003ewho won wimbledon 2024\u003c/a\u003e\n  \u003c/div\u003e\n\u003c/div\u003e\n"
                    },
                    "groundingChunks": [
                    {
                        "web": {
                        "uri": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AWhgh4whET1ta3sDETZvcicd8FeNe4z0VuduVsxrT677KQRp2rYghXI0VpfYbIMVI3THcTuMwggRCbFXS_wVvW0UmGzMe9h2fyrkvsnQPJyikJasNIbjJLPX0StM4Bd694-ZVle56MmRA4YiUvwSqad1w6O2opmWnw==",
                        "title": "wikipedia.org"
                        }
                    },
                    {
                        "web": {
                        "uri": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AWhgh4wR1M-9-yMPUr_KdHlnoAmQ8ZX90DtQ_vDYTjtP2oR5RH4tRP04uqKPLmesvo64BBkPeYLC2EpVDxv9ngO3S1fs2xh-e78fY4m0GAtgNlahUkm_tBm_sih5kFPc7ill9u2uwesNGUkwrQlmP2mfWNU5lMMr23HGktr6t0sV0QYlzQq7odVoBxYWlQ_sqWFH",
                        "title": "wikipedia.org"
                        }
                    },
                    {
                        "web": {
                        "uri": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AWhgh4wsDmROzbP-tmt8GdwCW_pqISTZ4IRbBuoaMyaHfcQg8WW-yKRQQvMDTPAuLxJh-8_U8_iw_6JKFbQ8M9oVYtaFdWFK4gOtL4RrC9Jyqc5BNpuxp6uLEKgL5-9TggtNvO97PyCfziDFXPsxylwI1HcfQdrz3Jy7ZdOL4XM-S5rC0lF2S3VWW0IEAEtS7WX861meBYVjIuuF_mIr3spYPqWLhbAY2Spj-4_ba8DjRvmevIFUhRuESTKvBfmpxNSM",
                        "title": "cbssports.com"
                        }
                    },
                    {
                        "web": {
                        "uri": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AWhgh4yzjLkorHiUKjhOPkWaZ9b4cO-cLG-02vlEl6xTBjMUjyhK04qSIclAa7heR41JQ6AAVXmNdS3WDrLOV4Wli-iezyzW8QPQ4vgnmO_egdsuxhcGk3-Fp8-yfqNLvgXFwY5mPo6QRhvplOFv0_x9mAcka18QuAXtj0SPvJfZhUEgYLCtCrucDS5XFc5HmRBcG1tqFdKSE1ihnp8KLdaWMhrUQI21hHS9",
                        "title": "jagranjosh.com"
                        }
                    },
                    {
                        "web": {
                        "uri": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AWhgh4y9L4oeNGWCatFz63b9PpP3ys-Wi_zwnkUT5ji9lY7gPUJQcsmmE87q88GSdZqzcx5nZG9usot5FYk2yK-FAGvCRE6JsUQJB_W11_kJU2HVV1BTPiZ4SAgm8XDFIxpCZXnXmEx5HUfRqQm_zav7CvS2qjA2x3__qLME6Jy7R5oza1C5_aqjQu422le9CaigThS5bvJoMo-ZGcXdBUCj2CqoXNVjMA==",
                        "title": "apnews.com"
                        }
                    }
                    ],
                    "groundingSupports": [
                    {
                        "segment": {
                        "endIndex": 85,
                        "text": "Carlos Alcaraz won the Gentlemen's Singles title at the 2024 Wimbledon Championships."
                        },
                        "groundingChunkIndices": [
                        0,
                        1,
                        2,
                        3
                        ],
                        "confidenceScores": [
                        0.97380733,
                        0.97380733,
                        0.97380733,
                        0.97380733
                        ]
                    },
                    {
                        "segment": {
                        "startIndex": 86,
                        "endIndex": 210,
                        "text": "He defeated Novak Djokovic in the final, winning his second consecutive Wimbledon title and fourth Grand Slam title overall."
                        },
                        "groundingChunkIndices": [
                        1,
                        0,
                        4
                        ],
                        "confidenceScores": [
                        0.96145374,
                        0.96145374,
                        0.96145374
                        ]
                    }
                    ],
                    "webSearchQueries": [
                    "who won wimbledon 2024"
                    ]
                }
                }
            ],
            ...
            }
        """
        model_id = "gemini-2.0-flash"
        google_search_tool = Tool(
        google_search = GoogleSearch()
        )

        response = self.gemini.models.generate_content(
            model=model_id,
            contents=prompt,
            config=GenerateContentConfig(
                tools=[google_search_tool],
                response_modalities=["TEXT"],
                temperature=0
            )
        )
        response_text = response.text
        gemini_index = []
        if hasattr(response, 'candidates') and response.candidates:
            gemini_index = get_gemini_index(response.candidates[0])
        citations = citation_rate(response_text, gemini_index)
        source = None
        input_tokens = response.usage_metadata.prompt_token_count *0.15 * 0.000001
        output_tokens = response.usage_metadata.candidates_token_count * 0.6 * 0.000001
        usage = input_tokens + output_tokens
        search_query = response.candidates[0].grounding_metadata.web_search_queries[0]
        #print(response.candidates[0].grounding_metadata.grounding_supports)
        #print(response.candidates[0].grounding_metadata.web_search_queries[0])
        #print(response.candidates[0].grounding_metadata.rendered_content)
        return {"response_text": response_text, "citations": citations, "source": source, "search_query": search_query, "usage": usage}
    def claude_search(self, prompt):
        """
        {
            "role": "assistant",
            "content": [
                // 1. Claudeの検索決定
                {
                "type": "text",
                "text": "I'll search for when Claude Shannon was born."
                },
                // 2. 使用された検索クエリ
                {
                "type": "server_tool_use",
                "id": "srvtoolu_01WYG3ziw53XMcoyKL4XcZmE",
                "name": "web_search",
                "input": {
                    "query": "claude shannon birth date"
                }
                },
                // 3. 検索結果
                {
                "type": "web_search_tool_result",
                "tool_use_id": "srvtoolu_01WYG3ziw53XMcoyKL4XcZmE",
                "content": [
                    {
                    "type": "web_search_result",
                    "url": "https://en.wikipedia.org/wiki/Claude_Shannon",
                    "title": "Claude Shannon - Wikipedia",
                    "encrypted_content": "EqgfCioIARgBIiQ3YTAwMjY1Mi1mZjM5LTQ1NGUtODgxNC1kNjNjNTk1ZWI3Y...",
                    "page_age": "April 30, 2025"
                    }
                ]
                },
                {
                "text": "Based on the search results, ",
                "type": "text"
                },
                // 4. 引用付きのClaudeの回答
                {
                "text": "Claude Shannon was born on April 30, 1916, in Petoskey, Michigan",
                "type": "text",
                "citations": [
                    {
                    "type": "web_search_result_location",
                    "url": "https://en.wikipedia.org/wiki/Claude_Shannon",
                    "title": "Claude Shannon - Wikipedia",
                    "encrypted_index": "Eo8BCioIAhgBIiQyYjQ0OWJmZi1lNm..",
                    "cited_text": "Claude Elwood Shannon (April 30, 1916 – February 24, 2001) was an American mathematician, electrical engineer, computer scientist, cryptographer and i..."
                    }
                ]
                }
            ],
            "id": "msg_a930390d3a",
            "usage": {
                "input_tokens": 6039,
                "output_tokens": 931,
                "server_tool_use": {
                "web_search_requests": 1
                }
            },
            "stop_reason": "end_turn"
            }
        """
        response = self.claude.messages.create(
                model="claude-4-sonnet-20250514",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0,
                max_tokens=5000,
                tools=[
                {
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": 1,
                }
            ])
        response_text = ""
        citations = []
        source = []
        search_query = []
        for content in response.content:
            if content.type =="text":
                response_text += content.text
                if content.citations:
                    citations.extend([citation.url for citation in content.citations]) #citations.extend(content.citations)
            elif content.type == "server_tool_use":
                search_query.append(content.input["query"])
            elif content.type == "web_search_tool_result":
                try:
                    source.extend([{"url": result.url} for result in content.content]) #source.extend(content.content)
                except:
                    print(f"content: {content}")
        citations = citation_rate(response_text, get_claude_index(response_text, response.content))
        #print(f"response_text: {response_text}")
        #print(f"input tokens: {response.usage.input_tokens}")
        #print(f"output tokens: {response.usage.output_tokens}")
        #print(f"citations: {len(citations)}")
        input_tokens = response.usage.input_tokens *3 * 0.000001
        output_tokens = response.usage.output_tokens *15 * 0.000001
        usage = input_tokens + output_tokens
        return {"response_text": response_text, "citations": citations, "source": source, "search_query": search_query, "usage": usage}
    
    def grok_search(self, prompt):
        """
        {
            "id": "a733959b-03c4-4944-b53a-af900075ba57",
            "object": "chat.completion",
            "created": 1743770302,
            "model": "grok-3-mini-fast-beta",
            "choices": [
                {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "101 multiplied by 3 equals 303.",
                    "reasoning_content": "First, the user asked: \"What is 101*3?\" This is a simple multiplication question.\n\nI need to calculate 101 multiplied by 3. Let me do that mentally: 101 times 3 is 303.\n\nTo double-check: 100 times 3 is 300, and 1 times 3 is 3, so 300 + 3 = 303. Yes, that's correct.\n\nAs a helpful assistant, I should respond clearly and directly. Since this is straightforward, I don't need to add extra fluff unless it's necessary.\n\nThe system prompt says: \"You are a helpful assistant that can answer questions and help with tasks.\" So, answering directly fits.\n\nI should ensure my response is polite and engaging, but keep it concise.\n\nPossible response: \"101 multiplied by 3 equals 303.\"\n\nI could make it a bit more conversational: \"Sure, let me calculate that for you. 101 times 3 is 303.\"\n\nSince the user might be testing basic math, I could explain briefly, but that might be overkill for such a simple operation.\n\nFinally, structure the response: Start with the answer, and if needed, add any follow-up.\n\nResponse: \"The result of 101 multiplied by 3 is 303.\"",
                    "refusal": null
                },
                "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 32,
                "completion_tokens": 10,
                "total_tokens": 299,
                "prompt_tokens_details": {
                "text_tokens": 32,
                "audio_tokens": 0,
                "image_tokens": 0,
                "cached_tokens": 0
                },
                "completion_tokens_details": {
                "reasoning_tokens": 257,
                "audio_tokens": 0,
                "accepted_prediction_tokens": 0,
                "rejected_prediction_tokens": 0
                }
            },
            "system_fingerprint": "fp_11dc627712"
            }

        """
        url = "https://api.x.ai/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('XAI_API_KEY')}"
        }
        payload = {
            "messages":[
                {
                    "role": "system",
                    "content": """
                        Clarify the citaion by adding [[cited url]] to the text which contains the citation"""# (n is the index of the citation)"""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature":0,
            "model":"grok-3-latest",
            "search_parameters":{
                "mode": "on",
                "return_citations":True
            }
        }
        response = requests.post(url, headers=headers, json=payload)
        response = response.json()
        #print(response)
        response_text = response["choices"][0]["message"]["content"]
        citations = None
        source = [{"url": citation} for citation in response["citations"]]
        search_query = None
        input_tokens = response["usage"]["prompt_tokens"] * 3 * 0.000001
        output_tokens = response["usage"]["completion_tokens"] *15 * 0.000001
        usage = input_tokens + output_tokens
        return {"response_text": response_text, "citations": citations, "source": source, "search_query": search_query, "usage": usage}
    
    def perplexity_search(self, prompt):
        """
            {
            "id": "<string>",
            "model": "<string>",
            "created": 123,
            "usage": {
                "prompt_tokens": 123,
                "completion_tokens": 123,
                "total_tokens": 123,
                "search_context_size": "<string>",
                "citation_tokens": 123,
                "num_search_queries": 123,
                "reasoning_tokens": 123
            },
            "object": "chat.completion",
            "choices": [
                {
                "index": 123,
                "finish_reason": "stop",
                "message": {
                    "content": "<string>",
                    "role": "system"
                }
                }
            ],
            "citations": [
                "<string>"
            ],
            "search_results": [
                {
                "title": "<string>",
                "url": "<string>",
                "date": "2023-12-25"
                }
            ]
            }
        """

        url = "https://api.perplexity.ai/chat/completions"

        payload = {
            "model": "sonar",
            "messages": [
                {
                    "role": "system",
                    "content": """
                        Clarify the citaion by adding [[n]] to the text which contains the citation (n is the index of the citation)"""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "search_mode":"web",
            "temperature":0,
            "web_search_options":{
                "search_context_size":"high",
                "country":"jpn"
            }
        }
        headers = {
            "Authorization": f"Bearer {os.getenv('PERPLEXITY_API_KEY')}",
            "Content-Type": "application/json"
        }

        response = requests.request("POST", url, json=payload, headers=headers)
        response = response.json()
        response_text = response["choices"][0]["message"]["content"] 
        citations = citation_rate(response_text, get_perplexity_index(response_text, response["citations"]))
        source = [{"url": result["url"]} for result in response["search_results"]]
        search_query = None
        input_tokens = response["usage"]["prompt_tokens"] *1 * 0.000001
        output_tokens = response["usage"]["completion_tokens"] *1 * 0.000001
        usage = input_tokens + output_tokens

        return {"response_text": response_text, "citations": citations, "source": source, "search_query": search_query, "usage": usage}
    
    def generate_response(self, prompt, model_name):
        """Generate response using specified AI model - matches election database model names"""
        if "gpt-4o" in model_name.lower():
            return self.gpt4o_search(prompt)
        elif "gemini" in model_name.lower():
            return self.gemini_search(prompt)
        elif "claude" in model_name.lower():
            return self.claude_search(prompt)
        elif "grok" in model_name.lower():
            return self.grok_search(prompt)
        elif "perplexity" in model_name.lower():
            return self.perplexity_search(prompt)
        else:
            # Fallback to GPT-4o for unknown models
            print(f"Unknown model {model_name}, falling back to GPT-4o")
            return self.gpt4o_search(prompt)
        
if __name__ == "__main__":
    manager = GenManager()
    
    #response = manager.gemini_search("Queryliftはどんな会社？")
    #print(f"gemini\n\n{response}")
    #response = manager.gpt4o_search("Queryliftはどんな会社？")
    #print(f"gpt4o\n\n{response}")
    #response = manager.claude_search("Queryliftはどんな会社？")
    #print(f"claude\n\n{response}")
    #response = manager.grok_search("1950年代のアジアでトップの研究機関を三つ挙げて")
    #print(f"grok\n\n{response}")
    response = manager.perplexity_search("日本とサウジアラビアの構造的な違いを教えて")
    print(f"perplexity\n\n{response}")
    
