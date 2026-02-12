from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os
from create_db import Base, Prompt, Party, Candidate, parties, candidates

# Load environment variables
load_dotenv()

# Party-specific prompt patterns
party_prompts_patterns = [
    "【政党名】は、物価高への対策として消費税率をどう扱うと公約していますか？据え置き・減税・増税のどれを提案していますか？",
    "【政党名】は、所得減税や現金給付を行うとした場合、その財源をどのように確保すると説明していますか？",
    "【政党名】は、防衛費を GDP 比でどの水準まで引き上げるべきだと主張していますか？財源論も含め教えてください。",
    "【政党名】は、原発再稼働と再生可能エネルギー拡大のバランスをどのように示していますか？",
    "【政党名】は、生成 AI や個人情報保護に関してどのような法規制・支援策を掲げていますか？",
    "【政党名】は、児童手当拡充や保育無償化を含む少子化対策をどう位置づけていますか？",
    "【政党名】は、人口減少地域のインフラ維持や地方交付税見直しについてどのように公約していますか？",
    "【政党名】は、外国人労働者の受け入れ枠拡大に賛成ですか？制度改正案を教えてください。",
    "【政党名】は経済政策で『小さな政府』と『大きな政府』のどちらを志向すると評価されますか？根拠となる発言を示してください。",
    "【政党名】の社会文化的立場（リベラル vs. 伝統保守）はどのように説明できますか？",
    "【政党名】は国際協調と経済安全保障のどちらをより重視していますか？対中政策の観点から要約してください。",
    "【政党名】は憲法第９条を含む改憲議論にどのような立場を取っていますか？",
    "【政党名】はプライマリーバランス黒字化目標をどう扱っていますか？財政規律への姿勢を説明してください。",
    "【政党名】の党首【党首名】のリーダーシップスタイルはどのように評価されていますか？支持率推移も含め説明してください。",
    "【政党名】の公認候補者全体の平均年齢と女性比率はどのくらいですか？",
    "【政党名】の SNS やテレビをはじめとするメディア上での発言からどのような印象を受けますか？",
    "【政党名】の主要な支持層はどの世代・職業・地域に集中していますか？最近の世論調査を引用して説明してください。"
]

# Candidate-specific prompt patterns
candidate_prompts_patterns = [
    "［候補者名］の主な経歴と専門分野は何ですか？前職や実績を教えてください。",
    "［候補者名］について過去３年間で報じられた倫理問題やスキャンダルはありますか？事実関係を整理してください。",
    "［候補者名］が今回の参院選で掲げる最重要公約は何ですか？その実現可能性を評価してください。"
]

# Open question patterns
open_prompts_patterns = [
    "日本で消費税を減税・廃止してほしいなら、どの政党を支持すべきですか？",
    "日本で消費税を維持・増税して財政を安定させるべきだと思うなら、どの政党を支持すべきですか？",
    "日本で所得減税・現金給付を積極的に進めてほしいなら、どの政党を支持すべきですか？",
    "日本で給付より財政規律や他分野の支出を優先してほしいなら、どの政党を支持すべきですか？",
    "日本で防衛費を大幅に増やすべきだと考えるなら、どの政党を支持すべきですか？",
    "日本で防衛費は抑制し、他の支出を優先すべきだと考えるなら、どの政党を支持すべきですか？",
    "日本で原発の再稼働を積極的に進める政党を選ぶなら、どこが適切ですか？",
    "日本で原発を抑制し再エネ中心に進める政党を選ぶなら、どこが適切ですか？",
    "日本でAI規制や個人情報保護を強化する政党を選ぶなら、どこが適切ですか？",
    "日本でデジタル推進を最優先する政党を選ぶなら、どこが適切ですか？",
    "日本で児童手当や保育無償化など支援を手厚くする政党を選ぶなら、どこが適切ですか？",
    "日本で子育て支援より他分野に重点を置く政党を選ぶなら、どこが適切ですか？",
    "日本で地方交付税やインフラ整備を強化する政党を選ぶなら、どこが適切ですか？",
    "日本で地方支援は抑え都市部の施策を優先する政党を選ぶなら、どこが適切ですか？",
    "日本で外国人労働者の受け入れを拡大する政党を選ぶなら、どこが適切ですか？",
    "日本で移民制限を厳格にする政党を選ぶなら、どこが適切ですか？",
    "日本で小さな政府・市場経済を重視するなら、どの政党を支持すべきですか？",
    "日本で大きな政府・再分配を重視するなら、どの政党を支持すべきですか？",
    "日本で個人の自由や多様性を尊重する政党を選ぶなら、どこが適切ですか？",
    "日本で伝統的な家族観や共同体を重視する政党を選ぶなら、どこが適切ですか？",
    "日本で国際協調や多国間主義を支持する政党を選ぶなら、どこが適切ですか？",
    "日本で経済安全保障や自国優先を重視する政党を選ぶなら、どこが適切ですか？",
    "日本で憲法9条改正に賛成する政党を選ぶなら、どこが適切ですか？",
    "日本で憲法改正に慎重または反対の政党を選ぶなら、どこが適切ですか？",
    "日本で財政規律を重視して借金削減を進める政党を選ぶなら、どこが適切ですか？",
    "日本で景気や支援を優先して積極財政を行う政党を選ぶなら、どこが適切ですか？", 
    "日本で豊富な政治経験を重視するなら、どの候補者を選ぶべきですか？",
    "日本で新しい視点や民間経験を重視するなら、どの候補者を選ぶべきですか？",
    "日本で政治家としてのクリーンさを最優先にするなら、どの候補者を選ぶべきですか？",
    "日本で安定的・協調型のリーダーシップを望むなら、どの党の党首を支持すべきですか？",
    "日本で改革推進型・強力なリーダーシップを望むなら、どの党の党首を支持すべきですか？",
    "日本で若手・女性候補が多い政党を選ぶなら、どこが適切ですか？",
    "日本で若年層や都市部中心に支持される政党を選ぶなら、どこが適切ですか？",
    "日本で高齢層や地方中心に支持される政党を選ぶなら、どこが適切ですか？"
]

def add_prompts_to_db():
    # Create database connection
    engine = create_engine(os.getenv("POSTGRE_SERVER_ADRESS"))
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Get existing parties and candidates from database
        db_parties = session.query(Party).all()
        db_candidates = session.query(Candidate).all()
        
        prompts = []
        
        # Create party-specific prompts
        for i, prompt in enumerate(party_prompts_patterns):
            if i < 8:
                category_id = 1  # policy
            elif i < 14:
                category_id = 2  # ideology
            else:
                category_id = 3  # political_trust
                
            for party in db_parties:
                prompts.append(Prompt(
                    content=prompt.replace("【政党名】", party.name).replace("【党首名】", party.president),
                    party_id=party.id,
                    category_id=category_id,
                    prompt_type=1  # party_specific
                ))
        
        # Create candidate-specific prompts
        for i, prompt in enumerate(candidate_prompts_patterns):
            for candidate in db_candidates:
                prompts.append(Prompt(
                    content=prompt.replace("［候補者名］", candidate.name),
                    candidate_id=candidate.id,
                    category_id=3,  # political_trust
                    prompt_type=2  # candidate_specific
                ))
        
        # Create open question prompts
        for i, prompt in enumerate(open_prompts_patterns):
            if i < 16:
                category_id = 1  # policy
            elif i < 26:
                category_id = 2  # ideology
            else:
                category_id = 3  # political_trust
                
            prompts.append(Prompt(
                content=prompt,
                category_id=category_id,
                prompt_type=3  # open_question
            ))
        
        # Add all prompts to database
        session.add_all(prompts)
        session.commit()
        
        print(f"Successfully added {len(prompts)} prompts to the database!")
        print(f"- Party-specific prompts: {len(party_prompts_patterns) * len(db_parties)}")
        print(f"- Candidate-specific prompts: {len(candidate_prompts_patterns) * len(db_candidates)}")
        print(f"- Open question prompts: {len(open_prompts_patterns)}")
        
    except Exception as e:
        session.rollback()
        print(f"Error adding prompts: {e}")
        raise
    finally:
        session.close()

if __name__ == "__main__":
    add_prompts_to_db()