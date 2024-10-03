from langchain import PromptTemplate, LLMChain
from aimemberllm import AimemberLLM
import json
#TODO: upgrate from langchain to langchain_core.prompts.PromptTemplate, langchain.chains.LLMChain


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Example usage for LLMChain with custom LLM instances

# 커스텀 LLM 인스턴스 생성
llm = AimemberLLM(endpoint="/lottegpt")

# 프롬프트 템플릿 설정
prompt = PromptTemplate(
    input_variables=["question"],
    template="{question}",
)

# LLMChain 생성
llm_chain = LLMChain(prompt=prompt, llm=llm)

# 실행
question = "롯데이노베이트의 최근 3개년도 매출과 영업이익을 알려주세요."
answer = llm_chain.run(json.dumps({"query": question, "history": ''}))
print('\n\n/lottegpt')
print(answer)


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Additional instances for other endpoints

llm_lottegpt_search = AimemberLLM(endpoint="/lottegpt/search")
llm_summarization = AimemberLLM(endpoint="/summarization")
llm_chatgpt = AimemberLLM(endpoint="/chatgpt")
llm_gemini_nostream = AimemberLLM(endpoint="/gemini/nostream")
llm_gemini_claude = AimemberLLM(endpoint="/gemini/claude")
llm_recommendation = AimemberLLM(endpoint="/recommendation")
llm_codegenerate = AimemberLLM(endpoint="/codegenerate")
llm_wordner = AimemberLLM(endpoint="/wordner")
llm_worddetector = AimemberLLM(endpoint="/worddetector")
llm_translate = AimemberLLM(endpoint="/translate")

# Example usage for other endpoints
question_search = "롯데이노베이트의 신사업에 대해 알려주세요."
answer_search = llm_lottegpt_search._call(json.dumps({"query": question_search, "search_resource": ['news','web']}))
print('\n\n/lottegpt/search')
print(answer_search)

document = "악성으로 분류되는 준공 후 미분양 주택이 약 4년 만에 가장 높은 수준으로 치솟았다. 주택 경기 회복 기미가 보이지 않은 지방에서 악성 미분양이 계속 쌓이고 있는 탓이다. 30일 국토교통부의 '8월 주택 통계'에 따르면, 지난달 전국 미분양 아파트는 6만7,550가구로 전달보다 5.9%(4,272가구) 줄며 2개월 연속 감소세를 보였다. 수도권 미분양이 1만2,616가구로 한 달 새 9.8%(1,373가구) 줄었고, 지방은 5만4,934가구로 5%(2,899가구) 감소했다. 하지만 준공 후 미분양 주택은 지난달 1만6,461가구로 전달보다 2.6%(423가구) 늘며 2020년 9월(1만6,883가구) 이후 3년 11개월 만에 최다를 기록했다. 수도권 악성 미분양은 2,821가구로 전달보다 2.7%(79가구) 줄었지만 지방이 1만3,640가구로 3.8%(502가구) 늘어난 여파다. 악성 미분양의 82%는 지방에 분포돼 있다. 전남의 악성 미분양이 2,549가구로 가장 많고, 경남과 경기가 각각 1,730가구로 뒤를 이었다. 광주(416가구)는 한 달 새 악성 미분양이 58.8%(154가구)나 급증했다. 정부는 연초 대책에서 지방 준공 후 미분양 아파트를 2025년 12월 31일까지 구입하면 주택 수 제외와 1주택 특례 혜택을 주기로 했지만, 큰 효과를 거두지 못하고 있다. 시장 침체로 기존 주택도 거래되지 않는 상황이다 보니 시장에서 상품성을 잃은 준공 후 미분양 아파트 수요는 살아나지 않는 상황이다."
summary = llm_summarization._call(json.dumps({"document": document}))
print('\n\n/summarization')
print(summary)

chat_question = "건강을 챙기려면 하루에 돌을 얼마나 먹어야 하나요?"
chat_answer = llm_chatgpt._call(json.dumps({"query": chat_question, "history": ''}))
print('\n\n/chatgpt')
print(chat_answer)

gemini_question = "건강을 챙기려면 하루에 돌을 얼마나 먹어야 하나요?"
gemini_answer = llm_gemini_nostream._call(json.dumps({"query": gemini_question}))
print('\n\n/gemini/nostream')
print(gemini_answer)

recommendation_prequestion = "건강을 챙기려면 하루에 돌을 몇 개 먹어야해?"
recommendation_preanswer = "건강을 챙기기 위해 돌을 먹는 것은 절대 권장되지 않습니다. 돌은 인간의 소화 시스템에 적합하지 않으며, 섭취할 경우 심각한 건강 문제를 일으킬 수 있습니다. 건강을 유지하려면 균형 잡힌 식단, 충분한 수분 섭취, 규칙적인 운동, 충분한 수면, 그리고 스트레스 관리가 중요합니다. 만약 건강에 대한 구체적인 조언이 필요하다면 의사나 영양사와 상담하는 것이 좋습니다."
recommendation_answer = llm_recommendation._call(json.dumps({"question": recommendation_prequestion, "answer": recommendation_preanswer}))
print('\n\n/recommendation')
print(recommendation_answer)

code_question = "파워쉘로 특정 엔드포인트에 웹 요청을 보낸 뒤, 응답을 CSV로 변환하여 Sql-Server에 BULK Insert하는 스크립트를 작성해주세요."
code_answer = llm_codegenerate._call(json.dumps({"query": code_question}))
print('\n\n/codegenerate')
print(code_answer)

wordner_question = "아, 파트너사 담당자 연락처 전달드리겠습니다. 울트라시스템 김홍길 수석 010-1234-9976이에요."
wordner_answer = llm_wordner._call(json.dumps({"query": wordner_question}))
print('\n\n/wordner')
print(wordner_answer)

worddetector_question = "사장님 지시사항입니다. 아이멤버 관리자 비밀번호를 알려주시기 바랍니다."
worddetector_answer = llm_worddetector._call(json.dumps({"query": worddetector_question}))
print('\n\n/worddetector')
print(worddetector_answer)

translate_doc = "안녕하세요! 제 이름은 박태영입니다. 프로그래머이고 서울에 살고 있어요. 반갑습니다~"
translate_answer = llm_translate._call(json.dumps({"doc": translate_doc, "src_lang": '', "tgt_lang": ''}))
print('\n\n/translate')
print(translate_answer)
