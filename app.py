from langchain import PromptTemplate, LLMChain
from aimemberllm import AimemberLLM

# 커스�텀 LLM 인스턴스 생성
llm = AimemberLLM(endpoint="/lottegpt")

# 프롬프트 템플릿 설정
prompt = PromptTemplate(
    input_variables=["question"],
    template="질문: {question}\n답변:",
)

# LLMChain 생성
llm_chain = LLMChain(prompt=prompt, llm=llm)

# 실행
question = "롯데이노베이트의 아이멤버는 무엇인가요?"
answer = llm_chain.run(question)
print(answer)

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
question_search = "롯데이노베이트의 검색 기능은 무엇인가요?"
answer_search = llm_lottegpt_search._call(json.dumps({"query": question_search, "search_resource": {}}))
print(answer_search)

document = "이 문서를 요약해 주세요."
summary = llm_summarization._call(json.dumps({"document": document}))
print(summary)

chat_question = "챗봇에게 질문을 해보세요."
chat_answer = llm_chatgpt._call(json.dumps({"query": chat_question, "history": ''}))
print(chat_answer)

gemini_question = "제미니에게 질문을 해보세요."
gemini_answer = llm_gemini_nostream._call(json.dumps({"query": gemini_question}))
print(gemini_answer)

recommendation_question = "추천 시스템에 대해 설명해 주세요."
recommendation_answer = llm_recommendation._call(json.dumps({"question": recommendation_question, "answer": ''}))
print(recommendation_answer)

code_question = "코드 생성 기능을 설명해 주세요."
code_answer = llm_codegenerate._call(json.dumps({"query": code_question}))
print(code_answer)

wordner_question = "워드NER 기능을 설명해 주세요."
wordner_answer = llm_wordner._call(json.dumps({"query": wordner_question}))
print(wordner_answer)

worddetector_question = "워드 감지 기능을 설명해 주세요."
worddetector_answer = llm_worddetector._call(json.dumps({"query": worddetector_question}))
print(worddetector_answer)

translate_doc = "이 문서를 번역해 주세요."
translate_answer = llm_translate._call(json.dumps({"doc": translate_doc, "src_lang": '', "tgt_lang": ''}))
print(translate_answer)
