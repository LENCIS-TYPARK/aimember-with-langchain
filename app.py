from langchain import PromptTemplate, LLMChain
from aimemberllm import AimemberLLM

# 커스텀 LLM 인스턴스 생성
llm = AimemberLLM()

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
