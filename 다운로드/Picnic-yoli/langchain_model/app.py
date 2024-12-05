import streamlit as st
from pgvector.psycopg2 import register_vector
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import numpy as np
import os
import psycopg2


load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")

st.set_page_config(page_title="피크닉(Picnic)")
st.title('공강 시간에 어딜 갈까?')

conn = psycopg2.connect(
    dbname=os.getenv("POSTGRES_DB"),
    user=os.getenv("POSTGRES_USER"),
    password=os.getenv("POSTGRES_PASSWORD"),
    host=os.getenv("POSTGRES_HOST"),
    port=os.getenv("POSTGRES_PORT")
)
register_vector(conn)


embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# HyDE Prompt Template
template_hyde = """Please describe a cafe that would suit the users need in korean
User's Need: {query}

##Example:
User's Need: 배고파ㅠㅠ
Response: 브런치를 잘하는, 빵이 맛있는, 배 채우기 좋은, 양이 많은, 밥 대신 먹기 좋은

User's Need: 혼자 있기 좋은 카페
Response: 공부하기 좋은, 아늑한, 조용한, 혼공,

User's Need: 미팅하기 좋은 카페
Response: 넓은, 쾌적한, 저렴한, 자리가 많은, 깨끗한, 대화하기 좋은,
"""

prompt_hyde = ChatPromptTemplate.from_template(template_hyde)


template_rag = """Explain why the below cafe will be perfect for the users needs, in korean. Make it concise, but in a friendly manner. explain based on the context, and don't make things up. :

{context}

User's needs: {question}
"""
prompt_rag = ChatPromptTemplate.from_template(template_rag)


# def extract_mood_from_query(query):
#     mood_keywords = {
#         "우울": ["우울한"],
#         "행복": ["행복한", "기쁜"],
#         "기분 좋": ["기분 좋은", "즐거운"],
#         "슬픔": ["슬픈", "울적한"],
#         "편안": ["편안한"],
#         "신남": ["신나는", "흥분된"],
#         "힘들": ["지친", "힘든", "우울한", "피곤한"]
#     }

#     for keyword, moods in mood_keywords.items():
#         if keyword in query:
#             return random.choice(moods)

#     return "일반적인" 

def get_top3_similar_docs_sorted_by_ratings(query_embedding, conn):
    embedding_array = np.array(query_embedding)
    cur = conn.cursor()
    
    cur.execute("""
        SELECT content, metadata 
        FROM store_vectors
        ORDER BY embedding <=> %s 
        LIMIT 3
    """, (embedding_array,))
    
    docs = cur.fetchall()
    
    parsed_docs = [
        {
            "content": doc[0],
            "metadata": doc[1]  
        }
        for doc in docs
    ]
    print(parsed_docs)
    sorted_docs = sorted(
        parsed_docs, 
        key=lambda x: float(x["metadata"].get("Ratings", 0)),  # Default to 0 if no Ratings
        reverse=True
    )
    print(sorted_docs)
    
    return sorted_docs



def generate_response_with_hyde(query):
    # mood = extract_mood_from_query(query)
    hyde_prompt = prompt_hyde.invoke({"query":query})
    hyde_passage = llm(hyde_prompt.to_messages())
    query_embedding = embedding_model.embed_query(hyde_passage.content.split("Response:")[1].strip())

    matching_docs = get_top3_similar_docs_sorted_by_ratings(query_embedding, conn)

    context = "\n".join(
    f"Store Name: {doc['metadata']['Store Name']}\nContent: {doc['content']}"
    for doc in matching_docs
)
    rag_prompt = prompt_rag.invoke({"context":context, "question":query})
    final_answer = llm(rag_prompt.to_messages()).content


    return final_answer


with st.form('Question'):
    query = st.text_area('원하는 활동 또는 기분을 입력하세요:', "힘들어서 힐링할 수 있는 카페 추천해줘")
    submitted = st.form_submit_button('추천 받기')
    if submitted:
        answer = generate_response_with_hyde(query)
        st.write("추천 장소:", answer)
