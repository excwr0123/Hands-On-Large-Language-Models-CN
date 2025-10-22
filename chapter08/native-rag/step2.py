from langchain_chroma import Chroma                      # 负责从本地向量数据库中检索相似文本
from langchain_openai import OpenAIEmbeddings, ChatOpenAI  #把文字转成向量，用来语义检索
from langchain_core.prompts import PromptTemplate        
from langchain_core.messages import HumanMessage          

# 1. 加载已有的向量数据库
embeddings = OpenAIEmbeddings(
    base_url="https://api.siliconflow.cn/v1",
    model="Qwen/Qwen3-Embedding-0.6B",
) #指定用 Qwen3 Embedding 模型 来生成向量

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)   #指向offline生成的数据库 -> vetorstore对象知道知识存在哪里了

# 2. 用户提问
query = "what is RAG？"

# 3. 检索相关文档（返回最相关的 3 个）
docs = vectorstore.similarity_search(query, k=3) # query通过同样的embedding转成向量

# 4. 拼接检索到的文档内容
context = "\n\n".join([doc.page_content for doc in docs]) #做为LLM的参考资料

# 5. 构建 Prompt 模板
prompt_template = """
Answer the user's question based on the following reference document. If there is no relevant information in the reference document, please honestly say you don't know. 
Reference document：{context}
User question：{question}
answer：
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"],
) # 防止LLM产生hallucination

# 6. 创建大语言模型（SiliconFlow 平台）
llm = ChatOpenAI(
    model="THUDM/glm-4-9b-chat",
    temperature=0,
    max_retries=3,
    base_url="https://api.siliconflow.cn/v1",
)

# 7. 生成最终 prompt 并发送请求
final_prompt = prompt.format(context=context, question=query) #把拼接好的prompt发给模型
print(f"final Prompt：{final_prompt}")

messages = [HumanMessage(content=final_prompt)]
response = llm.invoke(messages)

# 8. 输出结果
print(f"question: {query}")
print(f"answer: {response.content}")
print(f"\nNumber of reference documents: {len(docs)}")
