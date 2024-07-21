import gradio as gr

from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores import FAISS


def initialize_sales_bot(vector_store_dir: str="car_sales_qa"):
    db = FAISS.load_local(vector_store_dir, OpenAIEmbeddings(base_url="https://api.xiaoai.plus/v1", api_key="sk-9o1als9Y9It7G1YoF38936F9061b4fE9B7C082Ec4cA83215"),allow_dangerous_deserialization = True)
    llm = ChatOpenAI(base_url="https://api.xiaoai.plus/v1", api_key="sk-9o1als9Y9It7G1YoF38936F9061b4fE9B7C082Ec4cA83215", model_name="gpt-3.5-turbo", temperature=0)
    
    global SALES_BOT    
    SALES_BOT = RetrievalQA.from_chain_type(llm,
                                           retriever=db.as_retriever(search_type="similarity_score_threshold",
                                                                     search_kwargs={"score_threshold": 0.8}))
    # 返回向量数据库的检索结果
    SALES_BOT.return_source_documents = True

    return SALES_BOT

def sales_chat(message, history):
    print(f"[message]{message}")
    print(f"[history]{history}")
    enable_chat = True

    ans = SALES_BOT({"query": message})
    # 如果检索出结果，或者开了大模型聊天模式
    # 返回 RetrievalQA combine_documents_chain 整合的结果
    if ans["source_documents"] or enable_chat:
        print(f"[result]{ans['result']}")
        print(f"[source_documents]{ans['source_documents']}")
        return ans["result"]
    # 否则输出套路话术
    if len(ans["source_documents"]) == 0:
        template = """
        这是客户的历史对话记录: {history},
        现在顾客的最新问题是: {question},
        请根据上述顾客咨询问答，像一个人类一样，以专业资深的汽车销售顾问的身份，向客户给出一个更加流畅，自然，专业的回答。你不能回答汽车销售以外的问题。
        """
        llm = ChatOpenAI(model_name="gpt-4", temperature=0,base_url="https://api.xiaoai.plus/v1", api_key="sk-9o1als9Y9It7G1YoF38936F9061b4fE9B7C082Ec4cA83215")
        prompt = PromptTemplate(template=template, input_variables=["history", "question"])
        chain = LLMChain(llm=llm, prompt=prompt)
        return chain.run(history=history, question=message)
    else:
        return "您这个问题难住我了，我得问下我们的经理，稍后再给您一个满意的回答，请稍等"
    

def launch_gradio():
    demo = gr.ChatInterface(
        fn=sales_chat,
        title="汽车销售",
        chatbot=gr.Chatbot(height=600),
    )

    demo.launch(share=True, server_name="0.0.0.0")

if __name__ == "__main__":
    # 初始化
    initialize_sales_bot()
    # 启动 Gradio 服务
    launch_gradio()
