import streamlit as st
from langchain.chains import LLMMathChain,LLMChain
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from dotenv import load_dotenv
from langchain.callbacks import StreamlitCallbackHandler
import os
load_dotenv()
st.set_page_config(page_title="Text to Math Problem Solver and Data Search Assistant")

st.title("Text to Math Problem Solver using Google Gemma2")

groq_api_key=os.environ["GROQ_API_KEY"]

llm=ChatGroq(model="Gemma2-9b-It",groq_api_key=groq_api_key)

wikipedia_wrapper=WikipediaAPIWrapper()
wikipedia_tool=Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="A tool for searching the internet and solving your math problem"
    
)


math_chain=LLMMathChain.from_llm(llm=llm)
calculator=Tool(
    name="Calculator",
    func=math_chain.run,
    description="A tool for answering math related questions. Only input mathematical expression need to be provided"
)

prompt="""
You are an agent tasked for solving user's mathematical questions. Logically arive at the solution and provide a detail solution and display it point wise for the question below
Question:{question}
Answer:
"""

prompt_template=PromptTemplate(
    input_variables=["question"],
    template=prompt
)

chain=LLMChain(llm=llm,prompt=prompt_template)

reasoning_tool=Tool(
    name="Reasoning Tool",
    func=chain.run,
    description="A tool for answering logic based and reasoning questions."
)

assistant_agent=initialize_agent(
    tools=[wikipedia_tool,calculator,reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assistant","content":"Hi, I'm a Math chatBot who answer all your maths questions"}
    ]
    

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
    


def generate_response(question):
    response=assistant_agent.invoke({'input':question})
    return response

question=st.text_area("Enter your math related question")

if st.button("Find my answer"):
    if question:
        with st.spinner("Generate response.."):
            st.session_state.messages.append({"role":"user","content":question})
            st.chat_message("user").write(question)
            
            st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
            response=assistant_agent.run(st.session_state.messages,callbacks=[st_cb])
            
            st.session_state.messages.append({"role":"assistant","content":response})
            
            st.write('### Response:')
            st.success(response)
    
    else:
        st.warning("Please enter the question")
            