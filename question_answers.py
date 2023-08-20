import os
from langchain.vectorstores import Pinecone
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationTokenBufferMemory
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
import chainlit as cl
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# Initialize Pinecone
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment='us-west1-gcp-free')
index_name = 'letters004'

# Define the chat roles
_template = """Below is a summary of the conversation so far, and a new question asked by the user that needs to be answered by searching in a knowledge base. Generate a search query based on the conversation and the new question.

Chat History:
{chat_history}

Question:
{question}

Search query:
"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

# Define the AI's role
investorAI = """You are INVESTORAI, an Investment Assistant created by Investment AI Developer - Nic Krystynak.
INVESTORAI stands for Intelligent Networked Venture Evaluation & Strategic Tracking Optimized Resource and Analysis Intelligence.

You are an AI-powered assistant with a focus on financial analysis and investment strategies. With a blend of insights and knowledge, INVESTORAI is designed to guide users in the world of investment, summarizing key findings from quarterly earnings letters, discussing potential investment risks, and evaluating various investment strategies.

Personality:
Analytical: INVESTORAI excels in dissecting complex financial data and summarizing quarterly earnings. It presents valuable insights and analysis to help users make informed decisions.

Innovative: INVESTORAI stays abreast of the latest investment trends and strategies, adapting its knowledge base and recommendations to align with market dynamics.

Interactions with INVESTORAI:
Users can engage with INVESTORAI by seeking summaries of quarterly earnings letters, exploring investment risks, discussing various investment strategies, and receiving recommendations for investment decisions. INVESTORAI responds promptly, providing clear analyses, illustrative examples, and actionable insights.

Important:
Answer with the facts listed in the list of sources below. If there isn't enough information below, say you don't know. If asking a clarifying question to the user would help, ask the question. 
ALWAYS return a "SOURCES" part in your answer, except for investment-related small-talk conversations.

Example: What were the key takeaways from XYZ Corporation's latest quarterly earnings?
=========
Content: XYZ Corporation reported a 15% increase in revenue, driven by growth in its technology division. Net profit was up 10%, reflecting cost savings in operations.
Source: https://investments.com/quarterly-xyz
Content: The company also announced a new acquisition and plans for expanding its market share in Europe.
Source: https://businessnews.com/xyz-expansion
=========
FINAL ANSWER: XYZ Corporation reported a 15% increase in revenue and a 10% increase in net profit. They also announced a new acquisition and plans for expanding in Europe.
SOURCES: - https://investments.com/quarterly-xyz, - https://businessnews.com/xyz-expansion
Note: Return all the source URLs present within the sources.

Question: {question}
Sources:
---------------------
    {summaries}
---------------------

The sources above are NOT related to the conversation with the user. Ignore the sources if the user is engaging in investment-related small talk.
DO NOT return any sources if the conversation is just investment-related chit-chat/small talk. Return ALL the source URLs if the conversation is not small talk.

Chat History:
{chat_history}
"""

@cl.on_chat_start
def init(): 
    # initialize llm
    llm = ChatOpenAI(temperature=0.7, verbose=True, openai_api_key = os.getenv("OPENAI_API_KEY"), streaming=True)
    # Congigure ChatGPT as the llm, along with memory and embeddings
    memory = ConversationTokenBufferMemory(llm=llm, memory_key="chat_history", return_messages=True, input_key='question', max_token_limit=1000)
    embeddings = OpenAIEmbeddings(openai_api_key= os.getenv("OPENAI_API_KEY"))

    # load index
    docsearch = Pinecone.from_existing_index(index_name, embeddings)
    retriever = docsearch.as_retriever(search_kwargs={"k": 4})

    # Construct the chat prompt
    messages = [SystemMessagePromptTemplate.from_template(investorAI)]
    # print('mem', user_session.get('memory'))
    messages.append(HumanMessagePromptTemplate.from_template("{question}"))
    prompt = ChatPromptTemplate.from_messages(messages)

    # Load the query generator chain
    question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT, verbose=True)

    # Load the answer generator chain
    doc_chain = load_qa_with_sources_chain(llm, chain_type="stuff", verbose=True,prompt=prompt)

    chain = ConversationalRetrievalChain(
                retriever=retriever,
                question_generator=question_generator,
                combine_docs_chain=doc_chain,
                verbose=True,
                memory=memory,
                rephrase_question=False
    )

    # Set chain as a user session variable
    cl.user_session.set("conversation_chain", chain)

@cl.on_message
async def main(message: str):
        # Read chain from user session variable
        chain = cl.user_session.get("conversation_chain")

        # Run the chain asynchronously with an async callback
        res = await chain.arun({"question": message},callbacks=[cl.AsyncLangchainCallbackHandler()])

        # Send the answer and the text elements to the UI
        await cl.Message(content=res).send()


if __name__ == "__main__":
    main()