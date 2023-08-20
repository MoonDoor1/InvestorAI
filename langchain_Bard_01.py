import os
from pydoc import doc
from traceback import print_tb
from xml.dom.minidom import Document
import openai
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pinecone
import PyPDF2

#imports for langchain retreiver
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.vectorstores import Pinecone 

#imports for the conversation retrevial chain 
from langchain.memory import ConversationTokenBufferMemory
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
import chainlit as cl
import pinecone
from chainlit import user_session
#from prompts import load_query_gen_prompt, load_spark_prompt
from chainlit import on_message, on_chat_start





#Init the pinecone index
pinecone.init(      
	api_key='8062bda0-956d-471c-8498-0b91c7997b98',      
	environment='us-west1-gcp-free'      
)

index_name = 'letters003'
index = pinecone.Index(index_name)



def load_data(data_dir):
    documents = []
    for pdf_file in os.listdir(data_dir):
        if pdf_file.endswith('.pdf'):
            with open(os.path.join(data_dir, pdf_file), "rb") as f:
                pdf = PyPDF2.PdfReader(f)
                text = ''
                for i in range(len(pdf.pages)):
                    page = pdf.pages[i]
                    text += page.extract_text()

                metadata = {
                    "filename": pdf_file
                }
                
                documents.append((text, metadata))  # Append as a tuple

    return documents


def chunk_data(documents):
    chunks = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )

    for text, metadata in documents:
        print(metadata)
        # Using create_documents on a single text, so passing [text] as an argument
        chunked_texts = text_splitter.create_documents([text], [metadata])
        for chunk_text in chunked_texts:
            chunks.append(chunk_text)

    print(chunks)
    return chunks



  

def embed_data(chunks):
    """Embeds data using a language model.

    Args:
        chunks: A list of Document objects, where each document contains page_content and metadata.
    """
    pinecone_vectors = []
    print("starting embeddings")

    for i, chunk in enumerate(chunks):
        doc = chunk.page_content  # Extracting the text content from the Document object
        filename = chunk.metadata['filename']  # Extracting the filename from the metadata
        
        embedding = openai.Embedding.create(
            input=doc,
            model="text-embedding-ada-002"
        )
        vector = embedding["data"][0]["embedding"]
        
        # Create a Pinecone vector with the embedding and additional metadata
        pinecone_vector = (str(i), vector, {"text": doc, "source": filename})
        pinecone_vectors.append(pinecone_vector)

    print('finished embeddings starting upsert')
    # All vectors can be upserted to Pinecone in one go
    upsert_response = index.upsert(vectors=pinecone_vectors)
    print("upsert finished, Length of pinecone vectors", len(pinecone_vectors))

def retrieve(query):
    limit = 3750

    #setting up for querying the vector db
    embed_model = "text-embedding-ada-002"

    res = openai.Embedding.create(
        input=[query],
        engine=embed_model
    )

    # retrieve from Pinecone
    xq = res['data'][0]['embedding']

    # get relevant contexts
    res = index.query(xq, top_k=3, include_metadata=True)
    print("Retrieved documents:", res)  # Debugging statement

    contexts = [
        x['metadata']['text'] for x in res['matches']
    ]

    # Print out the metadata for each retrieved document
    for i, match in enumerate(res['matches']):
        print(f"Document {i+1} metadata:", match['metadata'])  # Debugging statement

    # build our prompt with the retrieved contexts included
    prompt_start = (
        "Answer the question based on the context below.\n\n"+
        "Context:\n"
    )
    prompt_end = (
        f"\n\nQuestion: {query}\nAnswer:"
    )
    # append contexts until hitting limit
    for i in range(1, len(contexts)):
        if len("\n\n---\n\n".join(contexts[:i])) >= limit:
            prompt = (
                prompt_start +
                "\n\n---\n\n".join(contexts[:i-1]) +
                prompt_end
            )
            break
        elif i == len(contexts)-1:
            prompt = (
                prompt_start +
                "\n\n---\n\n".join(contexts) +
                prompt_end
            )
    return prompt
def complete(prompt):
    res = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=400,
    )
    return res['choices'][0]['message']['content'].strip()


def buildRetreiver(embeddings):
    print("Loading index..")
    # load index
    docsearch = Pinecone.from_existing_index(index_name, embeddings)
    # initialize base retriever
    print("initialize retriever..")
    retriever = docsearch.as_retriever(search_kwargs={"k": 4})
    
    # Set up cohere's reranker
    print("setting up conherence ranker..")
    compressor = CohereRerank()
    reranker = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )

    # initialize llm
    llm = ChatOpenAI(temperature=0.7, verbose=True, openai_api_key = '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', streaming=True)

    print("Building conversation chain...")
    memory = ConversationTokenBufferMemory(llm=llm, memory_key="chat_history", return_messages=True, input_key='question', max_token_limit=1000)
    question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT, verbose=True)
    answer_chain = load_qa_with_sources_chain(llm, chain_type="stuff", verbose=True,prompt=chat_prompt)

    chain = ConversationalRetrievalChain(
                retriever=reranker,
                question_generator=question_generator,
                combine_docs_chain=answer_chain,
                verbose=True,
                memory=memory,
                rephrase_question=False
    )

    print("Propmting the model...")
    query = "Was the 60/40 portfolio a viable strategy for 2022? explain in detail"
    result = chain({"question": query})
    print(result)



from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# Chat roles
from langchain.prompts.prompt import PromptTemplate


_template = """Below is a summary of the conversation so far, and a new question asked by the user that needs to be answered by searching in a knowledge base. Generate a search query based on the conversation and the new question.

Chat History:
{chat_history}

Question:
{question}

Search query:
"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

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



system_message_prompt = SystemMessagePromptTemplate.from_template(investorAI)
# instruction = HumanMessagePromptTemplate.from_template(prompt_template)
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt])

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank

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

# if __name__ == '__main__':
#     data_dir = r"C:\Users\websu\Documents\LLM Projects\Investor Letters\Data\Letters\2022 Letters\Test Data"
#     api_key = '8062bda0-956d-471c-8498-0b91c7997b98'
#     openai_key = ''
#     documents = load_data(data_dir)
#     chunks = chunk_data(documents)
#     embed_data(chunks)
#     embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
#     buildRetreiver(embeddings)

    
    
    #embeddings = OpenAIEmbeddings()
    #vectorstore = Pinecone.from_documents(chunks, embeddings, index_name="letters003")
    # query = ("What are key take aways from these investment letters?")
    # query_with_contexts = retrieve(query)
    # print(query_with_contexts)
    # print(complete(query_with_contexts))
  

    