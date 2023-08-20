import os
import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pinecone
import PyPDF2
from dotenv import load_dotenv


load_dotenv()


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
        print(pinecone_vector)
        pinecone_vectors.append(pinecone_vector)

    print('finished embeddings starting upsert')
    # All vectors can be upserted to Pinecone in one go
    upsert_response = index.upsert(vectors=pinecone_vectors)
    print("upsert finished, Length of pinecone vectors", len(pinecone_vectors))


if __name__ == '__main__':
    data_dir = r"/Users/nickrystynak/Desktop/AI project/Investor Letters/InvestorAI/Allie Docs"
    api_key = os.getenv("PINECONE_API_KEY")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    print(openai.api_key)
    print("loading docs")
    documents = load_data(data_dir)
    chunks = chunk_data(documents)
    embed_data(chunks)
    embeddings = OpenAIEmbeddings(openai_api_key = openai.api_key)
    print("Finished!")
  