from sqlalchemy import create_engine, MetaData, Table, select
from sqlalchemy.orm import sessionmaker
import openai
from dotenv import load_dotenv
import os
import pinecone

# Load environment variables from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")  # Corrected here

# Get Pinecone API key from environment variables
pinecone_api_key = os.getenv("PINECONE_API_KEY")


def load_data_from_db():
    # Create an engine that connects to the sqlite database
    engine = create_engine('sqlite:////Users/nickrystynak/Desktop/AI project/Investor Letters/InvestorAI/DBtestAllieFiles.db')

    # Create a metadata instance
    metadata = MetaData()

    # Reflect the tables
    metadata.reflect(bind=engine)

    # Get the table
    responses_table = metadata.tables['responses']

    # Create a session
    Session = sessionmaker(bind=engine)
    session = Session()

    # Query the table
    result = session.query(responses_table).all()

    # Close the session
    session.close()

    # Print the first few records to understand the data structure
    print("First record:")
    for record in result[:1]:
        print(record)

    print(f"Loaded {len(result)} records from the database.")
    return result

def create_nodes_and_vector_store():
    # Load data from the database
    print("loading data..")
    data = load_data_from_db()
    print("loaded data from db")

    print(f"Creating nodes and metadata for {len(data)} records.")  # Debugging statement

    # Initialize Pinecone
    pinecone.init(api_key=pinecone_api_key, environment="us-west1-gcp-free")
    index = pinecone.Index('letters004')
    print(index, "This is the inde name")

    # Create a list of nodes and metadata
    nodes = []
    metadata_list = []
    for record in data:
        if record[2] is not None and record[3] is not None:
            node = f"{record[2]} {record[3]}"  # question is at index 2 and answer is at index 3
            metadata = {"source": record[1]}  # filename is at index 1
            nodes.append(node)
            metadata_list.append(metadata)
            print(f"Created node and metadata for record {record[0]}.")

    # Create a list to hold the Pinecone vectors
    pinecone_vectors = []

    # Create a new vector store index using the nodes and metadata
    for i, (node, metadata) in enumerate(zip(nodes, metadata_list)):
        embedding = openai.Embedding.create(
            input=node,
            model="text-embedding-ada-002"
        )
        vector = embedding["data"][0]["embedding"]
        print(f"Created Pinecone vector for node {i}.")
        
        # Update the metadata to include the "text" key
        metadata["text"] = node
        
        # Create a Pinecone vector with the embedding and updated metadata
        pinecone_vector = (str(i), vector, metadata)
        pinecone_vectors.append(pinecone_vector)

    print(f"Created {len(pinecone_vectors)} Pinecone vectors.")

    print('finished embeddings starting upsert')
    # All vectors can be upserted to Pinecone in one go
    upsert_response = index.upsert(vectors=pinecone_vectors)
    print("upsert finished, Length of pinecone vectors", len(pinecone_vectors))

def main():
    create_nodes_and_vector_store()

if __name__ == "__main__":
    main()