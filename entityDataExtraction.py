#open AI
import os
from platform import node
from webbrowser import get
import openai
#llama imports 
from llama_index.node_parser.extractors.metadata_extractors import (
    MetadataExtractor,
    EntityExtractor,
)
from llama_index.node_parser.simple import SimpleNodeParser
from llama_index import SimpleDirectoryReader

#pdf loader
import PyPDF2
#random for loading 
import random

#query imports
from llama_index import ServiceContext, VectorStoreIndex
from llama_index.llms import OpenAI



#Set up vars
os.environ["OPENAI_API_KEY"] = "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"
data_dir = r"C:\Users\websu\Documents\LLM Projects\Investor Letters\Data\Letters\2022 Letters\Test Data"



#Set up extractor and parser
entity_extractor = EntityExtractor(
    prediction_threshold=0.5,
    label_entities=False,  # include the entity label in the metadata (can be erroneous)
    device="cpu",  # set to "cuda" if you have a GPU
)

#define extractor 
metadata_extractor = MetadataExtractor(extractors=[entity_extractor])
#define node parser
node_parser = SimpleNodeParser.from_defaults(metadata_extractor=metadata_extractor)

#load data
def load_data(data_dir):
    # List all PDF files in the data directory
    pdf_files = [os.path.join(data_dir, pdf_file) for pdf_file in os.listdir(data_dir) if pdf_file.endswith('.pdf')]

    # Load the PDF files using SimpleDirectoryReader
    documents = SimpleDirectoryReader(input_files=pdf_files).load_data()
                    
    return documents

#Run the node parser on our documents
def getNodes(documents):
    nodes = node_parser.get_nodes_from_documents(documents)

    samples = random.sample(nodes, 5)
    for node in samples:
        print(node.metadata)
    return nodes

def queryNodes(nodes, question):
    service_context = ServiceContext.from_defaults(
        llm=OpenAI(model="gpt-3.5-turbo", temperature=0.2)
    )
    
    index = VectorStoreIndex(nodes, service_context=service_context)

    query_engine = index.as_query_engine()
    response = query_engine.query("What is said by Fox-Kemper?")
    print(response)

if __name__ == '__main__':
    print("Loading documents")
    documents = load_data(data_dir)
    print("Getting nodes")
    nodes = getNodes(documents)
    queryNodes(nodes, "What does the investment landscape in russia look like?")
