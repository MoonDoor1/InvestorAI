#open AI
import os
from platform import node
from webbrowser import get
import openai
from dotenv import load_dotenv
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

#importing for saving the answers into the DB
from sqlalchemy import create_engine, Table, Column, Integer, String, MetaData, Text
from sqlalchemy.orm import sessionmaker






#Set up vars
load_dotenv()
data_dir = r"2022 Letters/Test Data"
openai.api_key = os.getenv("OPENAI_API_KEY")  # Corrected here

#questions = ["What does the author of this letter think the best investment strategy is moving forward and why?", "What is the author's opinion on the current market conditions?", "What opportunities and potential industries or sectors are mentioned in this letter for the next quarter?","What types of risks are anticipated in the upcoming quarter according to this letter and what is the proposed strategy for their mitigation?", "How does this letter describe the performance of previous quarter's investments, including any standout or underperforming investments, and the lessons learned from them?", "Can you extract advice from this letter for navigating the investment landscape?"]
questions = [
    "What is the author's opinion on the current market conditions?", 
    "What opportunities and potential industries or sectors are mentioned in this letter for the next quarter?",
    "How does this letter describe the performance of previous quarter's investments, including any standout or underperforming investments, and the lessons learned from them?"
             ]


#Set up extractor and parser
entity_extractor = EntityExtractor(
    prediction_threshold=0.5,
    label_entities=True,  # include the entity label in the metadata (can be erroneous)
    device="cpu",  # set to "cuda" if you have a GPU
)

#define extractor 
metadata_extractor = MetadataExtractor(extractors=[entity_extractor])
#define node parser
node_parser = SimpleNodeParser.from_defaults(metadata_extractor=metadata_extractor)

#load data
def load_data(data_dir):
    # Check if directory exists
    if not os.path.exists(data_dir):
        raise Exception(f"Directory {data_dir} does not exist")

    # List all PDF files in the data directory
    pdf_files = [os.path.join(data_dir, pdf_file) for pdf_file in os.listdir(data_dir) if pdf_file.endswith('.pdf')]

    # Load the PDF files using SimpleDirectoryReader
    documents = SimpleDirectoryReader(input_files=pdf_files).load_data()
    

    return documents


def get_nodes(documents, engine, organizations_table, persons_table, locations_table):
    nodes_by_document = {}

    nodes = node_parser.get_nodes_from_documents(documents)

    # Create a session
    Session = sessionmaker(bind=engine)
    session = Session()

    for node in nodes:
        file_name = node.metadata['file_name']

        if file_name not in nodes_by_document:
            nodes_by_document[file_name] = []

        nodes_by_document[file_name].append(node)

        # Save the organizations, persons, and locations to the database
        for organization in node.metadata.get('organizations', []):
            session.execute(organizations_table.insert().values(document_id=file_name, organization=organization))
        for person in node.metadata.get('persons', []):
            session.execute(persons_table.insert().values(document_id=file_name, person=person))
        for location in node.metadata.get('locations', []):
            session.execute(locations_table.insert().values(document_id=file_name, location=location))

    # Commit the changes and close the session
    session.commit()
    session.close()

    return nodes_by_document



def query_document(nodes, question):
    service_context = ServiceContext.from_defaults(
        llm=OpenAI(model="gpt-3.5-turbo", temperature=0.2)
    )
    
    index = VectorStoreIndex(nodes, service_context=service_context)

    query_engine = index.as_query_engine()
    response = query_engine.query(question)

    # Convert the response to a string
    response_str = str(response)

    return response_str

#function for setting up the database 
def setup_database():
    # Create an engine that stores data in the local directory's
    # sqlalchemy_example.db file.
    engine = create_engine('sqlite:////newDBtest.db')

    # Create a metadata instance
    metadata = MetaData()

    # Declare a table
    responses_table = Table('responses', metadata,
              Column('id', Integer, primary_key=True),
              Column('document_id', Text, index=True),
              Column('question', Text, index=True),
              Column('response', Text)
              )
    
    # Declare tables for organizations, persons, and locations
    organizations_table = Table('organizations', metadata,
              Column('id', Integer, primary_key=True),
              Column('document_id', Text, index=True),
              Column('organization', Text)
              )

    persons_table = Table('persons', metadata,
              Column('id', Integer, primary_key=True),
              Column('document_id', Text, index=True),
              Column('person', Text)
              )

    locations_table = Table('locations', metadata,
              Column('id', Integer, primary_key=True),
              Column('document_id', Text, index=True),
              Column('location', Text)
              )

    # Create all tables
    metadata.create_all(engine)

    # Print table names
    print("Tables in the database: ", metadata.tables.keys())

    return engine, responses_table, organizations_table, persons_table, locations_table

def clear_tables(engine, *tables):
    with engine.connect() as connection:
        for table in tables:
            connection.execute(table.delete())
    print("Cleared tables")

def view_database(engine, table):
    print("view db function called")
    with engine.connect() as connection:
        print("made connection to db...")
        result = connection.execute(table.select())
        rows = result.fetchall()
        if rows:
            for row in rows:
                print(f"Record: {row}")
        else:
            print("No records found in the table.")

def test_database_connection(engine, table):
    try:
        # Create a session
        Session = sessionmaker(bind=engine)
        session = Session()

        # Insert a test record
        session.execute(table.insert().values(document_id="test", question="test", response="test"))
        session.commit()
        print("Inserted test record")
        
        # Retrieve the test record
        result = session.execute(table.select().where(table.c.document_id == "test"))
        for row in result:
            print(f"Retrieved test record: {row}")

        session.close()
    except Exception as e:
        print(f"An error occurred: {e}")

def clear_test_table(engine, table):
    try:
        # Create a session
        Session = sessionmaker(bind=engine)
        session = Session()

        # Delete the test record
        session.execute(table.delete().where(table.c.document_id == "test"))
        session.commit()
        print("Cleared test records")

        session.close()
    except Exception as e:
        print(f"An error occurred: {e}")
    

if __name__ == '__main__':
    # Setup the database
    print("Setting up database...")
    engine, responses_table, organizations_table, persons_table, locations_table = setup_database()
    
    print("Loading documents..")
    documents = load_data(data_dir)
    
    print("Getting nodes..")
    nodes_by_document = get_nodes(documents, engine, organizations_table, persons_table, locations_table)
    print(f"Got nodes for {len(nodes_by_document)} documents")

    # Create a session
    Session = sessionmaker(bind=engine)
    session = Session()

    for file_name, nodes in nodes_by_document.items():
        for question in questions:
            print(f"Querying document {file_name} with question '{question}'")
            response = query_document(nodes, question)
            print(f"Got response '{response}' for document {file_name}")

            # Save the response in the database
            try:
                session.execute(responses_table.insert().values(document_id=file_name, question=question, response=response))
                print(f"Saved response for document {file_name} in the database")

                # Retrieve and print the inserted data
                result = session.execute(responses_table.select().where(responses_table.c.document_id == file_name))
                for row in result:
                    print(f"Retrieved data from database: {row}")

            except Exception as e:
                print(f"An error occurred while inserting data into the database: {e}")

    # Commit the changes and close the session
    session.commit()
    session.close()

    # View the responses table
    print("Viewing responses table...")
    view_database(engine, responses_table)