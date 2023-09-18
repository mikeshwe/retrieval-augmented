from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import LLMChain

from langchain.chat_models import ChatOpenAI

from dotenv import load_dotenv, find_dotenv
import openai
import os
import json

MODEL = "gpt-4"
TEMPERATURE = 0.5

# Load JSON file from local storage
def load_json_file(file_path):
    with open(file_path) as file:
        data = json.load(file)
    return data

# This class holds a document in a form that the Chroma database likes. Namely, there's page_content with the 
# actual payload you want to index in the database, and metadata--here, just a title
class My_Document:
  def __init__(self,summary, title):
    self.page_content = summary
    self.metadata = {"title":title}

# Extract summary and title fields from JSON, and load into documents for vector database
def extract_fields(data):
    documents = []
    for item in data:
        if "summary" in item and "title" in item:
            summary = item["summary"]
            title = item["title"]
            # print ('title: '+title+'\n'+'summary: '+summary+'\n')
            document = My_Document(summary, title)
            documents.append(document)
    print ("number of documents: ", len(documents))
    return documents


# Load documents and embeddings into Chroma database
def load_into_chroma(documents, database_name):
    chroma = Chroma(database_name)
    embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')
    db = chroma.from_documents(documents, embeddings)
    return db

def show_similar_docs(db,query):
    similar_docs = find_similiar_docs(db,query,k=2,score=True)

    #print the top k results
    i = 0 
    for doc in similar_docs:
        print(i,": ",doc,"\n")
        i = i+1

# returns the k most-similar docs in the vector db to the query.
# For debugging purposes, this function can optionally return
# a score of how well the doc matched the query
def find_similiar_docs(db,query, k, score):
    if score:
        similar_docs = db.similarity_search_with_score(query, k=k)
    else:
        similar_docs = db.similarity_search(query, k=k)
    return similar_docs

# query the LLM reference by the QA chain, using the similar_docs as context
def get_answer(db,query,chain):
    similar_docs = find_similiar_docs(db,query, k=2, score=False)
    answer = chain.run(input_documents=similar_docs, question=query)
    return answer

# do a single Q&A interaction with the LLM, feeding it documents
# relevant to the query
def qa(db,query,model):
    llm = ChatOpenAI(model_name=model,temperature=TEMPERATURE)
    chain = load_qa_chain(llm, chain_type="stuff")
    answer = get_answer(db,query,chain)
    print("custom answer: \n"+ answer+"\n")

def qa_base(db,query,model):
    llm = ChatOpenAI(model_name=model,temperature=TEMPERATURE)
    prompt = PromptTemplate(
        input_variables=["product"], # bit of a hack for a null prompt
        template=query+"{product}",
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    print("generic answer: "+ "\n" + chain.run("")+"\n")

def compare_answers (db,query):
    print ("query: " + query + "\n")
    show_similar_docs(db,query)
    model = MODEL
    qa(db,query,model)
    qa_base(db,query,model)

# Main function
def main():
    _ = load_dotenv(find_dotenv()) # read the local .env file
    openai.api_key  = os.getenv('OPENAI_API_KEY')
    # Replace with your relative file path to a json file of articles with summary and title fields
    file_path = "docs/career_articles.json"  
    database_name = "my_database"  # Replace with your desired Chroma database name

    data = load_json_file(file_path)
    documents = extract_fields(data)
    
    db = load_into_chroma(documents, database_name)

    queries = ["How should I negotiate my salary and other benefits at work?"]
    i = 0
    for query in queries:
        i = i + 1
        print ("Question " + str(i) + "\n")
        compare_answers (db,query)

if __name__ == "__main__":
    main()





