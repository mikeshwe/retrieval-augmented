from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI

from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain

from langchain.chat_models import ChatOpenAI

from dotenv import load_dotenv, find_dotenv
import openai
import os
import json
from tqdm import tqdm


# Load JSON file
def load_json_file(file_path):
    with open(file_path) as file:
        data = json.load(file)
    return data

class My_Document:
  def __init__(self,summary, title):
    self.page_content = summary
    self.metadata = {"title":title}


# Extract summary and title fields from JSON documents
def extract_fields(data):
    documents = []
    for item in data:
        if "summary" in item and "title" in item:
            summary = item["summary"]
            title = item["title"]
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

def test(db,query):
    similar_docs = find_similiar_docs(db,query,k=3,score=True)

    #print the top 2 results
    i = 0 
    for doc in similar_docs[:3]:
        print(i,": ",doc,"\n")
        i = i+1

def find_similiar_docs(db,query, k=3, score=True):
    if score:
        similar_docs = db.similarity_search_with_score(query, k=k)
    else:
        similar_docs = db.similarity_search(query, k=k)
    return similar_docs

def get_answer(db,query,chain):
    similar_docs = find_similiar_docs(db,query, k=2, score=False)
    answer = chain.run(input_documents=similar_docs, question=query)
    return answer

def qa(db,query):
    model = "gpt-3.5-turbo"
    llm = ChatOpenAI(model_name=model)
    chain = load_qa_chain(llm, chain_type="stuff")
    answer = get_answer(db,query,chain)
    print(query)
    print(answer)


# Main function
def main():
    _ = load_dotenv(find_dotenv()) # read the local .env file
    openai.api_key  = os.getenv('OPENAI_API_KEY')
    # Replace with your relative file path to a json file of articles with summary and title fields
    file_path = "docs/Conversation.json"  
    database_name = "my_database"  # Replace with your desired Chroma database name

    data = load_json_file(file_path)
    documents = extract_fields(data)
    db = load_into_chroma(documents, database_name)
    #query = "How should you add an ask during a job review?"
    query = "How should you ask for professional advice at work?"
    test(db,query)
    qa(db,query)

if __name__ == "__main__":
    main()





