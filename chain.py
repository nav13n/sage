# Langchain imports
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from dotenv import load_dotenv


load_dotenv()

######################## Build RAG Chain #############################
######################################################################

#### Load  Documents
loader = PyMuPDFLoader(
    "https://d18rn0p25nwr6d.cloudfront.net/CIK-0001326801/c7318154-f6ae-4866-89fa-f0c589f2ee3d.pdf"
)

documents = loader.load()

#### Split Documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 800,
    chunk_overlap = 100
)

documents = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

###  Create Vector Store
vector_store = Qdrant.from_documents(
    documents,
    embeddings,
    location=":memory:",
    collection_name="Meta 10k Filings",
)

### Create Prmopt Template
template = """Answer the question based only on the following context. If you cannot answer the question with the context, please respond with 'I don't know':

Context:
{context}

Question:
{question}
"""
prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

### Setup RAG Chain

retriever = vector_store.as_retriever(search_type="similarity_score_threshold", 
                                      search_kwargs={"score_threshold": 0.6, "k":8})
primary_qa_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt 
    | primary_qa_llm
    | StrOutputParser()

)
