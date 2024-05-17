
import os
import time
from dotenv import load_dotenv
from operator import itemgetter
from typing_extensions import TypedDict
from typing import List

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser

from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.schema import Document
from langgraph.graph import END, StateGraph

from groq import Groq
from langchain_groq import ChatGroq

from utils import get_payroll_api_schema, dummy_payroll_api_call

load_dotenv()

# Setup the models
embed_model = FastEmbedEmbeddings(model_name="snowflake/snowflake-arctic-embed-m")



llm = ChatGroq(temperature=0,
                      model_name="Llama3-8b-8192",
                      api_key=os.getenv("GROQ_API_KEY"),)



# Load the documents
loader = PyMuPDFLoader("https://home.synise.com/HRUtility/Documents/HRA/UmaP/Synise%20Handbook.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500, chunk_overlap=200
)
doc_splits = text_splitter.split_documents(documents)

vectorstore = FAISS.from_documents(documents=doc_splits,embedding=embed_model)

# Setup the retriever
compressor = FlashrankRerank()
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 20})
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)


# Define RAG Chain
RAG_PROMPT_TEMPLATE = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

 Answer the question based only on the provided context. If you cannot answer the question with the provided context, please respond with 'I don't know" without any preamble, explanation, or additional text.

Context:
{context}

Question:
{question}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_prompt = PromptTemplate(
    template=RAG_PROMPT_TEMPLATE, input_variables=["question", "context"]
)

response_chain = (rag_prompt
    | llm
    | StrOutputParser()

)

# Setup Router Chain
ROUTER_AGENT_PROMPT_TEMPLATE = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert at delegating user questions to one of the most appropriate agents 'raqa' or 'payroll'.

Use the following criteria to determine the appropriate agents to answer the user que:

- If the query is regarding payslips, salary, tax deductions, basepay of a given month, use 'payroll'.
- If the question is closely related to general human resource queries, organisational policies, prompt engineering, or adversarial attacks, even if the keywords are not explicitly mentioned, use the 'raqa'.

Your output should be a JSON object with a single key 'agent' and a value of either 'raqa' or 'payroll'. Do not include any preamble, explanation, or additional text.

User's Question: {question}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

router_prompt = PromptTemplate(
    template=ROUTER_AGENT_PROMPT_TEMPLATE, input_variables=["question"]
)


router_chain = router_prompt | llm | JsonOutputParser()

payroll_schema = get_payroll_api_schema()


# Define Filter Extraction Chain
FILTER_EXTTRACTION_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Extract the month and year from a given user question about payroll. Use the following schema instructions to guide your extraction.

Instructions:
1. Your output should be a JSON object with only two keys, 'month' and 'year'.
2. 'month' key shall have value ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
3. 'year' shall be a number between 2020 and 2024.
4. If the user is suggesting current year or month, respond with "CUR" for 'month' and 'year' keys accordingly
5. If the user is suggesting previous year or month, respond with "PREV" for 'month' and 'year' keys accordingly


Do not include any preamble, explanation, or additional text.

User Question: {question}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

filter_extraction_prompt = PromptTemplate(
    template=FILTER_EXTTRACTION_PROMPT, input_variables=["question"]
)

fiter_extraction_chain = filter_extraction_prompt | llm | JsonOutputParser()


# Define Payroll QA Chain

PAYROLL_QA_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Answer the user query given the provided payroll data in json form. Use the  provided schema to understand the payroll data structure. If you cannot answer the question with the provided information, please respond with 'I don't know" without any preamble, explanation, or additional text

SCHEMA:
{schema}

PAYROLL DATA
{data}

PAYROLL DATA:
{data}

User Question: {question}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

payroll_qa_prompt = PromptTemplate(
    template=PAYROLL_QA_PROMPT, input_variables=["question", "data", "schema"]
)

########### Create Nodes Actions ###########

class AgentState(TypedDict):
    question : str
    answer : str
    documents : List[str]

def route_question(state):
    """
    Route question to payroll_agent or policy_agent to retrieve reevant data

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """
    print("---ROUTING---")
    question = state["question"]
    result = router_chain.invoke({"question": question})

    return result["agent"]

state = AgentState(question="What is my salary on jan 2024 ?", answer="", documents=None)
route_question(state)


def retrieve(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE DOCUMENTS---")
    question = state["question"]
    documents = compression_retriever.invoke(question)
    return {"documents": documents, "question": question}

# state = AgentState(question="What is leave policy?", answer="", documents=None)
# retrieve_policy(state)

def generate(state):
    """
    Generate answer using retrieved data

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE ANSWER---")
    question = state["question"]
    documents = state["documents"]


    answer = response_chain.invoke({"context": documents, "question": question})

    return {"documents": documents, "question": question, "answer": answer}

def payroll(state):
    """
    Query payroll api to retrieve payroll data

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updated state with retrived payroll data
    """

    print("---QUERY PAYROLL API---")
    question = state["question"]
    payroll_query_filters = fiter_extraction_chain.invoke({"question":question})
    payroll_api_query_results = dummy_payroll_api_call(1234, payroll_query_filters["month"], payroll_query_filters["year"])


    context = context = 'PAYROLL DATA SCHEMA: \n {payroll_schema} \n PAYROLL DATA: {payroll_api_query_results}'.format(
    payroll_schema=payroll_schema, payroll_api_query_results=payroll_api_query_results)

    documents = [Document(page_content=context)]
    return {"documents": documents, "question": question}

########### Build Execution Graph ###########
workflow = StateGraph(AgentState)

# Define the nodes
workflow.add_node("payroll", payroll)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

workflow.set_conditional_entry_point(
    route_question,
    {
        "payroll": "payroll",
        "raqa": "retrieve",
    },
)
workflow.add_edge("payroll", "generate")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()
