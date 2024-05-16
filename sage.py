
from langchain_community.vectorstores import FAISS

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()

embed_model = FastEmbedEmbeddings(model_name="snowflake/snowflake-arctic-embed-m")

from groq import Groq
from langchain_groq import ChatGroq


llm = ChatGroq(temperature=0,
                      model_name="Llama3-8b-8192",
                      api_key=os.getenv("GROQ_API_KEY"),)

loader = PyMuPDFLoader("https://home.synise.com/HRUtility/Documents/HRA/UmaP/Synise%20Handbook.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500, chunk_overlap=200
)
doc_splits = text_splitter.split_documents(documents)

vectorstore = FAISS.from_documents(documents=doc_splits,embedding=embed_model)

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank

compressor = FlashrankRerank()
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 20})
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

from operator import itemgetter
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


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

def dummy_payroll_api_call(employee_id, month, year):

  data = {
    2023: {
        "MAY": {
            "employeeDetails": {
                "employeeId": "E2468",
                "firstName": "Sarah",
                "lastName": "Thompson",
                "designation": "Product Manager"
            },
            "paymentDetails": {
                "year": 2023,
                "month": "JAN",
                "basicSalary": 5500,
                "allowances": [
                    {
                        "type": "Housing Allowance",
                        "amount": 1500
                    },
                    {
                        "type": "Travel Allowance",
                        "amount": 800
                    }
                ],
                "deductions": [
                    {
                        "type": "Provident Fund",
                        "amount": 650
                    },
                    {
                        "type": "Health Insurance",
                        "amount": 300
                    }
                ],
                "taxes": [
                    {
                        "type": "Income Tax",
                        "amount": 1300
                    }
                ],
                "grossSalary": 7800,
                "totalDeductions": 2250,
                "netSalary": 6650
            },
            "companyDetails": {
                "companyName": "Tech Solutions Ltd.",
                "address": "789 Maple Avenue, City"
            }
        }
    },
    2024: {
        "JAN": {
            "employeeDetails": {
                "employeeId": "E2468",
                "firstName": "Sarah",
                "lastName": "Thompson",
                "designation": "Product Manager"
            },
            "paymentDetails": {
                "year": 2024,
                "month": "JAN",
                "basicSalary": 6500,
                "allowances": [
                    {
                        "type": "Housing Allowance",
                        "amount": 1500
                    },
                    {
                        "type": "Travel Allowance",
                        "amount": 800
                    }
                ],
                "deductions": [
                    {
                        "type": "Provident Fund",
                        "amount": 650
                    },
                    {
                        "type": "Health Insurance",
                        "amount": 300
                    }
                ],
                "taxes": [
                    {
                        "type": "Income Tax",
                        "amount": 1300
                    }
                ],
                "grossSalary": 8800,
                "totalDeductions": 2250,
                "netSalary": 6550
            },
            "companyDetails": {
                "companyName": "Tech Solutions Ltd.",
                "address": "789 Maple Avenue, City"
            }
        },
        "FEB": {
            "employeeDetails": {
                "employeeId": "E2468",
                "firstName": "Sarah",
                "lastName": "Thompson",
                "designation": "Product Manager"
            },
            "paymentDetails": {
                "year": 2024,
                "month": "FEB",
                "basicSalary": 6500,
                "allowances": [
                    {
                        "type": "Housing Allowance",
                        "amount": 1500
                    },
                    {
                        "type": "Travel Allowance",
                        "amount": 800
                    }
                ],
                "deductions": [
                    {
                        "type": "Provident Fund",
                        "amount": 650
                    },
                    {
                        "type": "Health Insurance",
                        "amount": 300
                    }
                ],
                "taxes": [
                    {
                        "type": "Income Tax",
                        "amount": 1300
                    }
                ],
                "grossSalary": 8800,
                "totalDeductions": 2250,
                "netSalary": 6550
            },
            "companyDetails": {
                "companyName": "Tech Solutions Ltd.",
                "address": "789 Maple Avenue, City"
            }
        },
                "MAY": {
            "employeeDetails": {
                "employeeId": "E2468",
                "firstName": "Sarah",
                "lastName": "Thompson",
                "designation": "Product Manager"
            },
            "paymentDetails": {
                "year": 2024,
                "month": "MAY",
                "basicSalary": 6500,
                "allowances": [
                    {
                        "type": "Housing Allowance",
                        "amount": 1500
                    },
                    {
                        "type": "Travel Allowance",
                        "amount": 800
                    }
                ],
                "deductions": [
                    {
                        "type": "Provident Fund",
                        "amount": 650
                    },
                    {
                        "type": "Health Insurance",
                        "amount": 300
                    }
                ],
                "taxes": [
                    {
                        "type": "Income Tax",
                        "amount": 1500
                    }
                ],
                "grossSalary": 8800,
                "totalDeductions": 2450,
                "netSalary": 6350
            },
            "companyDetails": {
                "companyName": "Tech Solutions Ltd.",
                "address": "789 Maple Avenue, City"
            }
        },
        "APR": {
            "employeeDetails": {
                "employeeId": "E2468",
                "firstName": "Sarah",
                "lastName": "Thompson",
                "designation": "Product Manager"
            },
            "paymentDetails": {
                "year": 2024,
                "month": "APR",
                "basicSalary": 6500,
                "allowances": [
                    {
                        "type": "Housing Allowance",
                        "amount": 1500
                    },
                    {
                        "type": "Travel Allowance",
                        "amount": 800
                    }
                ],
                "deductions": [
                    {
                        "type": "Provident Fund",
                        "amount": 650
                    },
                    {
                        "type": "Health Insurance",
                        "amount": 300
                    }
                ],
                "taxes": [
                    {
                        "type": "Income Tax",
                        "amount": 1500
                    }
                ],
                "grossSalary": 8800,
                "totalDeductions": 2450,
                "netSalary": 6350
            },
            "companyDetails": {
                "companyName": "Tech Solutions Ltd.",
                "address": "789 Maple Avenue, City"
            }
        }
    }
}
  year= 2024 if year == "CUR" else year
  year= 2023 if year == "PREV" else year

  month= "MAY" if month == "CUR" else month
  month= "APR" if month == "PREV" else month


  return data[year][month]

# print(dummy_payroll_api_call(1234, 'CUR', 2024))

import time
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser

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

# print(router_chain.invoke({"question":"What is my salary on 6 2024 ?"}))

# print(router_chain.invoke({"question":"What is leave policy ?"}))

payroll_schema= {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Monthly Payslip",
  "description": "A schema for a monthly payslip",
  "type": "object",
  "properties": {
    "employeeDetails": {
      "type": "object",
      "properties": {
        "employeeId": {
          "type": "string",
          "description": "Unique identifier for the employee"
        },
        "firstName": {
          "type": "string",
          "description": "First name of the employee"
        },
        "lastName": {
          "type": "string",
          "description": "Last name of the employee"
        },
        "designation": {
          "type": "string",
          "description": "Designation or job title of the employee"
        }
      },
      "required": ["employeeId", "firstName", "lastName", "designation"]
    },
    "paymentDetails": {
      "type": "object",
      "properties": {
        "year": {
          "type": "integer",
          "description": "Year of the pay period"
        },
        "month": {
          "type": "string",
          "enum": ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"],
          "description": "Month of the pay period"
        },
        "basicSalary": {
          "type": "number",
          "description": "Basic salary of the employee"
        },
        "allowances": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "type": {
                "type": "string",
                "description": "Type of allowance"
              },
              "amount": {
                "type": "number",
                "description": "Amount of the allowance"
              }
            },
            "required": ["type", "amount"]
          }
        },
        "deductions": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "type": {
                "type": "string",
                "description": "Type of deduction"
              },
              "amount": {
                "type": "number",
                "description": "Amount of the deduction"
              }
            },
            "required": ["type", "amount"]
          }
        },
        "taxes": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "type": {
                "type": "string",
                "description": "Type of tax"
              },
              "amount": {
                "type": "number",
                "description": "Amount of the tax"
              }
            },
            "required": ["type", "amount"]
          }
        },
        "grossSalary": {
          "type": "number",
          "description": "Gross salary (basic salary + allowances)"
        },
        "totalDeductions": {
          "type": "number",
          "description": "Total deductions (including taxes)"
        },
        "netSalary": {
          "type": "number",
          "description": "Net salary (gross salary - total deductions)"
        }
      },
      "required": ["year", "month", "basicSalary", "allowances", "deductions", "taxes", "grossSalary", "totalDeductions", "netSalary"]
    },
    "companyDetails": {
      "type": "object",
      "properties": {
        "companyName": {
          "type": "string",
          "description": "Name of the company"
        },
        "address": {
          "type": "string",
          "description": "Address of the company"
        }
      },
      "required": ["companyName", "address"]
    }
  },
  "required": ["employeeDetails", "paymentDetails", "companyDetails"]
}

# print(str(payroll_schema))

import time
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser

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

# print(fiter_extraction_chain.invoke({"question":"What is my salary on 6 2024 ?"}))

import time
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser

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

payroll_qa_chain = payroll_qa_prompt | llm | StrOutputParser()

result = fiter_extraction_chain.invoke({"question":"What is my salary on jan 2024 ?"})

result

api_result = dummy_payroll_api_call(1234, result["month"], result["year"])

api_result

payroll_qa_chain.invoke({"question":"What is my salary on jan 2024 ?", "data":api_result, "schema":payroll_schema})




### Retrieval Grader

from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser


RETREIVAL_GRADER_PROMPT = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a grader assessing relevance of a retrieved document to a user question. \n 
    Here is the retrieved document: \n\n {document} \n\n
    Here is the user question: {question} \n
    If the document contains keywords related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score',
    Do not include any preamble, explanation, or additional text
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "document"],
)

retrieval_grader = RETREIVAL_GRADER_PROMPT | llm | JsonOutputParser()

# question = "agent memory"
# docs = retriever.get_relevant_documents(question)
# doc_txt = docs[1].page_content
# print(retrieval_grader.invoke({"question": question, "document": doc_txt}))


### Hallucination Grader

# Prompt
HALLUCINATION_GRADER_PROMPT = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a grader assessing whether an answer is grounded in / supported by a set of facts. \n 
    Here are the facts:
    \n ------- \n
    {documents} 
    \n ------- \n
    Here is the answer: {answer}
    Give a binary score 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts. \n
    Provide the binary score as a JSON with a single key 'score'.
    Do not include any preamble, explanation, or additional text
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["answer", "documents"],
)

hallucination_grader = HALLUCINATION_GRADER_PROMPT | llm | JsonOutputParser()
# hallucination_grader.invoke({"documents": docs, "generation": generation})


### Answer Grader

# Prompt
ANSWER_GRADER_PROMPT = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a grader assessing whether an answer is useful to resolve a question. \n 
    Here is the answer:
    \n ------- \n
    {answer} 
    \n ------- \n
    Here is the question: {question}
    Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question. \n
    Provide the binary score as a JSON with a single key 'score'.
     Do not include any preamble, explanation, or additional text
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["answer", "question"],
)

answer_grader = ANSWER_GRADER_PROMPT | llm | JsonOutputParser()

## Question Re-writer

# Prompt
REWRITER_PROMPT = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You a question re-writer that converts an input question to a better version that is optimized \n 
     for vectorstore retrieval. Look at the initial and formulate an improved question. \n
     Here is the initial question: \n\n {question}. Improved question with no preamble: \n 
     <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["answer", "question"],
)

question_rewriter = REWRITER_PROMPT | llm | StrOutputParser()
# question_rewriter.invoke({"question": question})

# answer_grader.invoke({"question": question, "generation": generation})

########### Create Nodes and Actions ###########
from typing_extensions import TypedDict
from typing import List

class AgentState(TypedDict):
    question : str
    answer : str
    documents : List[str]

import logging as log

def route_question(state):
    """
    Route question to payroll_agent or policy_agent to retrieve reevant data

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    question = state["question"]
    result = router_chain.invoke({"question": question})

    log.debug('Routing to {}....'.format(result["agent"]))

    return result["agent"]

state = AgentState(question="What is my salary on jan 2024 ?", answer="", documents=None)
route_question(state)

from langchain.schema import Document
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

# state = AgentState(question="What is leave policy?", answer="", documents=[Document(page_content="According to leave policy, there are two types of leaves 1: PL 2: CL")])
# generate_answer(state)

def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score["score"]
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
    return {"documents": filtered_docs, "question": question}

def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]

    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}

def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    question = state["question"]
    filtered_documents = state["documents"]

    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
        )
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"
    
def fallback(state):
    """
    Fallback to default answer.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

   
    return {"answer": "Sorry,I don't know the answer to this question."}
    
def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    answer = state["answer"]

    score = hallucination_grader.invoke(
        {"documents": documents, "answer": answer}
    )
    grade = score["score"]

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "answer": answer})
        grade = score["score"]
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"

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
    payroll_api_query_results = dummy_payroll_api_call(1234, result["month"], result["year"])


    context = context = 'PAYROLL DATA SCHEMA: \n {payroll_schema} \n PAYROLL DATA: {payroll_api_query_results}'.format(
    payroll_schema=payroll_schema, payroll_api_query_results=payroll_api_query_results)

    documents = [Document(page_content=context)]
    return {"documents": documents, "question": question}

# state = AgentState(question="Tell me salary for Jan 2024?", answer="", documents=None)
# query_payroll(state)


########### Build Execution Graph ###########
from langgraph.graph import END, StateGraph
workflow = StateGraph(AgentState)

# Define the nodes
workflow.add_node("payroll", payroll)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
# workflow.add_node("grade_documents", grade_documents)  # grade documents
# workflow.add_node("transform_query", transform_query)  # transform_query
# workflow.add_node("fallback", fallback)

workflow.set_conditional_entry_point(
    route_question,
    {
        "payroll": "payroll",
        "raqa": "retrieve",
    },
)
workflow.add_edge("payroll", "generate")
# workflow.add_edge("retrieve", "generate")
# workflow.add_edge("generate", END)
workflow.add_edge("retrieve", "generate")
# workflow.add_conditional_edges(
#     "grade_documents",
#     decide_to_generate,
#     {
#         "transform_query": "transform_query",
#         "generate": "generate",
#     },
# )
# workflow.add_edge("transform_query", "retrieve")
# workflow.add_conditional_edges(
#     "generate",
#     grade_generation_v_documents_and_question,
#     {
#         "not supported": "generate",
#         "useful": END,
#         "not useful": "fallback",
#     },
# )
workflow.add_edge("generate", END)

app = workflow.compile()
