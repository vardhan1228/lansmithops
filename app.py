from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import os
from dotenv import load_dotenv

load_dotenv()

# LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an experienced Forward Deployed Engineer (FDE).
Your task is to answer the user's question using only the information
provided in the context.

Instructions:
- Give a clear and practical answer.
- Explain concepts in simple technical language.
- If the answer is not available in the context, say:
  "The provided context does not contain enough information to answer this."
- Do not make up information.
"""
    ),
    (
        "user",
        """Question: {question}

Context:
{context}

Based on the context above, provide the most relevant answer."""
    )
])

# Gemini
model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0
)

output_parser = StrOutputParser()

chain = prompt | model | output_parser

question = "What are the key responsibilities of a Forward Deployed Engineer?"

context = """
A Forward Deployed Engineer (FDE) works closely with customers to understand their
business problems and translate them into practical technical solutions. An FDE may
work across cloud platforms, APIs, data pipelines, AI systems, and application
infrastructure depending on the customer's requirements. The role requires strong
problem-solving skills because every customer environment can have different
technical constraints. FDEs often build prototypes, integrate existing systems,
develop automation, and deploy solutions into production environments. They work
with engineering and product teams to convert customer feedback into scalable
features. In AI projects, an FDE may integrate LLMs, RAG pipelines, vector databases,
agents, and observability tools into real business applications. They also troubleshoot
production issues and continuously improve the deployed solution. Communication is
important because FDEs need to explain technical solutions to both engineers and
business stakeholders. The overall goal of an FDE is to move from a customer problem
to a working, reliable, and production-ready technical solution.
"""

print(
    chain.invoke({
        "question": question,
        "context": context
    })
)