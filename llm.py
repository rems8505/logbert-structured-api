# llm.py

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, ValidationError
from langchain_google_genai import ChatGoogleGenerativeAI
import os

# API key setup (ensure you load from env in prod)
# GOOGLE_API_KEY = "YOUR_GEMINI_KEY"  # Replace this or set via os.environ
GOOGLE_API_KEY = "AIzaSyC0To1gZRtPE3WVqbJCDTXWZ7HFq4BH_vs"

# Output Schema
class RCAResponse(BaseModel):
    root_cause: str = Field(..., description="The most likely root cause of the issue")
    suggested_fix: str = Field(..., description="Recommended fix for the issue")

# Output Parser
parser = PydanticOutputParser(pydantic_object=RCAResponse)

# Prompt Template
prompt = ChatPromptTemplate.from_template(
    template="""
You are an expert in analyzing system logs and identifying root causes.
Given the following anomalous log, respond in JSON format with two fields: root_cause and suggested_fix.

Anomalous Log:
{log_text}

{format_instructions}
""",
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# Memory for conversation context (optional)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize LLM chain
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-latest",
    google_api_key=GOOGLE_API_KEY
)

rca_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True
)

# Callable function
def analyze_log(log_text: str) -> dict:
    try:
        response = rca_chain.run(log_text=log_text)
        parsed_response = parser.parse(response)
        return {
            "root_cause": parsed_response.root_cause,
            "suggested_fix": parsed_response.suggested_fix
        }
    except ValidationError as ve:
        return {"error": "Response parsing failed", "details": str(ve)}
    except Exception as e:
        return {"error": "LLM call failed", "details": str(e)}
