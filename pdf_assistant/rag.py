from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.knowledge.pdf import PDFUrlKnowledgeBase
# from phi.vectordb.pgvector import PgVector, SearchType
from phi.vectordb.lancedb import LanceDb, SearchType
from phi.embedder.ollama import OllamaEmbedder
from phi.model.groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

# db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"
# knowledge_base = PDFUrlKnowledgeBase(
#     urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
#     vector_db=PgVector(table_name="recipes", db_url=db_url, search_type=SearchType.hybrid),
# )

knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    # Use LanceDB as the vector database
    vector_db=LanceDb(
        table_name="recipes",
        uri="lancedb",
        search_type=SearchType.hybrid,
        embedder=OllamaEmbedder(model='nomic-embed-text:latest'),
    ),
)
# Load the knowledge base: Comment out after first run
knowledge_base.load(upsert=True)

agent = Agent(
    model=Groq(id="llama-3.2-1b-Preview"),
    knowledge=knowledge_base,
    # Add a tool to search the knowledge base which enables agentic RAG.
    search_knowledge=True,
    # Add a tool to read chat history.
    read_chat_history=True,
    show_tool_calls=True,
    markdown=True,
    # debug_mode=True,
)
agent.print_response("what data you have", stream=True)
