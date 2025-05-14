from phi.agent import Agent
from phi.model.openai import OpenAIChat
# from phi.embedder.openai import OpenAIEmbedder
from phi.embedder.ollama import OllamaEmbedder
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.lancedb import LanceDb, SearchType
from phi.model.groq import Groq
from dotenv import load_dotenv

load_dotenv()


# Create a knowledge base from a PDF
knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    # Use LanceDB as the vector database
    vector_db=LanceDb(
        table_name="recipes",
        uri="tmp/lancedb",
        search_type=SearchType.vector,
        embedder=OllamaEmbedder(model='nomic-embed-text:latest'),
    ),
)
# Comment out after first run as the knowledge base is loaded
knowledge_base.load()

agent = Agent(
    model=Groq(id="llama-3.2-1b-Preview"),
    # Add the knowledge base to the agent
    knowledge=knowledge_base,
    show_tool_calls=True,
    markdown=True,
)
agent.print_response("How many recipes are there in the document", stream=True)