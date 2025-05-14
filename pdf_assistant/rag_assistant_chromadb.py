import typer
from rich.prompt import Prompt
from typing import Optional

from phi.agent import Agent
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.chroma import ChromaDb
from phi.embedder.ollama import OllamaEmbedder
from phi.model.groq import Groq
from phi.vectordb.lancedb import LanceDb
from phi.vectordb.search import SearchType
from dotenv import load_dotenv
import os

load_dotenv()

vector_db = LanceDb(
    table_name="recipes",
    uri=os.path.join(os.getcwd(), 'lancedb'),
    search_type=SearchType.keyword,
    embedder=OllamaEmbedder(model='nomic-embed-text:latest')
)

knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db= vector_db
    
    # ChromaDb(path=os.path.join(os.getcwd(), 'chromadb'),
    #                    collection="recipes",
    #                    embedder=OllamaEmbedder(model='nomic-embed-text:latest'))
                       )

# Comment out after first run
knowledge_base.load(recreate=True)


def pdf_agent(user: str = "user"):
    run_id: Optional[str] = None

    agent = Agent(
        model=Groq(id="llama-3.2-1b-Preview"),
        run_id=run_id,
        user_id=user,
        knowledge_base=knowledge_base,
        use_tools=True,
        show_tool_calls=True,
        debug_mode=True,
        instructions= ["Always include sources", "Use table to display tha data"],
        markdown=True,
        search_knowledge=True
    )

    
    if run_id is None:
        run_id = agent.run_id
        print(f"Started Run: {run_id}\n")
    else:
        print(f"Continuing Run: {run_id}\n")

    while True:
        message = Prompt.ask(f"[bold] :sunglasses: {user} [/bold]")
        if message in ("exit", "bye"):
            break
        agent.print_response(message)


if __name__ == "__main__":
    typer.run(pdf_agent)