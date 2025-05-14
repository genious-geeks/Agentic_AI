import typer
from typing import Optional,List
from rich.prompt import Prompt
from phi.assistant import Assistant
from phi.storage.assistant.sqllite import SqlAssistantStorage
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.chroma import ChromaDb
from phi.embedder.ollama import OllamaEmbedder
from phi.model.groq import Groq

import os
from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv('GROQ_API_KEY')

knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=ChromaDb(collection="recipes",
                       embedder=OllamaEmbedder(model='nomic-embed-text:latest')),
)

knowledge_base.load(recreate=False)

storage = SqlAssistantStorage(table_name="pdf_assistant")

def pdf_assistant(new: bool = False, user: str = "user"):
    run_id: Optional[str] = None

    if not new:
        existing_run_ids: List[str] = storage.get_all_run_ids(user)
        if len(existing_run_ids) > 0:
            run_id = existing_run_ids[0]

    assistant = Assistant(
        model=Groq(id="llama-3.2-1b-Preview"),
        run_id= run_id,
        user_id= user,
        knowledge_base= knowledge_base,
        storage= storage,
        show_tool_calls=True,
        search_knowledge=True,
        read_chat_history=True,
        instructions= ["Always include sources", "Use table to display tha data"]
    )

    if run_id is None:
        run_id = assistant.run_id
        print(f"Starting Run: {run_id} \n")

    else:
        print(f"Continuing Run: {run_id} \n")

    assistant.cli_app(markdown=True)

    while True:
        message = Prompt.ask(f"[bold] :sunglasses: {user} [/bold]")
        if message in ("exit", "bye"):
            break
        assistant.print_response(message)

if __name__ == "main":
    typer.run(pdf_assistant)

