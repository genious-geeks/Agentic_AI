
from phi.knowledge.pdf import PDFKnowledgeBase, PDFReader
# from phi.vectordb.pgvector import PgVector
from phi.vectordb.lancedb import LanceDb, SearchType
from phi.embedder.ollama import OllamaEmbedder
from phi.model.groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()


pdf_knowledge_base = PDFKnowledgeBase(
    path="C:\\Users\\nivashinib\\Downloads\\Retail Banking - Deposits.pdf",
    # Table name: ai.pdf_documents
    vector_db=LanceDb(
        table_name="pdf",
        uri="tmp/lancedb",
        search_type=SearchType.vector,
        embedder=OllamaEmbedder(model='nomic-embed-text:latest'),
    ),
    reader=PDFReader(chunk=True),
)

from phi.agent import Agent

agent = Agent(
    model=Groq(id="llama-3.2-1b-Preview"),
    knowledge=pdf_knowledge_base,
    search_knowledge=True,
    read_chat_history=True,
    show_tool_calls=True,
    markdown=True
)
agent.knowledge.load(upsert=True)

agent.print_response("what is term desposit")