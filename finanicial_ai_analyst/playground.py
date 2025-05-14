from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import phi.api
from dotenv import load_dotenv
import os
from phi.playground import Playground, serve_playground_app
import phi

phi.api = os.getenv('API_KEY')

load_dotenv()
Groq.api_key = os.getenv('GROQ_API_KEY')

wed_search_agent = Agent(
    name = "Web Search Agent",
    role = "Search in the web for info",
    model = Groq(id="llama-3.2-1b-Preview"),
    tools = [DuckDuckGo()],
    show_tool_calls=True,
    markdown=True
)

finance_agent = Agent(
    name = "Finance AI agent",
    model = Groq(id="llama-3.2-1b-Preview"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_news=True)],
    instructions= ["Use tables to display the data"],
    show_tool_calls=True,
    markdown=True
)

app = Playground(agents= [wed_search_agent, finance_agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("playground:app", reload=True)

