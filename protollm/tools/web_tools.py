from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.tools.render import render_text_description
import os


tavily_tool = None

if os.getenv('TAVILY_API_KEY') is not None:
    tavily_tool = TavilySearchResults(max_results=5)
    web_tools = [tavily_tool]
else:
    web_tools = [DuckDuckGoSearchResults()]
    
web_tools_rendered = render_text_description(web_tools).replace('duckduckgo_results_json', 'DuckDuckGoSearchResults')
