import os
from textwrap import dedent
from crewai import Agent
from tools.browser_tools import BrowserTools
from tools.search_tools import SearchTools
from langchain.agents import load_tools

from crewai_tools.tools import WebsiteSearchTool, SerperDevTool

web_search_tool = WebsiteSearchTool()
serper_dev_tool = SerperDevTool()

from langchain_openai import ChatOpenAI

agent_llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo')


#from langchain.llms import Ollama

class MarketingAnalysisAgents:
	def __init__(self):
		self.llm = agent_llm #Ollama(model=os.environ['MODEL'])

	def product_competitor_agent(self):
		return Agent(
			role="Lead Market Analyst",
			goal=dedent("""\
				Conduct amazing analysis of the products and
				competitors, providing in-depth insights to guide
				marketing strategies."""),
			backstory=dedent("""\
				As the Lead Market Analyst at a premier
				digital marketing firm, you specialize in dissecting
				online business landscapes."""),
			tools=[
					web_search_tool,
					serper_dev_tool
			],
			llm=self.llm,
			verbose=True,
			max_iter=4,
			allow_delegation=True
		)

	def strategy_planner_agent(self):
		return Agent(
			role="Chief Marketing Strategist",
			goal=dedent("""\
				Synthesize amazing insights from product analysis
				to formulate incredible marketing strategies."""),
			backstory=dedent("""\
				You are the Chief Marketing Strategist at
				a leading digital marketing agency, known for crafting
				bespoke strategies that drive success."""),
			tools=[
					web_search_tool,
					SearchTools.search_internet,
					SearchTools.search_instagram
			],
			llm=self.llm,
			verbose=True,
			max_iter=4,
			allow_delegation=True
		)

	def creative_content_creator_agent(self):
		return Agent(
			role="Creative Content Creator",
			goal=dedent("""\
				Develop compelling and innovative content
				for social media campaigns, with a focus on creating
				high-impact Instagram ad copies."""),
			backstory=dedent("""\
				As a Creative Content Creator at a top-tier
				digital marketing agency, you excel in crafting narratives
				that resonate with audiences on social media.
				Your expertise lies in turning marketing strategies
				into engaging stories and visual content that capture
				attention and inspire action."""),
			tools=[
					web_search_tool,
					SearchTools.search_internet,
					SearchTools.search_instagram
			],
			llm=self.llm,
			verbose=True,
			max_iter=4,
			allow_delegation=True
		)
