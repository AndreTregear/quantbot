import os
from dotenv import load_dotenv
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set in your environment or .env file.")


from textwrap import dedent
from crewai import Agent, Crew, Process

from tasks import MarketingAnalysisTasks
from agents import MarketingAnalysisAgents

from langchain_openai import ChatOpenAI

tasks = MarketingAnalysisTasks()
agents = MarketingAnalysisAgents()

print("## Welcome to the marketing Crew")
print('-------------------------------')
product_website = input("What is the product website you want a marketing strategy for?\n")
product_details = input("Any extra details about the product and or the instagram post you want?\n")


# Create Agents
product_competitor_agent = agents.product_competitor_agent()
strategy_planner_agent = agents.strategy_planner_agent()
creative_agent = agents.creative_content_creator_agent()
# Create Tasks
website_analysis = tasks.product_analysis(product_competitor_agent, product_website, product_details)
market_analysis = tasks.competitor_analysis(product_competitor_agent, product_website, product_details)
campaign_development = tasks.campaign_development(strategy_planner_agent, product_website, product_details)
write_copy = tasks.instagram_ad_copy(creative_agent)

llm = ChatOpenAI(
    temperature=0,
    model="gpt-3.5-turbo",
    openai_api_key=openai_api_key
)

copy_crew = Crew(
	agents=[product_competitor_agent, strategy_planner_agent, creative_agent],
	tasks=[website_analysis, market_analysis, campaign_development, write_copy],
	verbose=True,
    process=Process.hierarchical,
    manager_llm=llm,
    memory=True
)
"""
# Create Crew responsible for Copy
copy_crew = Crew(
	agents=[
		product_competitor_agent,
		strategy_planner_agent,
		creative_agent
	],
	tasks=[
		website_analysis,
		market_analysis,
		campaign_development,
		write_copy
	],
	verbose=True,
    process=Process.hierarchical,
    manager_llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
    memory=True
)
"""

ad_copy = copy_crew.kickoff()

# Print results
print("\n\n########################")
print("## Here is the result")
print("########################\n")
print("Your post copy:")
print(ad_copy)
print("'\n\nYour midjourney description:")