from biomni.agent import A1
from prompt import PROMPT
from biomni.config import default_config

default_config.llm='claude-haiku-4-5-20251001'

QUERY = """{{"label": ["CD8a:#00FF00", "MART1:#FF0000", "MHC-I:#FFFF00"]}}"""

# mode="minimal"  -> direct LLM call, no tools, no data lake, no search
# mode="db"       -> data lake + documents only, no web/literature search
# mode="full"     -> everything (default)

agent = A1(llm='claude-sonnet-4-6', mode='db', custom_prompt=PROMPT)
with open("system_prompt.txt", "w", encoding="utf-8") as f:
    f.write(agent.get_system_prompt())
log, response = agent.go(QUERY, max_steps=50, image="C:\\Users\\chaha\\dev\\Biomni\\bioset_biomni\\data\\example.png")
print(response)
