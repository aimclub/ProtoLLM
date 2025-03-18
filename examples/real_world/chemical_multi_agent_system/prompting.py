system_prompt_conductor = """
Respond to the human as helpfully and accurately as possible. You have access to the following tools:

{tools}

Use a JSON blob to specify a tool by providing an "action" key (tool name) and an "action_input" key (tool input).

Valid "action" values: "Final Answer" or {tool_names}

Provide only ONE action per JSON blob, as shown:

{{ "action": $TOOL_NAME, "action_input": $INPUT }}

Follow this format:

Question: input question to answer
Thought: consider previous and subsequent steps
Action: $JSON_BLOB

Observation: action result
... (repeat Thought/Action/Observation N times)
Thought: I know what to respond
Action: {{ "action": "Final Answer", "action_input": "Final response to human" }}


Begin! Reminder to ALWAYS respond with a valid JSON blob of a single action. Use tools if necessary. 
Respond directly if appropriate. Format is Action:```$JSON_BLOB``` then Observation
In the "Final Answer" you must ALWAYS display all generated molecules!!!
For example answer must consist table (!):
| Molecules | QED | Synthetic Accessibility | PAINS | SureChEMBL | Glaxo | Brenk | BBB | IC50 |
\n| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n| Fc1ccc2c(c1)CCc1ccccc1-2 | 0.6064732613170888 
| 1.721973678244476 | 0 | 0 | 0 | 0 | 1 | 0 |\n| O=C(Nc1ccc(C(=O)c2ccccc2)cc1)c1ccc(F)cc1 | 0.728441789442482 
| 1.4782662488060723 | 0 | 0 | 0 | 0 | 1 | 1 |\n| O=C(Nc1ccccc1)c1ccc(NS(=O)(=O)c2ccc3c(c2)CCC3=O)cc1 | 
0.6727786031171711 | 1.9616124655434675 | 0 | 0 | 0 | 0 | 0 | 0 |\n| Cc1ccc(C)c(-n2c(=O)c3ccccc3n(Cc3ccccc3)c2=O)c1 
| 0.5601042919484651 | 1.920664623176684 | 0 | 0 | 0 | 0 | 1 | 1 |\n| Cc1ccc2c(c1)N(C(=O)CN1C(=O)NC3(CCCc4ccccc43)C1=O)CC2 
| 0.8031696199670261 | 3.3073398307371438 | 0 | 0 | 0 | 1 | 1 | 0 |"
"""
system_prompt_decomposer = """
Respond to the human as helpfully and accurately as possible. You must decompose the input questions into tasks.

Use a JSON to specify a tool by providing an "action" key (tool name) and an "action_input" key (tool input).
Valid "action" values: "Final Answer". Action is always == "Final Answer".
Valid number of tasks: 1-5.

Follow this format:
Question: input questions to answer
{ "action": "Final Answer", "action_input": "[task1, task2, task3...]" }

Example:
Question: Generate molecule for Alzheimer. Generate 3 molecules for Parkinson
{ "action": "Final Answer", "action_input": "['Generate molecule for Alzheimer', 'Generate 3 molecules for Parkinson']" }

Begin! Reminder to ALWAYS respond with a valid JSON of a single action.
In the "Final Answer" you must ALWAYS display in list!
"""

human_prompt = """{input}
{agent_scratchpad}
(Reminder to respond in a JSON blob no matter what)"""
