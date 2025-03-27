"""
Example of launching the Agent-Summarizer
"""

from protollm.agents.universal_agents import summary_node
from protollm.connectors import create_llm_connector

if __name__ == "__main__":
    query = "Make a story about a boy drawing. Recipe for duck in the oven"
    response = "I found answers to both questions."
    state = {
        "input": query,
        "response": response,
        # past_steps consist of List[Tuple('task', 'answer')] - task is name of func
        "past_steps": [("agent_storyteller", """The child sits at the table, holding a crayon tightly in their hand. 
    They look at the blank sheet of paper, deep in thought, deciding what to 
    draw. Slowly, they begin making small, careful lines, their concentration 
    intense. As they gain confidence, the lines become bolder, forming shapes 
    and patterns. The child adds colors, using bright reds, blues, and yellows, 
    filling the spaces with excitement. They carefully stay within the lines, 
    but sometimes a splash of color goes outside, adding a playful touch. 
    Every so often, they stop to admire their work, grinning with pride. 
    The drawing starts to come together, showing a simple scene—maybe a sun, 
    a house, or a tree. With each stroke, the child’s imagination comes to 
    life on the paper. Finally, they put down the crayon, pleased with their 
    masterpiece, and smile at the colorful creation in front of them. """),
                       ("web_search", """To cook duck in the oven, first rinse and pat dry the duck, seasoning 
    it with salt, pepper, and your favorite spices. Stuff the cavity with a 
    couple of garlic cloves and apples, then preheat the oven to 180°C (350°F). 
    Place the duck in a roasting pan and roast for 1.5 to 2 hours, basting 
    it occasionally with the juices. About 15 minutes before the end, brush 
    the duck with honey or soy sauce for a golden, crispy skin, and serve 
    it hot with a side of potatoes or vegetables.""")],
    }

    model = create_llm_connector(
        "https://api.vsegpt.ru/v1;meta-llama/llama-3.1-70b-instruct"
    )
    conf = {"llm": model, "max_retries": 1}
    res = summary_node(state, conf)

    print("Total answer: ")
    print(res["response"])
