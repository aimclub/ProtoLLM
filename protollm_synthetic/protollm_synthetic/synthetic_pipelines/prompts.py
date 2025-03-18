from typing import List

# Summarisation


def generate_summary_system_prompt() -> str:
    return f"""Summarize the following text in a concise manner, in the same language as the original text
    Pay attention that you should save only importaint information so the final summary is short and concise."""


def generate_summary_human_prompt() -> str:
    return "Text that should be summarised:\n\n{text}\n\n Summary:"


def generate_detailed_summary_prompt(text: str) -> str:
    return """Provide a detailed summary of the following text, highlighting key points and important details, in the same language as the original text."""


def generate_bullet_point_summary_system_prompt(text: str) -> str:
    return f"Summarize the following text using bullet points for clarity, in the same language as the original text:\n\n{text}\n\nBullet Point Summary:"


def generate_question_based_summary_prompt(text: str) -> str:
    return f"Summarize the following text by answering the question: What are the main ideas and conclusions, in the same language as the original text?\n\n{text}\n\nSummary:"


def generate_summary_with_length_constraint_prompt(text: str, max_length: int) -> str:
    return f"Summarize the following text in no more than {max_length} words, in the same language as the original text:\n\n{text}\n\nSummary:"


# Aspect summarisation


def generate_aspect_summarisation_prompt(aspect: str) -> str:
    return f"""You are a professional specialist in {aspect}.
    Summarise the following text by identifying and summarising the main aspects, 
    Strictly follow the rules below:
    - Do not include any information that is not related to the aspect. If the text is not related to the aspect, return an empty string.
    - Use the same language as the original text.
    - Final summary should be short and concise. Shorter than the original text.
    """


# test generation


def generate_quiz_system_prompt() -> str:
    return """You are a professional teacher and you know that students learn better when they complete quizes with multiple choices on the material they are learning.
    You will be provided with a text and you need to generate a quiz for it.
    Strictly follow the rules below:
    - Generate up to 5 questions and answers for a given text.
    - The questions should be relevant to the text. It means that the question can be answered after reading the text.
    - Each question should have 4 options with one or more correct answers.
    - The correct options should be marked with a special symbol '(X)' after the option.
    - The questions should not be obvious so students should have to think to choose the correct options.
    - The questions should be diverse and cover different aspects of the text.
    - The questions should be in the same language as the text.
    The response should be in the following format:
    {response_format_description}
    """


def generate_quiz_human_prompt() -> str:
    return """Text that should be used for quiz generation: 
    {text}
    """


# RAG


def generate_rag_system_prompt() -> str:
    return """Create a set of qestion and answer pairs to check the quality of a RAG system.
    User will provide a text for you to create a set of questions and corresponding answers.
    When generating follow the rules below:
    - Generate 5 questions and answers
    - The questions should be relevant to the text. It means that the question can be answered after reading the text.
    - The questions should be of different categories: "easy" - simple question asking about some fact from the text, "medium" - question require to analyse and synthesise information from the text e.g. comparision or reasoning, "hard" - requires deep understanding on application level and the answer is not obvious.
    - The questions should be diverse and cover different aspects of the text.
    - The questions should be in the same language as the text. 
    Return answer in the following format:
    {response_format_description}
    """


def generate_rag_human_prompt() -> str:
    return """Text that should be used for question and answers generation: 
    {text}
    """


# Instruction design


def generate_instruction_design_prompt(tasks: List[str], solutions: List[str]) -> str:
    """
    Generates a prompt for the LLM to design an instruction for itself to accomplish the given tasks.

    :param tasks: A list of tasks that need to be accomplished.
    :param solutions: A list of solutions corresponding to each task.
    :return: A formatted string prompt for the LLM.
    """
    task_solution_pairs = "\n".join(
        f"Task: {task}\nSolution: {solution}"
        for task, solution in zip(tasks, solutions)
    )
    return (
        f"Design an instruction for yourself to accomplish the following tasks based on the provided solutions:\n\n"
        f"{task_solution_pairs}\n\nInstruction:"
    )


def generate_instruction_one_shot_system_prompt() -> str:
    return """Design an instruction for yourself to transform the following text to the provided result.
    You will be provided with some text and result that user want to obtain from the text.
    You should define the instruction following the rules below:
    - The instruction should be concise and to the point.
    - Define all the special conditions that should be met to transform the text to the result.
    Return answer in the following format:
    {response_format_description}"""


def generate_instruction_one_shot_human_prompt() -> str:
    return """Text that should be transformed:
    {text}
    Result that should be obtained:
    {result}"""


def merge_instructions() -> str:
    return """Merge the following instructions into one instruction according to the set of presented fields.
    You will be provided with several instructions to transform text into the predefined resultand you need to merge them into one json instruction.
    Save as much details as possible to save the quality of instructions.
    Return answer in the following format:
    {response_format_description}"""


def merge_instructions_human_prompt() -> str:
    return """Instructions to merge in json format:
    {text}"""


# Augmentation


def paraphrase_text() -> str:
    return "Paraphrase the following text in a concise manner, in the same language as the original text:\n\n{text}\n\nParaphrased Text:"


# Evaluation


def generate_summary_evaluation_system_prompt() -> str:
    return """Evaluate the quality of the following summary and provide a score from 1 to 10 with feedback.
    Return answer in the following format: 
    {response_format_description}"""


def generate_aspect_summarisation_evaluation_system_prompt(aspect: str) -> str:
    return (
        f"""Evaluate the quality of the following aspect summarisation on a topic of {aspect} and provide a score from 1 to 10 with feedback.
    """
        + """Return answer in the following format: 
    {response_format_description}"""
    )


def check_summary_quality_human_prompt() -> str:
    return "Initial text: {text}\n\nSummary: {summary}\n"


# Genetic Evolution


def generate_genetic_evolution_prompt() -> str:
    return """You are a genetic algorithm that evolves the instruction to answer the question.
    You will be provided with an instruction and a question.
    You need to evolve the instruction to answer the question.
    """
