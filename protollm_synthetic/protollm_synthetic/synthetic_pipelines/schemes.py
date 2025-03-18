from pydantic import BaseModel, Field

from typing import List


class SummaryQualitySchema(BaseModel):
    is_informative: bool = Field(
        ...,
        description="Does the summary contain all the necessary information from the text?",
    )
    is_concise: bool = Field(
        ..., description="Is the summary concise and to the point?"
    )
    is_language_correct: bool = Field(
        ..., description="Is the summary in the same language as the original text?"
    )
    score: int = Field(
        ...,
        description="Based on the evaluated criteria of the summary, score the resulting summary from 1 to 10",
    )


class AspectSummarisationQualitySchema(BaseModel):
    is_aspect_covered: bool = Field(
        ..., description="Does the aspect summarisation cover the aspect?"
    )
    is_aspect_informative: bool = Field(
        ..., description="Is the aspect summarisation informative?"
    )
    is_aspect_concise: bool = Field(
        ..., description="Is the aspect summarisation concise?"
    )
    is_language_correct: bool = Field(
        ...,
        description="Is the aspect summarisation in the same language as the original text?",
    )
    score: int = Field(
        ...,
        description="Based on the evaluated criteria of the aspect summarisation, score the resulting aspect summarisation from 1 to 10",
    )


class RAGExample(BaseModel):
    context: str = Field(
        ...,
        description="The context that the RAG system should use to answer the question",
    )
    question: str = Field(
        ..., description="The question that the RAG system should answer"
    )
    difficulty: str = Field(..., description="The difficulty of the question")
    answer: str = Field(..., description="The answer to the question")


class RAGScheme(BaseModel):
    context: List[RAGExample] = Field(
        ...,
        description="The context that the RAG system should use to answer the question",
    )


class QuizExample(BaseModel):
    question: str = Field(..., description="The quiz question on a provided text")
    options: List[str] = Field(
        ...,
        description="The 4 options that are presented to the student. Mark the correct options",
    )


class QuizScheme(BaseModel):
    quiz: List[QuizExample] = Field(
        ..., description="The quiz for students on a given text"
    )


class FreeQueryScheme(BaseModel):
    system_role_part: str = Field(
        ...,
        description="Define a role that llm will play in solving the task, e.g. 'You are a professional teacher who is generating free-answerquestions for students to check their knowledge'",
    )
    system_general_instruction_part: str = Field(
        ...,
        description="Define a general instruction for llm to follow, e.g. 'You should generate a question that is easy to understand and follow and a correct answer based on the provided text'",
    )
    system_specifics_instruction_part: str = Field(
        ...,
        description="Define a set of strict rules and domain specific information for llm to know and follow, e.g. '1. The question should be in the same language as the text, 2. The question should be easy to understand and follow, 3. The question should be a free-answer question'",
    )
    system_output_format_part: str = Field(
        ...,
        description="Define the output format of the llm, e.g. 'The output should be a json object with the following fields: question, options, correct_options'",
    )


class FreeQueryMerger(BaseModel):
    system_role_part: str = Field(
        ..., description="Merge the proposed role parts into one role"
    )
    system_general_instruction_part: str = Field(
        ...,
        description="Merge the proposed general instruction parts into one general instruction",
    )
    system_specifics_instruction_part: str = Field(
        ...,
        description="Merge the proposed specifics instruction parts into one detailed instruction",
    )
    system_output_format_part: str = Field(
        ..., description="Merge the proposed output format parts into one output format"
    )
