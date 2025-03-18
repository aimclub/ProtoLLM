import pytest

from protollm_sdk.models.job_context_models import PromptModel, PromptMeta, ChatCompletionModel

@pytest.mark.ci
def test_from_prompt_model():
    prompt_model = PromptModel(
        job_id="test_job_123",
        meta=PromptMeta(
            temperature=0.5,
            tokens_limit=100,
            stop_words=["stop", "words"],
            model="gpt-3"
        ),
        content="This is a test prompt"
    )

    chat_completion = ChatCompletionModel.from_prompt_model(prompt_model)

    assert chat_completion.job_id == prompt_model.job_id
    assert chat_completion.meta == prompt_model.meta

    assert len(chat_completion.messages) == 1

    assert chat_completion.messages[0].role == "user"
    assert chat_completion.messages[0].content == prompt_model.content

    assert chat_completion.meta.temperature == 0.5
    assert chat_completion.meta.tokens_limit == 100
    assert chat_completion.meta.stop_words == ["stop", "words"]
    assert chat_completion.meta.model == "gpt-3"
