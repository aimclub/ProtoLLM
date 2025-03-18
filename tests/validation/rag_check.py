import uuid
from protollm_tools.sdk.protollm_sdk.jobs.utility import construct_job_context
from protollm_tools.sdk.protollm_sdk.utils.reddis import get_reddis_wrapper, load_result
from protollm.rags.jobs import RAGJob

# Шаг 1. Инициализация уникального номера идентификации
job_id = str(uuid.uuid4())
# Шаг 2.  Инициализация переменных доступа к БД и SDK
job_name = "fast_validation"
ctx = construct_job_context(job_name)
# Шаг 3. Запуск поиска релевантных документов
RAGJob().run(
    job_id,
    ctx,
    user_prompt="Какой бывает арматура железобетонных конструкций?",
    use_advanced_rag=False,
)
# Шаг 4. Получение ответа модели из базы данных.
rd = get_reddis_wrapper()
result = load_result(rd, job_id, job_name)
