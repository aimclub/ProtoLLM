
# Пример заполнения шаблона - objects = 'характеристики различных зданий в Санкт-Петербурге',
# user_field = 'ГИС-системы', N=10

synthetic_prompt_template_basic = '''В файле даны {objects}. 
Проанализируй файл и сформулируй {N} вопросов, которые мог бы задать 
пользователь {user_field}.'''

# data_desc = 'зданий в Санкт-Петербурге', stucture - JSON-описание, role - 'урбанист',
# work = 'анализ различных  данных о городе для получения некоторых практических выводов'
synthetic_prompt_template_advanced = ''' 
Датасет с данными {data_desc} имеет следующую структуру: {structures}. 
Представь, что ты - {role}. В круг твоих задач входит {work}. 
Сформулируй {N} вопросов, которые мог бы задать {role}, основываясь на 
данных датасета.'''

prompt_template_enrichment = """Теперь представь, что у тебя есть другие данные о {data_desc}, и 
можно ориентироваться не только на структуру датасета. Сгенерируй ещё {N} вопросов по теме {data_desc}."""

prompt_template_file_only = """В файле даны характеристики различных {data_desc}. 
Проанализируй файл и сформулируй {N} вопросов, которые мог бы 
задать пользователь {user_field}. Ответь на них на основе информации из файла."""

prompt_template_expert_role = """Представь, что ты - {role}, 
который занимается {work}. Вот примеры вопросов, которые задают специалисты в этой области: 
{questions}. Сформулируй ещё {N} вопросов. Они могут быть как широкими и обобщёнными, так и узкоспециализированными.
"""
