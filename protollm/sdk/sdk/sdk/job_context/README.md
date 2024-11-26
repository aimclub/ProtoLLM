>Этот файл является шаблоном readme для разрабатываемых модулей, который призван облегчить работу разработчиков бекенда по интеграции внешних модулей.
>Все содержание под чертой должно быть скопировано в начало файла `README.md` в корне проекта, чтобы markdown отображался на главной странице проекта.
>То, что написано в кавычках, переносится без изменения и без кавычек.

___
# Название модуля (stairs-...)
>Краткое опиание предназначения модуля через use-case в 1-2 предложениях.

## Оглавление 
>Если вы ведете README и у вас есть несколько разделов, то обязательно добавьте оглавление. Иначе можно пропустить.
## "Интеграция"
### Название исполняемой функциональности (например: Парсинг pdf)
"Данная функциональность реализована через класс, наследующий Job" `MyJobName`. "Он импортируется по следующем пути:"
```python
from stairs-<...>.my_module.my_job import MyJobName
```
"В его методе `run` принимаются следуюшие параметры:"
- `param_1: int` - описание параметра
- `param_2: MyArgModel` - описание параметра, путь импорта `from stairs-<...>.my_module.args import MyArgModel`
  - `field_1: list[float]` - описание поля модели
  - ...

"Метод `run` записывает в хранилище результат в формате модели" `MyResultModel`, "она импортируется по следующему пути:"
```python
from stairs-<...>.my_module.result import MyResultModel
```
"Описание полей модели:"
- `field_1: str` - описание поля
- `field_2: list[MyInnerResultModel]` - описание поля, путь импорта `from stairs-<...>.my_module.result import MyInnerResultModel`
  - `field_1: list[float]` - описание поля модели
  - ...
Note: каждsq Job наследник описывается в отдельном подразделе интеграции
## Другой раздел
Какой-то текст