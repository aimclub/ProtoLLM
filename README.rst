**ProtoLLM**

.. start-badges
.. list-table::
   :stub-columns: 1

   * - license
     - | |license|
   * - support
     - | |tg|
   * - languages
     - | |eng| |rus|
   * - mirror
     - | |gitlab|
   * - funding
     - | |ITMO| |SAI|
.. end-badges

**Proto-LLM** - это фреймворк с открытым исходным кодом для быстрого протипирования приложений,
основанных на больших языковых моделях (БЯМ). Он позволяет создавать ИИ-ассистентов,
основанных на применения агентных БЯМ, вызова функции, RAG и методов ансамблирования.


Основные функции
==================

- Быстрое создание прототипов информационно-поисковых систем на основе БЯМ с использованием технологии retrieval-augmented generation (RAG).
- Разработка и интеграция многофункциональных приложений с БЯМ, включая возможность подключения внешних сервисов и моделей через систему плагинов.
- Оптимизация производительности БЯМ путем реализации ансамблевых методов и мультиагентных подходов.
- Генерация сложных синтетических данных для дальнейшего обучения и улучшения БЯМ.
- Ускорение процесса разработки и внедрения систем, основанных на БЯМ, в различных прикладных областях.


Установка
=========

- Установщик пакетов для Python **pip**

Самый простой способ установить ProtoLLM - это использовать ``pip``:

.. code-block::

  $ pip install protollm

Модули с инструментами могут быть установлены отдельно:

.. code-block::
  $ pip install protollm-api

  $ pip install protollm-sdk


Структура проекта
=================

Последний стабильный релиз ProtoLLM находится в ветке `master <https://github.com/ITMO-NSS-team/ProtoLLM/tree/master>`__.

Репозиторий включает следующие директории:

* Пакет `protollm <https://github.com/ITMO-NSS-team/ProtoLLM/tree/master/protollm>`__ содержит основные модули. Это *ядро* фреймворка ProtoLLM;
* Пакет `protollm_tools <https://github.com/ITMO-NSS-team/ProtoLLM/tree/master/protollm_tools>`__ содержит дополнительные *инструменты* с тдельными зависимостями;
* Пакет `examples <https://github.com/ITMO-NSS-team/ProtoLLM/tree/master/examples>`__ содержит несколько *примеров*, с помощью которых вы можете начать знакомство с работой ProtoLLM;
* Все *модульные и интеграционные тесты* можно посмотреть в директории `test <https://github.com/ITMO-NSS-team/ProtoLLM/tree/master/test>`__;
* Исходники документации находятся в директории `docs <https://github.com/ITMO-NSS-team/ProtoLLM/tree/master/docs>`__.

Руководство по участию
======================

- Руководство по внесению вклада доступно в этом `файле <https://github.com/ITMO-NSS-team/ProtoLLM/blob/master/docs/source/contribution.rst>`__.

Благодарности
=============

Мы выражаем благодарность разработчикам фреймворка, а также участникам  научных конференций и
семинаров за ценные советы и предложения.

Поддержка
=========

Исследование проводится при поддержке `Исследовательского центра сильного искусственного интеллекта в промышленности <https://sai.itmo.ru/>`_
`Университета ИТМО <https://itmo.ru/>`_ в рамках мероприятия программы центра:
"Разработка фреймворка быстрого прототипирования приложений на основе больших языковых моделей "

Контакты
========
- `Институт ИИ, ИТМО <https://aim.club/>`_
- `Анна Калюжная <https://scholar.google.com/citations?user=bjiILqcAAAAJ&hl=ru>`_ (anna.kalyuzhnaya@itmo.ru)
- `Чат поддержки <https://t.me/protollm_helpdesk>`_

Статьи о решениях, основанных на ProtoLLM:
==========================================
- Kalyuzhnaya A. et al. LLM Agents for Smart City Management: Enhancing Decision Support Through Multi-Agent AI Systems //Smart Cities. – 2025. – Т. 8. – №. 1. – С. 19.
- Zakharov K. et al. Forecasting Population Migration in Small Settlements Using Generative Models under Conditions of Data Scarcity //Smart Cities. – 2024. – Т. 7. – №. 5. – С. 2495-2513.
- Kovalchuk M. A. et al. SemConvTree: Semantic Convolutional Quadtrees for Multi-Scale Event Detection in Smart City //Smart Cities. – 2024. – Т. 7. – №. 5. – С. 2763-2780.


.. |ITMO| image:: https://raw.githubusercontent.com/aimclub/open-source-ops/43bb283758b43d75ec1df0a6bb4ae3eb20066323/badges/ITMO_badge.svg
   :alt: Acknowledgement to ITMO
   :target: https://en.itmo.ru/en/

.. |SAI| image:: https://raw.githubusercontent.com/aimclub/open-source-ops/43bb283758b43d75ec1df0a6bb4ae3eb20066323/badges/SAI_badge.svg
   :alt: Acknowledgement to SAI
   :target: https://sai.itmo.ru/

.. |license| image:: https://img.shields.io/github/license/aimclub/ProtoLLM
   :alt: Licence for repo
   :target: https://github.com/aimclub/ProtoLLM/blob/master/LICENSE.md

.. |tg| image:: https://img.shields.io/badge/Telegram-Group-blue.svg
   :target: https://t.me/protollm_helpdesk
   :alt: Telegram Chat

.. |gitlab| image:: https://img.shields.io/badge/mirror-GitLab-orange
   :alt: GitLab mirror for this repository
   :target: https://gitlab.actcognitive.org/itmo-sai-code/ProtoLLM

.. |eng| image:: https://img.shields.io/badge/lang-en-red.svg
   :target: /README_en.rst

.. |rus| image:: https://img.shields.io/badge/lang-ru-yellow.svg
   :target: /README.rst
