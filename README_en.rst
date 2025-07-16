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

Intro
#####

**Proto-LLM** is an open-source framework for fast protyping of LLM-based applications.


Proto LLM features
==================
- Rapid prototyping of information retrieval systems based on LLM using RAG:
   Implementations of architectural patterns for interacting with different databases and web service interfaces;
   Methods for optimising RAG pipelines to eliminate redundancy.

- Development and integration of applications with LLM with connection of external services and models through plugin system:
   Integration with AutoML solutions for predictive tasks;
   Providing structured output generation and validation;

- Implementation of ensemble methods and multi-agent approaches to improve the efficiency of LLMs:
   Possibility of combining arbitrary LLMs into ensembles to improve generation quality, automatic selection of ensemble composition;
   Work with model-agents and ensemble pipelines;

- Generation of complex synthetic data for further training and improvement of LLM:
   Generating examples from existing models and data sets;
   Evolutionary optimisation to increase the diversity of examples; Integration with Label Studio;

- Providing interoperability with various LLM providers:
   Support for native models (GigaChat, YandexGPT, vsegpt, openrouter, etc.).
   Interaction with open-source models deployed locally.

Project Structure
=================

The latest stable release of ProtoLLM is in the `main branch <https://github.com/ITMO-NSS-team/ProtoLLM/tree/main>`__.

The repository includes the following directories:

* Package `protollm <https://github.com/ITMO-NSS-team/ProtoLLM/tree/main/protollm>`__  contains the main modules. It is the *core* of the ProtoLLM framework;
* Package `protollm_tools <https://github.com/ITMO-NSS-team/ProtoLLM/tree/main/protollm_tools>`__  contains side tools with specific dependensied;
* Package `examples <https://github.com/ITMO-NSS-team/ProtoLLM/tree/main/examples>`__ includes several *how-to-use-cases* where you can start to discover how ProtoLLM works;
* All *unit and integration tests* can be observed in the `test <https://github.com/ITMO-NSS-team/ProtoLLM/tree/main/test>`__ directory;
* The sources of the documentation are in the `docs <https://github.com/ITMO-NSS-team/ProtoLLM/tree/main/docs>`__ directory.

Packages `protollm-api <https://github.com/aimclub/protollm-api>`__ and `protollm-sdk <https://github.com/aimclub/protollm-sdk>`__ are available in external repositories.

Installation
============

- The simplest way to install ProtoLLM is using ``pip``:

.. code-block::

  $ pip install protollm

A standard installation of ProtoLLM includes the main package with dependencies.


- Modules with tools can be installed separately:

.. code-block::

  $ pip install protollm-api

  $ pip install protollm-sdk


Contribution Guide
==================

- The contribution guide is available in this `repository <https://github.com/ITMO-NSS-team/ProtoLLM/blob/main/docs/source/contribution.rst>`__.

Acknowledgments
===============

We acknowledge the contributors for their important impact and the participants of the numerous scientific conferences and
workshops for their valuable advice and suggestions.

Supported by
============

The study is supported by the Research `Center Strong Artificial Intelligence in Industry <https://sai.itmo.ru/>`_
of `ITMO University <https://itmo.ru/>`_ as part of the plan of the center's program
"Framework for rapid application prototyping based on large language models".


Contacts
========
- `AI Institute, ITMO <https://aim.club/>`_
- `Anna Kalyuzhnaya <https://scholar.google.com/citations?user=bjiILqcAAAAJ&hl=ru>`_ (anna.kalyuzhnaya@itmo.ru)
- `Helpdesk chat <https://t.me/protollm_helpdesk>`_

Papers about ProtoLLM-based solutions:
======================================
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
