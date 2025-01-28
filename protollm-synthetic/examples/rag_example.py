import os
import os
import json
import logging
from protollm_synthetic.synthetic_pipelines.chains import RAGChain
from protollm_synthetic.utils import Dataset, VLLMChatOpenAI
import asyncio

import logging
from protollm_synthetic.synthetic_pipelines.chains import RAGChain
from protollm_synthetic.utils import Dataset, VLLMChatOpenAI
import asyncio


# Сохраняем набор данных 
# Сохраняем набор данных 
texts = [
    """Формирование Стратегии 2030 осуществлялось на основе анализа устойчивых тенденций социально-экономического развития Санкт-Петербурга, а также с учетом результатов социально-экономического развития Санкт-Петербурга в 2012-2013 годах.
Представленная в Стратегии 2030 система целей социально-экономического развития Санкт-Петербурга структурирована по четырем уровням: генеральная цель определяет 4 стратегических направления, в рамках которых сформированы 17 стратегических целей, исходя из содержания которых сформулированы программно-целевые установки. Всего в Стратегии 2030 сформулированы 114 целей социально-экономического развития Санкт-Петербурга различных уровней, каждой из которых, кроме генеральной, соответствуют целевые показатели, характеризующие степень их достижения.
Генеральной целью Стратегии 2030 является обеспечение стабильного улучшения качества жизни горожан и повышение глобальной конкурентоспособности Санкт-Петербурга на основе реализации национальных приоритетов развития, обеспечения устойчивого экономического роста и использования результатов инновационно-технологической деятельности.
""",
    """Задачи государственной программы
Обеспечение приоритета профилактики в сфере охраны здоровья и развития первичной медико-санитарной помощи.
Повышение эффективности оказания специализированной, включая высокотехнологичную, медицинской помощи, скорой, в том числе скорой специализированной, медицинской помощи, медицинской эвакуации и паллиативной медицинской помощи.
Развитие и внедрение инновационных методов диагностики, профилактики и лечения.
Повышение эффективности службы родовспоможения и детства.Развитие медицинской реабилитации населения и совершенствование системы санаторно-курортного лечения, в том числе детей.
Обеспечение медицинской помощью неизлечимых больных, в том числе детей. Обеспечение системы здравоохранения высококвалифицированными и мотивированными кадрами."""
    """Региональные проекты, реализуемые в рамках государственной программы. Развитие экспорта медицинских услуг (город федерального значения Санкт-Петербург)
Борьба с онкологическими заболеваниями (город федерального значения Санкт-Петербург)
Развитие системы оказания первичной медико-санитарной помощи (город федерального значения Санкт-Петербург)
Обеспечение медицинских организаций системы здравоохранения квалифицированными кадрами (город федерального значения Санкт-Петербург)
Борьба с сердечно-сосудистыми заболеваниями (город федерального значения Санкт-Петербург)
Модернизация первичного звена здравоохранения Российской Федерации (город федерального значения Санкт-Петербург)
Старшее поколение (город федерального значения Санкт-Петербург)
Создание единого цифрового контура в здравоохранении на основе единой государственной информационной системы здравоохранения (ЕГИСЗ) (город федерального значения Санкт-Петербург)
Развитие детского здравоохранения, включая создание современной инфраструктуры оказания медицинской помощи детям (город федерального значения Санкт-Петербург)
Формирование системы мотивации граждан к здоровому образу жизни, включая здоровое питание и отказ от вредных привычек (город федерального значения Санкт-Петербург)
""",
    """Ожидаемые результаты реализации государственной программы
К 2029 году должен сложиться качественно новый уровень состояния жилищной сферы, характеризуемый следующими целевыми ориентирами:
оказание государственного содействия в улучшении жилищных условий в форме предоставления социальных выплат за счет средств бюджета Санкт-Петербурга и федерального бюджета в отношении 26113 петербургских семей;
расселение 6,08 тыс.кв.м аварийного жилищного фонда, признанного таковым до 01.01.2017, в рамках реализации первого этапа расселения аварийного жилищного фонда, и по второму этапу расселения аварийного жилищного фонда, признанного таковым в период с 01.01.2017 до 01.01.2022, переселение из него 0,38 тыс. человек и расселение 1,15 тыс.кв.м аварийного жилищного фонда, признанного таковым после 01.01.2017, переселение из него 0,113 тыс. человек;
формирование государственного жилищного фонда Санкт-Петербурга общей площадью 602,67 тыс.кв.м для предоставления жилых помещений 10,6 тыс. семей;
проведение капитального ремонта общего имущества по необходимым видам работ, включая мероприятия в области энергосбережения и повышения энергетической эффективности, в 81,5% многоквартирных домов от общего количества домов, включенных в региональную программу капитального ремонта общего имущества в многоквартирных домах в Санкт-Петербурге (далее - региональная программа);
повышение степени удовлетворенности населения Санкт-Петербурга уровнем жилищно-коммунального обслуживания до 67%;
достижение уровня доступности оплаты жилищно-коммунальных услуг гражданами.
""",
    """Расселение многоквартирных домов, признанных после 01.01.2017 аварийными и подлежащими сносу или реконструкции, осуществляется на основании распоряжений ЖК о признании многоквартирных домов аварийными и подлежащими сносу или реконструкции. С 01.01.2017 в Санкт-Петербурге были признаны аварийными и подлежащими сносу или реконструкции, а также подлежащими расселению 48 многоквартирных домов, в которых проживало 190 семей (556 человек), из них по состоянию на 01.12.2023 завершено расселение 31 аварийного многоквартирного дома, 157 семей (443 человека) переселены в благоустроенные жилые помещения.
В целях обеспечения социальной поддержки детей-сирот и детей, оставшихся без попечения родителей, данная категория подлежит обеспечению специализированными жилыми помещениями. Детям-сиротам как не имеющим закрепленного жилья, так и тем, у кого такое жилое помещение есть, однако проживание в нем по ряду обстоятельств невозможно, предоставляются отдельные квартиры по договорам найма специализированного жилищного фонда на пять лет. Жилищный фонд для этих целей сформирован из квартир в домах бюджетного строительства, приобретаемых в собственность Санкт-Петербурга, а также квартир свободного жилищного фонда районов Санкт-Петербурга. В период с 2021 по 2023 год 2656 детей-сирот обеспечено отдельными квартирами специализированного жилищного фонда Санкт-Петербурга.
""",
    """Ожидаемые результаты реализации государственной программы
Созданы условия для повышения уровня производительности труда в промышленности и роста конкурентоспособности предприятий;
осуществлено ускоренное развитие высокотехнологичного сектора промышленности Санкт-Петербурга;
обеспечено сохранение позиций Санкт-Петербурга как одного из ведущих промышленных регионов Российской Федерации, увеличение объема отгруженной продукции и ее доли в общем объеме отгруженной продукции по стране до 5,5 процентов;
обеспечено повышение инновационной активности предприятий; население Санкт-Петербурга обеспечено качественными и безопасными продуктами питания
""",
    """Ожидаемые результаты реализации государственной программы
Выполнение обязательств государства по социальной поддержке отдельных категорий граждан;
снижение бедности среди получателей мер социальной поддержки на основе расширения сферы применения адресного принципа ее предоставления;
расширение масштабов предоставления мер социальной поддержки отдельным категориям граждан в денежной форме;
сохранение размера оплаты труда социальных работников государственных учреждений социальной защиты населения Санкт-Петербурга на уровне не ниже 100% среднемесячного дохода от трудовой деятельности в Санкт-Петербурге;
удовлетворение потребностей всех категорий граждан в социальном обслуживании;
обеспечение поддержки и содействие социальной адаптации граждан, попавших в трудную жизненную ситуацию или находящихся в социально опасном положении;
создание прозрачной и конкурентной среды в сфере социального обслуживания населения;
рост рождаемости; преобладание семейных форм устройства детей, оставшихся без попечения родителей; расширение охвата детей-инвалидов социальным обслуживанием;
развитие конкуренции на рынке социального обслуживания; создание прозрачной и конкурентной системы государственной поддержки СО НКО; обеспечение эффективности и финансовой устойчивости СО НКО;
увеличение объемов социальных услуг, оказываемых негосударственными организациями, в том числе СО НКО;
расширение возможностей граждан пожилого возраста для социальной интеграции в общество; развитие рекреационных поселений в границах территорий ведения жителями Санкт-Петербурга садоводства для собственных нужд
"""
    """Социальная защита населения представляет собой систему правовых, экономических, организационных и иных мер, гарантированных государством отдельным категориям населения. Категории граждан - получателей социальной поддержки и (или) социальных услуг, виды и формы социальной поддержки и условия ее предоставления определены законодательством Российской Федерации, законодательством Санкт-Петербурга, иными нормативными правовыми актами.
Государственная политика Санкт-Петербурга в области социальной защиты населения формируется в соответствии с положениями Конституции Российской Федерации, в которой определено, что в Российской Федерации обеспечивается государственная поддержка семьи, материнства, отцовства и детства, инвалидов и пожилых граждан, развивается система социальных служб, устанавливаются государственные пенсии, пособия и иные гарантии социальной защиты. При этом Конституцией Российской Федерации установлено, что в совместном ведении Российской Федерации и субъектов Российской Федерации находятся вопросы социальной защиты, включая социальное обеспечение, защиты семьи, материнства, отцовства и детства; защиты института брака как союза мужчины и женщины; создания условий для достойного воспитания детей в семье, а также для осуществления совершеннолетними детьми обязанности заботиться о родителях.
На развитие института социальной защиты населения оказывает влияние ряд следующих факторов:
экономические (уровень и темпы экономического развития, занятость и доходы населения, состояние государственных финансов, уровень развития производительных сил);
демографические (сокращение рождаемости, увеличение продолжительности жизни);""",
    """Промышленность Санкт-Петербурга является основой экономики Санкт-Петербурга, главным источником доходов бюджета Санкт-Петербурга.
На долю промышленности в Санкт-Петербурге приходится 12,7 процента валового регионального продукта (по данным за 2022 год).
Вклад промышленности в формирование доходной части бюджета Санкт-Петербурга является наибольшим: промышленные предприятия по итогам 2023 года обеспечили 39,7 процента налоговых поступлений в бюджетную систему Российской Федерации, 15,0 процента поступлений в бюджет Санкт-Петербурга (по оценке КППИТ на основе данных Управления Федеральной налоговой службы по Санкт-Петербургу).
Развитие промышленности как базового сектора экономики оказывает влияние на различные аспекты социально-экономического развития региона, в том числе на доходы бюджета, занятость и уровень благосостояния населения, решение социальных задач, состояние потребительского рынка.
Одним из основных факторов обеспечения стратегической конкурентоспособности и необходимым условием устойчивого развития промышленности Санкт-Петербурга является наличие в Санкт-Петербурге значительного инновационного потенциала.
Санкт-Петербург находится в центре передового инновационного развития и на протяжении нескольких лет занимает лидирующие позиции в различных рейтингах, в том числе международных. Начиная с 2014 года Санкт-Петербург входит в тройку лидеров в Рейтинге инновационных регионов России, разработанном ассоциацией экономического взаимодействия субъектов Российской Федерации "Ассоциация инновационных регионов России" совместно с Министерством экономического развития Российской Федерации (далее - Рейтинг), и в 2018 году Санкт-Петербург в Рейтинге занял первое место.
Согласно ежегодному Рейтингу инновационного развития субъектов Российской Федерации, который проводится Институтом статистических исследований и экономики знаний Национального исследовательского университета "Высшая школа экономики" в рамках деятельности Российской кластерной обсерватории, Санкт-Петербург на протяжении ряда лет располагается в первой тройке инновационных регионов.
С учетом указанных предпосылок, целей и задач социально-экономического развития Санкт-Петербурга на долгосрочную перспективу инновационное развитие Санкт-Петербурга определено как одно из приоритетных направлений.
Эффективным сектором экономики, включающим в себя предприятия пищевой и перерабатывающей промышленности и предприятия сельского хозяйства, является агропромышленный комплекс Санкт-Петербурга, который призван решать одну из важнейших задач, - содействие обеспечению продовольственной безопасности Санкт-Петербурга.
Характеристики текущего состояния промышленности, инновационной деятельности и агропромышленного комплекса Санкт-Петербурга с указанием основных проблем приведены в соответствующих разделах подпрограмм государственной программы.
""",
    """Ожидаемые результаты реализации государственной программы
Созданы условия для повышения уровня производительности труда в промышленности и роста конкурентоспособности предприятий;
осуществлено ускоренное развитие высокотехнологичного сектора промышленности Санкт-Петербурга;
обеспечено сохранение позиций Санкт-Петербурга как одного из ведущих промышленных регионов Российской Федерации, увеличение объема отгруженной продукции и ее доли в общем объеме отгруженной продукции по стране до 5,5 процентов;
обеспечено повышение инновационной активности предприятий; население Санкт-Петербурга обеспечено качественными и безопасными продуктами питания"""
]

path = 'tmp_data/sample_data_city_rag.json'

data_dict = {'content': texts}

with open('tmp_data/sample_data_rag_spb.json', 'w', encoding='utf-8') as file:
    json.dump(data_dict, file, ensure_ascii=False)


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


path = 'tmp_data/sample_data_rag_spb.json'
dataset = Dataset(data_col='content', path=path)

qwen_large_api_key = os.environ.get("OPENAI_API_KEY")
qwen_large_api_base = os.environ.get("OPENAI_API_BASE")

logger.info("Initializing LLM connection")

llm=VLLMChatOpenAI(
        api_key=qwen_large_api_key,
        base_url=qwen_large_api_base,
        model="/model",
        max_tokens=2048,
    )

rag_chain = RAGChain(llm=llm)

logger.info("Starting generating")
asyncio.run(rag_chain.run(dataset, 
                          n_examples=5))

logger.info("Saving results")
path = 'tmp_data/sample_data_city_rag_generated.json'

# An alternative way to save data
# rag_chain.save_chain_output('tmp_data/sample_data_city_rag_generated.json')

df = rag_chain.data.explode('generated')
df['question'] = df['generated'].apply(lambda x: x['question'])
df['answer'] = df['generated'].apply(lambda x: x['answer'])
df = df[['content', 'question', 'answer']]

logger.info(f"Writing result to {path}")
df.to_json(path, orient="records")

logger.info("Generation successfully finished")
logger.info("Generation successfully finished")