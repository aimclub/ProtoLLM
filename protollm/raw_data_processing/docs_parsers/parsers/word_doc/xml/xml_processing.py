import io
from uuid import uuid4

from PIL import Image
from docx.text.paragraph import Paragraph
from lxml import etree

from protollm.raw_data_processing.docs_parsers.parsers.word_doc.xml.xml_tag import (
    XMLTag,
)


def _convert_to_latex(
    xml_element: etree.Element, parsing_config: "DocxParsingConfig"
) -> str:
    math_ml = parsing_config.omml2mml_transformation(xml_element).getroot()
    tex = parsing_config.mml2tex_transformation(math_ml)
    return str(tex)


def _extract_image_data(
    xml_element: etree.Element, parsing_config: "DocxParsingConfig"
) -> dict:
    rid = xml_element.find(".//" + XMLTag.blip.value).get(XMLTag.embed.value)
    image_element = parsing_config.document_relationships[rid].target_part
    image = Image.open(io.BytesIO(image_element.blob))
    # [Example Future Functionality] extracted_data = process_image(image)  # TODO: add image processing
    return {"image_filename": image_element.filename, "image": image}


def _parse_raw_xml_element(
    raw_xml_element: etree.Element, parsing_config: "DocxParsingConfig"
) -> tuple[list[str], dict]:
    """Parse raw xml element."""
    texts = []
    extracted_image_data = {}

    for child_element in raw_xml_element:
        match child_element.tag:
            case XMLTag.text:
                texts.append(child_element.text)
            case XMLTag.raw:
                child_texts, child_extracted_image_data = _parse_raw_xml_element(
                    child_element, parsing_config
                )
                texts.extend(child_texts)
                extracted_image_data.update(child_extracted_image_data)
            case XMLTag.image:
                if parsing_config.extract_images:
                    image_name = "image-" + str(uuid4())
                    texts.append(f"{{{image_name}}}")
                    extracted_image_data[image_name] = _extract_image_data(
                        child_element, parsing_config
                    )

    return texts, extracted_image_data


def process_paragraph_body(
    paragraph: Paragraph, parsing_config: "DocxParsingConfig"
) -> tuple[str, dict[str, dict]]:
    xml_paragraph = etree.fromstring(paragraph._element.xml)
    texts = []
    extracted_data = {"images": {}, "formulas": {}}
    for element in xml_paragraph:
        match element.tag:
            case XMLTag.raw:
                extracted_texts, extracted_image_data = _parse_raw_xml_element(
                    element, parsing_config
                )
                texts.extend(extracted_texts)
                extracted_data["images"].update(extracted_image_data)
            case XMLTag.math | XMLTag.math_paragraph:
                if parsing_config.parse_formulas:
                    formula_name = "formula-" + str(uuid4())
                    texts.append(f"{{{formula_name}}}")
                    extracted_data["formulas"][formula_name] = _convert_to_latex(
                        element, parsing_config
                    )

    return "".join(texts), extracted_data
