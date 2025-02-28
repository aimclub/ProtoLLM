from pathlib import Path

from lxml import etree


def _get_omml2mml_transformation() -> etree.XSLT:
    omml2mml_file = Path(__file__).parent / "xsl/omml2mml/OMML2MML.XSL"
    omml2mml = etree.XSLT(etree.parse(omml2mml_file))
    return omml2mml


def _get_mml2tex_transformation() -> etree.XSLT:
    mml2tex_file = Path(__file__).parent / "xsl/mml2tex/mmltex.xsl"
    mml2tex = etree.XSLT(etree.parse(mml2tex_file))

    return mml2tex
