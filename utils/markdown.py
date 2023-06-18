import re

from markdownify import markdownify as md


def lang_callback(el):
    lang = el['class'][0] if el.has_attr('class') else None

    if not lang is None:
        lang = lang.split("-")[-1]
    return lang


def html2md(text):
    text = md(text, code_language_callback=lang_callback)
    text = re.sub(r"\n\s*\n", "\n\n", text).strip()
    # return text.encode('utf-8', 'replace').decode()
    return text
