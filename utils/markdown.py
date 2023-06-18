import re
from typing import List, Callable

from markdownify import markdownify as md


REG_CODE_BLOCK = re.compile('^(([ \t]*`{3,4})([^\n]*)([\s\S]+?)(^[ \t]*`{3,4}))', re.MULTILINE)
REG_CODE = re.compile('(?<=`)[^`\r\n]+(?=`)')
REG_LINK = re.compile('\[([^\[]+)\](\(.*\))')


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


def find_non_text_intervals(string):
    code_intervals = []
    for m in re.finditer(REG_CODE_BLOCK, string):
        code_intervals.append((m.start(), m.end()))
    for m in re.finditer(REG_CODE, string):
        code_intervals.append((m.start(), m.end()))
    for m in re.finditer(REG_LINK, string):
        code_intervals.append((m.start(), m.end()))
    return code_intervals


def replace_substring(s, newstring, start_index, end_index):
    return s[:start_index] + newstring + s[end_index:]


def apply_transformation_for_non_code(text: str, functions: List[Callable]):
    code_intervals = find_non_text_intervals(text)

    prev_end = 0
    out = ''
    for r in code_intervals:
        substr = text[prev_end:r[0]]
        prev_end = r[1]
        for f in functions:
            augmented_text = f(substr)
            out += augmented_text + text[r[0]:r[1]]

    for f in functions:
        augmented_text = f(text[prev_end:])
        out += augmented_text

    return out