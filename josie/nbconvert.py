# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import re

from dataclasses import dataclass
from urllib.parse import quote

from nbconvert.exporters import TemplateExporter
from nbconvert.preprocessors import (
    ExecutePreprocessor,
    Preprocessor,
    TagRemovePreprocessor,
)


@dataclass
class RegexReplace:
    """A :mod:`dataclass` used to store regex patterns and their
    replacements"""

    regex: str
    replace: str


class SkipPreprocessor(Preprocessor):
    """A Preprocessor that removes cell tagged as "skip" """

    def preprocess(self, nb, resources):
        # Filter out cells that meet the conditions
        nb.cells = [
            cell
            for index, cell in enumerate(nb.cells)
            if self.check_cell_conditions(cell, resources, index)
        ]

        return nb, resources

    def check_cell_conditions(self, cell, resources, index):
        try:
            if cell.metadata.slideshow.slide_type == "skip":
                return False
            else:
                return True

        except AttributeError:
            return True


class CleanOutputPreprocessor(TagRemovePreprocessor):
    """A pre-processor that removes cells tagged with ``remove_output``"""

    remove_all_outputs_tags = set(["remove_output"])
    remove_cell_tags = set(["skip_conversion"])


class BinderBadgePreprocessor(Preprocessor):
    """This preprocessor adds a BinderBadge on top of the notebook"""

    BINDER_URL = "https://mybinder.org"

    def preprocess(self, nb, resources):
        filepath = quote(
            "examples/" + resources["metadata"]["name"] + ".ipynb", safe=""
        )

        # Add the Binder Badge on top of the first cell
        badge = (
            f"[![binder]({self.BINDER_URL}/badge_logo.svg)]"
            f"({self.BINDER_URL}/v2/gl/rubendibattista%2Fjosiepy/master/"
            f"?filepath={filepath})"
        )

        cell0 = nb.cells[0]
        cell0.source = badge + "\n" + cell0.source

        return nb, resources


class MathFixPreprocessor(Preprocessor):
    """This preprocessor fix the markdown for the math formulas using
    the :mod`recommonmark` notation
    """

    regexs = [
        RegexReplace(
            regex=r"\\begin\{[a-zA-Z]+\}([\s\S]*?)\\end\{[a-zA-Z]+\}",
            replace=r"```math\n \1 \n```",
        ),
        RegexReplace(regex=r"\$\$\n([ \S]+)\n\$\$", replace=r"```math\n \1 \n```"),
        RegexReplace(regex=r"\$(.*)\$", replace=r"`$ \1 $`"),
    ]

    def preprocess_cell(self, cell, resources, index):
        for reg_repl in self.regexs:
            cell.source = re.sub(reg_repl.regex, reg_repl.replace, cell.source)

        return cell, resources


class MplInlinePreprocessor(MathFixPreprocessor):
    """This preprocessors replaces the magic command for matplolib to
    :code:`inline`.
    """

    regexs = [RegexReplace(regex=r"(%matplotlib) .*", replace=r"\1 inline")]


class MdBinderExporter(TemplateExporter):
    """A :mod:`nbconvert` exporter that exports Notebooks as markdown files
    with a Binder badge on top ready to be used in Sphinx documentation
    """

    BINDER_URL = "https://mybinder.org"

    file_extension = ".md"
    template_file = "markdown/index.md.j2"

    ExecutePreprocessor.timeout = -1

    preprocessors = [
        MplInlinePreprocessor,
        ExecutePreprocessor,
        SkipPreprocessor,
        CleanOutputPreprocessor,
        MathFixPreprocessor,
        BinderBadgePreprocessor,
    ]
