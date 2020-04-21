from __future__ import annotations

import re

from dataclasses import dataclass

from nbconvert.exporters import TemplateExporter


class RSTBinderExporter(TemplateExporter):
    """ A :mod:`nbconvert` exporter that exports Notebooks as rst files with
    a Binder badge on top """

    BINDER_URL = "https://mybinder.org"

    raw_template = rf"""
    {{% extends 'rst.tpl'%}}
    {{% block header %}}
    {{% set filename = resources.metadata.name + '.ipynb' %}}
.. image:: {BINDER_URL}/badge_logo.svg
   :target: {BINDER_URL}/v2/gh/rubendibattista/josiepy/master/{{{{ filename | urlencode | replace("/", "%2F") }}}}
    {{% endblock header %}}
    """  # noqa: 501

    file_extension = ".rst"


@dataclass
class RegexReplace:
    regex: str
    replace: str


def cleanup(input, **kwargs):
    """ A Jinja filter that """

    # This is a list of stuff we want to parse out
    regexs = [
        # Remove magic stuff
        RegexReplace(r"(get_ipython\(\).*)", r"#\1"),
        # If there's a comment to initalize matplotlib with headless backend
        # that means even if matplotlib is not explicitly imported in the
        # notebook it is probably called under the hoods. Let's uncomment it
        RegexReplace(r"# *(import matplotlib; matplotlib.use\(.*\))", r"\1"),
        # Inject matplotlib backend for headless run
        RegexReplace(
            r"(import matplotlib\.pyplot as plt)",
            r"import matplotlib\nmatplotlib.use('SVG')\n\1",
        ),
        # Inject matplotlib backend for headless run when there's just the
        # magic command
        RegexReplace(
            r"(.*get_ipython.*matplotlib.*)",
            r"import matplotlib\nmatplotlib.use('SVG')\n#\1",
        ),
    ]

    lines = input.splitlines(True)
    output = ""
    for line in lines:
        new_line = None
        for reg_repl in regexs:
            if re.match(reg_repl.regex, line):
                new_line = re.sub(reg_repl.regex, reg_repl.replace, line)

        if new_line:
            output += new_line
        else:
            output += line

    return output


class NoMagicPythonExporter(TemplateExporter):

    raw_template = r"""
    {% extends 'python.tpl'%}
    {% block input %}
      {{ super() | cleanup }}
    {% endblock input %}
    """

    filters = {"cleanup": cleanup}

    file_extension = ".py"
