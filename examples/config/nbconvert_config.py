import re

from dataclasses import dataclass

c = get_config()


@dataclass
class RegexReplace:
    regex: str
    replace: str


# This is a list of stuff we want to parse out
magics_regexs = [
    # Remove magic stuff
    RegexReplace(r"(get_ipython\(\).*)", r"#\1"),
    # Inject matplotlib backend for headless run
    RegexReplace(
        r"(import matplotlib\.pyplot as plt)",
        r"import matplotlib\nmatplotlib.use('SVG')\n\1",
    ),
]


def comment_magics(input, **kwargs):
    lines = input.splitlines(True)
    output = ""
    for line in lines:
        new_line = None
        for regex in magics_regexs:
            if re.match(regex.regex, line):
                new_line = re.sub(regex.regex, regex.replace, line)

        if new_line:
            output += new_line
        else:
            output += line

    return output


# Export all the notebooks in the current directory to the sphinx_howto format.
c.NbConvertApp.notebooks = ["*.ipynb"]
c.Exporter.filters = {"comment_magics": comment_magics}

# https://github.com/ipython/ipython/issues/3707/#issuecomment-327971946
