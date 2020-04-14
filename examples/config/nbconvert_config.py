import re

c = get_config()

# This is a list of magics that we do not want in our output scripts
magics_regexs = [r"get_ipython\(\).*"]


def comment_magics(input, **kwargs):
    lines = input.splitlines(True)
    output = ""
    for line in lines:
        new_line = ""
        for regex in magics_regexs:
            if re.match(regex, line):
                new_line = new_line + "#" + line
        if new_line:
            output = output + new_line
        else:
            output = output + line
    return output


# Export all the notebooks in the current directory to the sphinx_howto format.
c.NbConvertApp.notebooks = ["*.ipynb"]
c.Exporter.filters = {"comment_magics": comment_magics}

# https://github.com/ipython/ipython/issues/3707/#issuecomment-327971946
