# Introduction

**What**

This document is both a template and guideline for a Python package. It is meant to be used as a starting point for new packages. It is not meant to be used as a package itself.

**Why**

It is important to follow guidelines and conventions when writing code. This will make it easier for others to understand your code and use it. It will also make it easier for you to maintain your code. This document will help you follow those guidelines and conventions.

**Important :warning: ️**

> :warning: This is a very important section. Please read it carefully.

NTNU has certain guidelines for licensing that needs to be followed. Your license needs may vary depending on your use case. The choice of the license **must be** agreed upon between the _contributing parties_ (Advisors, students, innovation center, data provides, industry partners). NTNU encouranges the use of Open Source licenses. You **must** engage in a discussion before _publishing_ your code with the _contributing parties_.
Please refer to NTNU wiki pages for more information.
- https://i.ntnu.no/wiki/-/wiki/English/Open+source
- https://i.ntnu.no/wiki/-/wiki/English/Guidelines+for+policy+for+Open+Science
- https://commission.europa.eu/about-european-commission/departments-and-executive-agencies/informatics/open-source-software-strategy_en

<details>
<summary>Publication Check list</summary>

:warning: Follow this checklist before publishing your code. If you fail to meet the requirements, do not publish your code. Please contact your advisor for help.

- [ ] The license to this software has been decided upon.
- [ ] Intellectual property rights has been considered (Patents, copy right statements, etc.)
- [ ] Necessary permissions has been obtained to publish the data that comes with this software.
- [ ] The grants (if exist) that funded this software has been acknowledged.
- [ ] Author information and license header is included in the code.
- [ ] The code is documented and the documentation follows the convention given in this document.
- [ ] This document has been read and understood.

</details>

<hr/>

<details>
<summary>Licensing</summary>

## Licensing

Licensing is an important step of the publishment process. It is important to choose the right license for your project. You can read more about it [here](https://choosealicense.com/).

NTNU favors the Open Source publication. **Please refer to [here](https://i.ntnu.no/wiki/-/wiki/English/Guidelines+for+policy+for+Open+Science)** for more information.

The head of your files _at least_ should include the following information.

```python
# Author: <Your name>, <Your e-mail>
# License: <License name>
# Copyright (C) <year>: <copyright holder>, <place>
```

If, for example, [GPLv3](https://choosealicense.com/licenses/gpl-3.0/) is chosen as the license, the header should look like this.

<dl><dd><dl><dd><dl><dd><dl><dd>
<details style="margin-left: 4em">
<summary>GPLv3 License Header for hypothetical project</summary>

```python
# This code is part of <Your project name>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# !Disclaimer: Below paragraph is an absolute nonsense. It is here to show you
#              what to include in your license header. You should replace it
#              with your own description.
# This code models the whole universe through a single function. It is intended
# to be used as a starting point for other universes, hence opening the doors to
# the multiverse.
#
# Author: Johny The John Junior, john.johny@ntnu.no
# Copyright (C) 2023 Johny The John Junior
```
</details>

<details style="margin-left: 4em">
<summary>GPLv3 License Header for PhD project</summary>

:warning: Notice the change in copyright holder.
```python
# This code is part of <Your project name>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# !Disclaimer: Below paragraph is an absolute nonsense. It is here to show you
#              what to include in your license header. You should replace it
#              with your own description.
# This code models the whole universe through a single function. It is intended
# to be used as a starting point for other universes, hence opening the doors to
# the multiverse.
#
# Author: Johny The John Junior, john.johny@ntnu.no
# Copyright (C) 2023, NTNU, Trondheim
```
</details>

<details style="margin-left: 4em">
<summary>GPLv3 License Header for Master's project</summary>

:warning: Notice the change in copyright holder.
```python
# This code is part of <Your project name>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# !Disclaimer: Below paragraph is an absolute nonsense. It is here to show you
#              what to include in your license header. You should replace it
#              with your own description.
# This code models the whole universe through a single function. It is intended
# to be used as a starting point for other universes, hence opening the doors to
# the multiverse.
#
# Author: Johny The John Junior, john.johny@ntnu.no
# Copyright (C) 2023, <discuss it with your supervisor>, Trondheim
```
</details>
</dd></dl></dd></dl></dd></dl></dd></dl>

:warning: Do not forget the include the `LICENSE.txt` file in your repository.

:warning: I can not stress this enough. **Please** read the [NTNU guidelines](https://i.ntnu.no/wiki/-/wiki/English/Open+source) for more information, **Please** refer to your supervisor and the innovation office.

</details>

<hr/>

<details>
<summary>Code Quality</summary>

## Code Quality

### Python code conventions

All python code must follow the PEP-8 convention. You can read more about it [here](https://www.python.org/dev/peps/pep-0008/).

It is often tedious to follow the PEP-8 convention manually. There are tools that can help you with that.
For example, autopep8 can automatically format your code to follow the PEP-8 convention. You can read more about it [here](https://github.com/hhatto/autopep8). You can also install it as a [Visual Studio Code](https://code.visualstudio.com/) [extension](https://marketplace.visualstudio.com/items?itemName=ms-python.autopep8).

Visual Studio Code is a highly customizable open source code editor. It is available for Windows, Mac, and Linux. It is highly recommended that you use it for your development. It can be configured through JSON files. If you attach the following, you can have automatic formatting on save.

```json
{
    ..., // Other settings
    "[python]": {
        "editor.formatOnType": true,
        "editor.defaultFormatter": "ms-python.autopep8",
    },
    ... // Other settings
}
```

> When in doubt, **please** create a github issue and ask for help.

### Development process

It is important to know the aim of the project. The procedure that you may follow can change depending on it. For example, if you are improving a code base that already exists, you may want to create a git branch for your changes. If you are starting a new project, you may want to create a new repository.

If you are conducting a research based on a previous work, you need to know if you are to improve it or just _consume_ it as a library. If you are using it as a library, simply consuming it rather than improving it, you should not create a new branch in that repository. You should simply install it as a dependency in your project. If you are improving it, you should create a new branch and make your changes there. You should then create a pull request to merge your changes to the main branch.

### Using external libraries

You may use the external libraries as you please but there are some important things to consider.

- Try to reduce the number of external libraries that you use. Each library that you use is a dependency that you need to maintain.
- Try **not** to use libraries that are not maintained anymore. You can check the last commit date of a library on GitHub. If it is not maintained anymore, you may want to find an alternative.
- Try **not** to use libraries that are not popular. If a library is not popular, it may not be maintained anymore. You can check the number of stars and forks on GitHub to see how popular a library is.
- Try to use a package manager. You may use [Conda](https://docs.conda.io/en/latest/) if you must. Aim to use only [`pip`](https://pip.pypa.io/en/stable/) if you can. You can read more about the differences between the two [here](https://www.anaconda.com/blog/understanding-conda-and-pip).
- Write all the external dependencies to `requirements.txt` file. You can use `pip freeze > requirements.txt` to generate the file.
    > :warning: If you are not using a virtual environment `pip freeze` will include all the packages that are installed in your system. You should use a virtual environment to generate the `requirements.txt` file.
    > To read more about virtualenvs, you can read [this](https://docs.python.org/3/tutorial/venv.html) and [this](https://realpython.com/python-virtual-environments-a-primer/).

### Tests

:smile: _[So it goes](https://en.wikipedia.org/wiki/Kurt_Vonnegut)_

</details>

<hr/>

<details>
<summary>Documentation</summary>

## Documentation

### Quick start



1. Use docstrings to document your code.
    1. Define inputs, outputs, and exceptions.
    2. When proper, include examples using `.. code-block:: python` lines
    3. When proper, include mathematical equations using `.. math::` lines
    4. Use [sphinx conventions](https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html) for formatting.
2. Write separate documentation files for other non-code related documentation.
    1. Use Markdown syntax.
    2. Place the files in the `docs` folder.
3. When in doubt, **please** create a github issue and ask for help.

### Conventions

Make yourself familiar with Markdown. It is a simple markup language that is used to write documentation. You can read more about it [here](https://www.markdownguide.org/basic-syntax/).

Use [sphinx conventions](https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html) for docstrings. This will allow us to use [Sphinx](http://www.sphinx-doc.org/en/master/) to generate documentation.

For example, say, you wrote a function called `amazing` that takes two arguments `a` and `b` and returns `c`. You should document it as follows, and it can be parsed by Sphinx. It is a good example of how to document your code.

```python
def amazing(a, b):
    r"""
    This function does something amazing.

    .. math::

        c = a + b

    .. code-block:: python

        >>> # A usage example for the amazing function.
        >>> amazing(1, 1)
        2

    :param a: The first argument.
    :type a: int
    :param b: The second argument.
    :type b: int
    :return: The result of the amazing operation.
    :rtype: int
    :raises ValueError: If the arguments are not equal.
    :raises TypeError: If the arguments are not integers.
    """
    if not isinstance(a, int) or not isinstance(b, int):
        raise TypeError("Arguments must be integers.")
    if a != b:
        raise ValueError("Arguments must be equal.")
    return a + b
```


### Documentation Generation

> ℹ️ _You do not need to go over this by yourself. Everything should be fine as long as you follow the conventions above._

Your code documentation should live in the code. This convention is called docstrings. You can read more about it [here](https://www.python.org/dev/peps/pep-0257/). There are parsers that can read these docstrings and turns them into automated documentation. One of the most popular is [Sphinx](http://www.sphinx-doc.org/en/master/). Below is a quick guide on how to get started with Sphinx.

Install documentation requirements
```bash
pip install -r requirements.docs.txt
```

Initialize the documentation
```bash
# Say [y] to separate source and build directory
sphinx-quickstart
```

Add the following content to the `conf.py` file to add additional functionality.
_Some of those variables already exist in the file. You should modify them instead of adding them again._
```python
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    'sphinx.ext.imgmath',
    'sphinx.ext.githubpages',
    'sphinx_rtd_theme',
    'sphinx.ext.autosectionlabel',
    'myst_parser',
    'hoverxref.extension',
]

myst_enable_extensions = [
    "amsmath",
    "dollarmath"
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}
```

Build and test the generated documentation
```bash
# Build the Api Docs
sphinx-apidoc.exe -f -o .docs/source ./src/template_package/
# Build the documentation
sphinx-build ./docs/source ./docs/build
# Test to see if it works
python -m http.server 8080
```

</details>

<hr/>

<details>
<summary>Versioning</summary>

## Versioning

**Why**

Research projects are often long term projects. It is important to keep track of the changes that you make to your code. It is also important to keep track of the changes that others make to your code. This is where versioning comes in.

**What**

We use [`git`](https://git-scm.com/) for versioning. You can read more about it [here](https://git-scm.com/book/en/v2/Getting-Started-About-Version-Control).

However, working with Git can be a little bit confusing. You should spend some
time learning it. There are some graphical user interfaces and IDE integrations
which can make it easier to work with Git. You can find a list of them
[here](https://git-scm.com/downloads/guis).

**Warning** :warning:

Most common mistake when using git is to not ignoring the files that should not be tracked. You can _ignore_ the files that should not be on the repository by adding them to the `.gitignore` file. You can read more about it [here](https://git-scm.com/docs/gitignore).
This is an important subject to prevent you from accidentally sharing your private information. For example, you may have a file that contains your password. You should not share this file with others. You should add it to the `.gitignore` file to prevent it from being tracked by git.

</details>

<hr/>

<details>
<summary>How to use this template</summary>

## How to use the template

1. Rename all the references to `template_package` to your package name.
    1. Modify the `pyproject.toml` contents
        - Change the project name
        - Change the package name
        - Change the author name and author e-mail
    2. Change the name of the `src/template_package` directory
2. Clean up the `src/template_package` directory content.
3. Read, understand, and remove the contents of the `docs` directory.
    - Important file is the `docs/source/conf.py` file.
4. Remove this `starthere.md` document since it does not belong to your package.

</details>

<hr/>

<details>
<summary>Help</summary>

## Help

When in doubt, **please** create a github issue and ask for help. Someone will get back to you as soon as possible.

If you use GitHub to ask your questions, others can benefit from the answers as well.

</details>

