# Established coding standard
## Python3 project
Interpreter: Python 3.10 \
Style guide: based on PEP 8, official coding standard. \
Projekt packages: requirements.txt
## Indentation
4 spaces per indentation level, tabs should solely to remain consistent with code that is already indented with tabs.

## Closing brace/bracket/parenthesis on multiline constructs
```python
EXAMPLE = (
    'foo',
    'bar',
    'two'
)
```
## Spacing
Between function: 2 lines.
```python
def foo():
    pass


def bar():
    pass
```
Line comments: 2 spaces.
```python
a = 10  # Comment
```
Method definitions in classes: 1 or 2 lines.
```python
class OneClass:
    def foo(self):
        pass
    
    def bar(self):
        pass
```

## Naming
**Classes**: CapWords
```python
class MyClass:
    pass
```
**Global variables**: snake_case.
```python
my_variable = 10
```
**Functions**: snake_case.
```python
def my_function(one_var):
    pass
```
**Constants**: ALL_CAPITAL.
```python
ONE_CONSTANT = 10
```
**Files**: snake_case.
```
my_new_file.py
```
## Line length
Limit to 120 characters.

## Imports
Separate lines divided into groups, always on top of the file.\
Preferred order: 1. Standard library, 2. Third part libraries, 3. Local packages
```python
import os

import cv2
import numpy

from my_pkg import *
```
## Quotes
Preferred: double quote system.
```python
a = "abc"
d["f"] = 1
```
