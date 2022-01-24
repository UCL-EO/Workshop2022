import setuptools

with open("../README.md", "r") as fh:
    long_description = fh.read()

setup = {
    'url':'https://github.com/UCL-EO/Workshop2022',
    'version':'0.1',
    'name':'Workshop2022',
    'description':'UCL/NCEO/GSSTI Workshop, 2022',
    'author':'Prof. P. Lewis',
    'author_email':'p.lewis@ucl.ac.uk',
    'license':'GNU 3',
    'keywords':'Terrestrial Carbon: modelling and monitoring',
    'long_description':long_description,
    'long_description_content_type':"text/markdown",
    'packages':setuptools.find_packages(),
    'classifiers':[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU 3 License",
        "Operating System :: OS Independent",
    ],
    'python_requires':'>=3.6',
}


setuptools.setup(**setup) 

