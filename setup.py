import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup( # Finally, pass this all along to distutils to do the heavy lifting.
    name             = 'ReinforcementLearning',
    version          = '0.0.1',
    description      = 'Some reinforcement learning codes',
    author           = 'Marco Picione',
    author_email     = 'marcopicione98@gmail.com',
    url              = 'https://github.com/MarcoPicione/ReinforcementLearning',
    install_requires = read('requirements.txt').splitlines(),
    package_dir      = {'': '.'},
    packages         = [''],
    python_requires  = '>=3.10', 
    zip_safe         = False,
)