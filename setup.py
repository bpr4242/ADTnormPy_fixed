from setuptools import setup, find_packages

VERSION = '1.0'
DESCRIPTION = 'Python Wrapper for yezhengSTAT/ADTnorm'

setup(  name="adtnormpy_brett", 
        version=VERSION,
        author="Daniel Caron and Brett R",
        author_email="<bpr5bf@virginia.edu>",
        description=DESCRIPTION,
        packages=find_packages(),
        url = 'https://github.com/bpr4242/ADTnormPy_fixed/',
        keywords=['ADTnorm_brett'],
	    install_requires=['pandas','numpy','anndata','mudata','rpy2'],
        extras_require= {'pytest':['pytest']},
      classifiers= [])
