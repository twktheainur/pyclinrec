from setuptools import setup, find_packages

setup(name='pyclinrec',
      version='0.0.1',
      description='A dictionary-based clinical entity-linking module',
      url='http://github.com/twktheainur/pyclinrec',
      author='Andon Tchechmedjiev',
      author_email='andon.tchechmedjiev@mines-ales.fr',
      license='MIT',
      packages=find_packages(include=['pyclinrec', 'pyclinrec.recognizer', 'pyclinrec.dictionary']),
      keywords='clinical nlp concept recognizer entity linking',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 2.8',
          'Topic :: Text Processing :: Linguistic',
      ],
      install_requires=[
          'regex==2020.10.15',
          'torch',
          'transformers',
          'pandas==1.1.3',
          'tqdm',
          'nltk',
          'requests',
          'Metafone'
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      include_package_data=True,
      zip_safe=False)
