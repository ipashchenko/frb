try:
    from setuptools import setup
    setup
except ImportError:
    from distutils.core import setup
    setup


setup(
    name='frb',
    version='0.1',
    author='Alexander Kutkin',
    author_email='ikutkin@asc.rssi.ru',
    packages=['frb', 'tests'],
    scripts=[],
    url='https://github.com/akutkin/frb',
    license='LICENSE',
    description='Search FRB in Radioastron data',
    long_description=open('README.md').read(),
    install_requires=[
        "numpy >= 1.7.2",
        "scipy >= 0.12.0", 'astropy'
    ],)
