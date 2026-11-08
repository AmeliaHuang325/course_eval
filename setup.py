from pathlib import Path

from setuptools import setup, find_packages

setup(
    name='course_eval',
    version='0.2.0',
    packages=find_packages(exclude=['tests']),
    python_requires='>=3.9',
    install_requires=[
        'pandas>=1.5',
        'numpy>=1.23',
        'matplotlib>=3.6',
        'seaborn>=0.12',
        'plotly>=5.0',
        'nltk>=3.8',
    ],
    extras_require={'test': ['pytest>=7']},
    author='Shuyang Huang',
    description='Tools for analyzing and visualizing course evaluations from Canvas LMS exports.',
    long_description=Path(__file__).with_name('README.md').read_text(encoding='utf-8'),
    long_description_content_type='text/markdown',
    license='MIT',
    keywords='course evaluation, canvas lms, education, data analysis, sentiment analysis',
    url='https://github.com/AmeliaHuang325/course_eval',
)
