import io
from setuptools import find_packages, setup
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    reqs = []
    for line in f:
        line = line.strip()
        reqs.append(line.split('==')[0])

setup(
    name='angt',
    version='0.0.1',
    author="Yeonchan Ahn",
    author_email="acha21@europa.snu.ac.kr",
    description="Ahn's nlg tools",
    long_description=long_description,
    packages=find_packages(exclude=['docs', 'tests*']),
    python_requries=">=3.6",
    install_requires=reqs,
    package_data={"": ['angt/data/cmr/*.txt']}
)