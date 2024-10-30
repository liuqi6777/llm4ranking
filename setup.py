from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

setup(
    name='llm4ranking',
    version='0.0.1',
    packages=find_packages("src"),
    package_dir={'': 'src'},
    url='https://github.com/liuqi6777/llm4ranking',
    license='MIT',
    author='Qi Liu',
    author_email='qiliu6777@gmail.com',
    description='',
    python_requires='>=3.10',
    install_requires=[
        'accelerate>=1.0.1',
        'transformers>=4.46.0',
    ]
)