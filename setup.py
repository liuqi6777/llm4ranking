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
        'torch>=2.1.0',
        'accelerate>=1.0.1',
        'transformers>=4.46.0',
        'datasets>=2.20.0',
        'openai>=1.0.0',
        'pytrec_eval>=0.5',
        'bm25s>=0.2.0',
        'ujson>=5.10.0',
        'peft>=0.12.0',
        'jinja2>=3.1.0',
    ],
    extras_require={
        'vllm': ['vllm>=0.6.0'],
    },
)
