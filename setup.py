from setuptools import setup, find_packages

setup(
    name="smart_mrag",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain>=0.3.14",
        "langchain-community>=0.3.14",
        "langchain-openai>=0.3.0",
        "langchain-core>=0.3.29",
        "faiss-cpu>=1.9.0",
        "pymupdf>=1.25.1",
        "Pillow>=10.4.0",
        "pypdf>=5.1.0",
        "python-dotenv>=1.0.1",
        "openai>=1.59.6",
        "numpy>=1.26.2",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A smart Multi-modal RAG (Retrieval Augmented Generation) library for processing PDFs",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/smart_mrag",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
) 