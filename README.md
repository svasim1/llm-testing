# LLM Testing

## Description

This project is designed to help me learn about Large Language Models, with a specific focus on Retrieval-Augmented Generation (RAG) models.

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

Also, rename the `.env.sample` file and fill in the required fields.

### Usage

Run whichever script you want to test, for example:

```python
python main-openai.py
```

Note: remove `storage/`, or whatever you have set `PERSIST_DIR` to in `.env`, everytime you update the `data/` folder.

### Status

**main-gpt.py** - kind of working, but not very intelligent

**main-llama.py** - kind of working, but not as intended

**main-openai.py** - working with context and everything, it can communicate through an API

### Documentation

[LlamaIndex using API](https://docs.llamaindex.ai/en/stable/getting_started/starter_example/)

[LlamaIndex local](https://docs.llamaindex.ai/en/stable/getting_started/starter_example_local/)
