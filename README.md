# LLM Testing

## Description.

This project is for me learning about Large Language Models, with a specific focus on Retrieval-Augmented Generation (RAG) models.

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

Also, rename the `.env.sample` file to `.env` and fill in the required fields.

### Usage

In the `app/` folder, start the Uvicorn server:

```bash
uvicorn main:app --reload
```

Note: remove `storage/`, or whatever you have set `PERSIST_DIR` to in `.env`, everytime you update the `data/` folder.

#### Docker

To run the project in a Docker container, configure `docker-compose.yml` according to the `docker-composer.yml.sample` and run:

```bash
docker compose up -d --build
```

Note that the database url is a bit tricky to set up depending on your system since it should be placed in a mirrored mount point. Read the `.env.sample` carefully.

### Status

**main.py** - working with context and everything, it can communicate through an API, can be run in a Docker container, and uses SQLite for storing user credentials used for authorizing.

### Todo

- [x] Add logging
- [x] Run in Docker container
- [x] Migrate to SQLite
- [x] Redo the chatbot code - read [this](https://docs.llamaindex.ai/en/stable/examples/llm/openai/).

  - To be able to send user information to OpenAI as stated in [OpenAI: Safety best practices](https://platform.openai.com/docs/guides/safety-best-practices)

- [x] Send user email to OpenAI
- [x] Split the code into modules
- [x] Split documents into chunks
  - [x] Redo logging in modules
  - [x] Test docker with modules
- [x] Custom Swagger UI for the API documentation
- [x] Add support for PDF, and multiple files
- [ ] Check the **Important information** section

### Documentation

[LlamaIndex using API](https://docs.llamaindex.ai/en/stable/getting_started/starter_example/)

### Important information

[OpenAI: Safety best practices](https://platform.openai.com/docs/guides/safety-best-practices)

[OpenAI: Europe Terms of Use](https://openai.com/policies/terms-of-use/)

[LlamaIndex: Building Performant RAG Applications for Production](https://docs.llamaindex.ai/en/stable/optimizing/production_rag/)
