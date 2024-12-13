# LLM Testing

[![License](https://img.shields.io/github/license/svasim1/llm-testing)](https://github.com/svasim1/llm-testing/blob/main/LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/release/python-380/)

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Steps](#steps)
- [Usage](#usage)
- [Configuration](#configuration)
  - [Chatbot Behavior](#chatbot-behavior)
  - [API](#api)
  - [Data](#data)
  - [Database](#database)
- [Architecture](#architecture)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Project Overview

This project is for me learning about Large Language Models, with a specific focus on Retrieval-Augmented Generation (RAG) models.

## Installation

##### Prerequisites

- Docker (optional)
- Python 3.8+
- pip

##### Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/svasim1/llm-testing.git
   cd llm-testing
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Rename the `.env.sample` file to `.env` and fill in the required fields.

## Usage

In the `app/` folder, start the Uvicorn server:

```bash
uvicorn main:app --reload
```

Note: remove `storage/`, or whatever you have set `PERSIST_DIR` to in `.env`, everytime you update the `data/` folder.

##### Docker

To run the project in a Docker container, configure `docker-compose.yml` according to the `docker-composer.yml.sample` and run:

```bash
docker compose up -d --build
```

Note that the database URL is a bit tricky to set up depending on your system since it should be placed in a mirrored mount point. Read the `.env.sample` carefully.

## Configuration

To customize the project for your needs, To adapt the project to your needs, you can, among other things, change the following:

##### Chatbot Behavior

- You can change the LLM model in `chat.py` by changing the `model` parameter.
- You can change the "length" of the answer in `chat.py` by changing the `max_tokens` parameter.
- The context prompt for the chatbot is in `chat.py` in the `context_prompt` variable, but also the hardcoded message history in the `chatbot` function.

##### API

- You can customize the Swagger UI in `main.py` by changing the parameters in the `FastAPI` object.
- You can change the CORS settings in `main.py` in the middleware section.
- You can change the Question and Issue requirements in `main.py` in the `Question` and `Issue` classes.
- The authentication expiration time is in `main.py` in the `ACCESS_TOKEN_EXPIRE_MINUTES` constant.
- The limiter under `/chat` endpoint.

##### Data

- The `data/` folder contains the documents that the chatbot will use to answer questions. Everytim you update the documents, you need to remove `storage/`.
- You can change the chunk size in `data_processing.py` by changing the `chunk_size` parameter.

##### Database

- You can customize the database tables in `models.py`.

## Architecture

The project is structured as follows:

- `app`: Contains the main application code.
  - `main.py`: Sets up the FastAPI server and defines the endpoints.
  - `chat.py`: Contains the logic for interacting with the AI model.
  - `data_processing.py`: Handles the processing and storage of documents.
  - `models.py`: Defines the database models.
  - `database.py`: Contains functions related to user authentication and database interactions.
- `data/`: Contains the documents that the chatbot will use to answer questions.
- `storage/`: Directory to store chatbot data. Named after `PERSIST_DIR` in `.env`.
- `Dockerfile`: Defines the Docker image. Optional.
- `docker-compose.yml`: Defines the Docker services. Optional.

## Contributing

Contributions are welcome! Please read the [contributing guidelines](https://github.com/svasim1/llm-testing/blob/main/CONTRIBUTING.md) for more information.

## License

This project is licensed under the GPL-3.0 License. See the [LICENSE](https://github.com/svasim1/llm-testing/blob/main/LICENSE) file for more details.

## Acknowledgments

- [OpenAI](https://openai.com/) for providing the LLM model.
- [LlamaIndex](https://docs.llamaindex.ai/) for the RAG model implementation.
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework.
- [SQLAlchemy](https://www.sqlalchemy.org/) for the ORM.
- [Docker](https://www.docker.com/) for containerization.
