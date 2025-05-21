# Qwen3 No-Think API Gateway

An intelligent FastAPI gateway designed to let you easily and transparently use the "non-thinking" (nothink) mode of Qwen3 large language models via a standard OpenAI API interface. Supports vLLM or SGLang as backend inference engines.

## Core Problem Solved

Qwen3 models support "thinking" and "non-thinking" modes. Directly invoking these modes via vLLM/SGLang, especially the "non-thinking" mode, might require client-side handling of non-standard API parameters. This gateway addresses these challenges:

1.  **Transparently Enable "Non-Thinking" Mode**: Clients can make requests to `/v1/chat/completions` like any standard OpenAI API without modification. The gateway automatically injects the `{"chat_template_kwargs": {"enable_thinking": False}}` parameter, instructing the Qwen3 model to operate in its efficient "non-thinking" mode.
2.  **Optimized Default Parameters**: For "non-thinking" mode, it automatically applies recommended sampling parameters (e.g., `temperature`, `top_p`) if not already specified by the client, enhancing response quality and performance.
3.  **Standard API Proxying**: Besides special handling for chat completions, other standard OpenAI API endpoints (e.g., `/v1/models`) are transparently proxied to the backend.
4.  **Addressing Content Misplacement Issues**:
    * **Background**: Under certain vLLM configurations (especially when a reasoning parser incompatible with `enable_thinking: False` is active on the backend), Qwen3's responses might incorrectly appear in the `reasoning_content` field instead of the standard `content` field (similar to issues like [vLLM GitHub Issue #17349](https://github.com/vllm-project/vllm/issues/17349)).
    * **How This Gateway Helps**: By ensuring `enable_thinking: False` is sent, and **in conjunction with appropriate backend inference engine configuration** (e.g., running vLLM without a conflicting `--reasoning-parser`, or using a version that correctly handles this flag), this gateway helps ensure that "non-thinking" mode responses have their content correctly placed in the `content` field. The gateway itself doesn't fix backend bugs but enables a usage pattern that circumvents such issues.

**In a Nutshell**: This gateway allows your applications to seamlessly switch to Qwen3's "non-thinking" mode without worrying about underlying parameter details, while also providing a strategy to address specific content misplacement issues.

## Key Features

* Listens for standard OpenAI-compatible requests (default port `8000`).
* **For `/v1/chat/completions`**: Automatically enables Qwen3 "non-thinking" mode and optimizes parameters.
* **For other API paths**: Acts as a transparent asynchronous proxy to the backend.
* Supports vLLM and SGLang backends.
* Supports both streaming and non-streaming responses.
* Asynchronous architecture using FastAPI and HTTPX for high performance.
* Configurable via an `.env` file.

## Prerequisites

* Python 3.8+
* A running backend inference server (vLLM or SGLang) loaded with a Qwen3 model (e.g., [Qwen/Qwen3-235B-A22B-GPTQ-Int4](https://huggingface.co/Qwen/Qwen3-235B-A22B-GPTQ-Int4)), providing an OpenAI-compatible API on the specified port (default `10001`).

### Backend Inference Engine Configuration Advice

* **Direct Backend Access**: You can configure your backend (e.g., port `10001`) to support "thinking" mode (e.g., by using `--enable-reasoning --reasoning-parser <parser>` with vLLM).
* **Access via This Gateway**: This gateway (port `8000`) will always enable "non-thinking" mode for `/v1/chat/completions` requests.

## Quick Start

1.  **Clone or download the project.**
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Configure environment variables:**
    Copy `.env_example` to `.env` and modify `VLLM_BASE_URL` as needed (pointing to your backend service, e.g., `http://localhost:10001`).
    ```bash
    cp .env_example .env
    ```
4.  **Run the gateway:**
    ```bash
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
    ```
