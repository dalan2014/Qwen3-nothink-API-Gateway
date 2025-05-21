# Qwen3 nothink API Gateway

This FastAPI application serves as an intelligent asynchronous API gateway designed to seamlessly integrate Qwen3 large language models (deployed with vLLM) into existing OpenAI-compatible workflows.

## What Problem This Gateway Solves

Interacting with Qwen3 models' "thinking" and "non-thinking" modes via vLLM can require specific API parameters that are not part of the standard OpenAI client requests. This gateway addresses these challenges by:

1.  **Transparent Non-Thinking Mode:** It automatically modifies incoming standard OpenAI chat completion requests to instruct the Qwen3 model to operate in its efficient "non-thinking" mode (`enable_thinking: False`). This is done by injecting the necessary `chat_template_kwargs`.
2.  **Optimal Default Parameters:** It applies recommended default sampling parameters (like `temperature`, `top_p`, `presence_penalty`) for Qwen's non-thinking mode if these are not already specified in the client's request, ensuring better performance and response quality.
3.  **Correct Content Placement:** When used with a correctly configured vLLM backend (see Prerequisites), this gateway helps ensure that the model's responses in non-thinking mode are correctly placed in the standard `content` field of the OpenAI API response. This avoids issues where answers might be misplaced (e.g., into a `reasoning_content` field), similar to problems reported in contexts like [vLLM GitHub Issue #17349](https://github.com/vllm-project/vllm/issues/17349) (Note: The gateway itself doesn't fix backend bugs but enables a usage pattern and backend configuration that circumvents such specific issues when `enable_thinking: False` is used).
4.  **Standard OpenAI Endpoint Proxying:** Besides the specialized handling for `/v1/chat/completions`, the gateway also acts as a generic proxy for other standard OpenAI API endpoints (e.g., `/v1/models`), forwarding these requests unmodified to your vLLM backend.

Essentially, this gateway allows client applications to interact with a Qwen3 model via vLLM as if it were a standard OpenAI endpoint, benefiting from Qwen3's non-thinking mode without requiring client-side changes.

## Key Features

-   Listens for OpenAI-compatible requests on port 8000 (default).
-   **Specifically for `/v1/chat/completions` requests:**
    -   Injects `{"chat_template_kwargs": {"enable_thinking": False}}` to activate Qwen's non-thinking mode.
    -   Applies Qwen-recommended default sampling parameters for non-thinking mode if not provided by the client.
-   **For all other paths (e.g., `/v1/models`):**
    -   Acts as a transparent asynchronous proxy to the backend vLLM server.
-   Forwards requests to a backend vLLM server (configurable, default `http://localhost:10001`).
-   Supports both streaming and non-streaming responses.
-   Asynchronous architecture using FastAPI and HTTPX for high performance.
-   Configurable via environment variables.

## Prerequisites

-   Python 3.8+
-   A running vLLM instance serving a Qwen3 model (e.g., `Qwen/Qwen3-235B-A22B-GPTQ-Int4`) on the specified backend port (default `10001`).

    **Crucial vLLM Configuration for Non-Thinking Mode:**
    To ensure compatibility with the gateway's primary function of enabling `enable_thinking: False` (for non-thinking responses correctly placed in the `content` field), it is **essential to run your vLLM server *without* certain reasoning/parsing flags that can conflict.**

    Based on Qwen documentation, if you pass `enable_thinking=False` (which this gateway does for chat completions), you should *disable conflicting server-side parsing of thinking content* in vLLM.

    **Recommended vLLM Startup Command Example:**
    ```bash
    vllm serve YourHuggingFaceModelID/YourQwen3ModelName \
      --tensor-parallel-size YOUR_TP_SIZE \
      --port 10001 \
      --served-model-name YourServedModelNameForAPI \
      # Add other necessary flags like --quantization, --trust-remote-code, --max-model-len, etc.
      # Critically, OMIT flags like --enable-reasoning --reasoning-parser some_parser
      # if they cause conflicts with enable_thinking: False as per Qwen/vLLM documentation.
    ```
    For instance:
    ```bash
    vllm serve Qwen/Qwen3-235B-A22B-GPTQ-Int4 \
      --tensor-parallel-size 4 \
      --port 10001 \
      --served-model-name Qwen3-235B \
      --quantization gptq_marlin \
      --trust-remote-code \
      --max-model-len 32000 \
      --gpu-memory-utilization 0.92
    ```
    Always refer to the latest official Qwen and vLLM documentation for deploying Qwen3 models, especially concerning the interaction between `enable_thinking` API flags and vLLM's reasoning/parsing features.

## Setup

1.  **Clone the repository (or create the files as provided in the project structure).**

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure environment variables:**
    Copy `.env_example` to `.env` and customize if needed:
    ```bash
    cp .env_example .env
    ```
    Key variables in `.env`:
    -   `VLLM_BASE_URL`: Base URL of your vLLM server (e.g., `http://localhost:10001`).
    -   `VLLM_REQUEST_TIMEOUT`: Timeout for requests to the vLLM backend.
    -   `DEFAULT_TEMPERATURE`, `DEFAULT_TOP_P`, etc.: Default Qwen non-thinking parameters applied if not sent by client.

## Running the Gateway

Use Uvicorn to run the FastAPI application:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload