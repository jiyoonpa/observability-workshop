"""Model factory: OpenAI Sol models hosted on Amazon Bedrock + Bedrock embeddings.

Chat:
    The app keeps using LangChain's ChatOpenAI, but its OpenAI SDK clients are pointed at
    Bedrock's OpenAI-compatible endpoint (https://bedrock-runtime.<region>.amazonaws.com/openai/v1)
    and signed with AWS credentials (SigV4). No OPENAI_API_KEY is needed.

Embeddings:
    OpenAI does not offer an embedding model on Bedrock, so embeddings use Amazon Titan
    Text Embeddings V2 (1024 dimensions) through langchain-aws.

Environment variables:
    AWS_REGION               Bedrock region (default: ap-northeast-2)
    AWS_PROFILE              Optional AWS profile (default credential chain otherwise)
    BEDROCK_CHAT_MODEL_ID    Used when the configured model is "bedrock-default"
                             (default: global.openai.gpt-6.1-sol)
    BEDROCK_EMBED_MODEL_ID   Embedding model ID (default: amazon.titan-embed-text-v2:0)
"""
import os
from typing import Optional

from langchain_aws import BedrockEmbeddings
from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI, OpenAI
from openai.providers.bedrock import bedrock

DEFAULT_CHAT_PLACEHOLDER = "bedrock-default"
DEFAULT_CHAT_MODEL_ID = "global.openai.gpt-6.1-sol"
DEFAULT_EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"
DEFAULT_REGION = "ap-northeast-2"


def get_region() -> str:
    return os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or DEFAULT_REGION


def resolve_chat_model_id(model: Optional[str] = None) -> str:
    """Map the configured model name to a Bedrock model / inference profile ID."""
    if model and model != DEFAULT_CHAT_PLACEHOLDER:
        return model
    return os.getenv("BEDROCK_CHAT_MODEL_ID") or DEFAULT_CHAT_MODEL_ID


def _bedrock_provider():
    return bedrock(
        endpoint="runtime",
        region=get_region(),
        profile=os.getenv("AWS_PROFILE") or None,
    )


def get_chat_model(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    name: Optional[str] = None,
) -> ChatOpenAI:
    """Return a ChatOpenAI that talks to a Sol model on Bedrock.

    `temperature` is accepted for compatibility with existing callers but is NOT sent:
    Sol models only support the default temperature (1) and reject any other value with
    "Unsupported value: 'temperature' does not support 0.1 with this model".
    Set BEDROCK_CHAT_TEMPERATURE only if you switch to a model that accepts it.
    """
    sync_client = OpenAI(provider=_bedrock_provider())
    async_client = AsyncOpenAI(provider=_bedrock_provider())
    kwargs = {"name": name} if name else {}
    env_temperature = os.getenv("BEDROCK_CHAT_TEMPERATURE")
    if env_temperature:
        kwargs["temperature"] = float(env_temperature)
    # Sol rejects function tools on /v1/chat/completions ("To use function tools, use
    # /v1/responses"), so the Responses API is required for the agent's tool calls.
    return ChatOpenAI(
        model=resolve_chat_model_id(model),
        use_responses_api=True,
        root_client=sync_client,
        root_async_client=async_client,
        client=sync_client.chat.completions,
        async_client=async_client.chat.completions,
        **kwargs,
    )


def get_embeddings(model: Optional[str] = None) -> BedrockEmbeddings:
    model_id = os.getenv("BEDROCK_EMBED_MODEL_ID") or model or DEFAULT_EMBED_MODEL_ID
    return BedrockEmbeddings(model_id=model_id, region_name=get_region())
