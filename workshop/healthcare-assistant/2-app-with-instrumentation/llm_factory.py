"""Amazon Bedrock model factory (replaces direct ChatOpenAI / OpenAIEmbeddings use).

Credentials come from the standard AWS chain (AWS_PROFILE / SSO / env vars / IAM role),
so no API key is needed.

Environment variables:
    AWS_REGION                  Bedrock region (default: ap-northeast-2)
    BEDROCK_CHAT_MODEL_ID       Chat model or inference profile ID, used when the
                                configured model is "bedrock-default"
    BEDROCK_EMBED_MODEL_ID      Embedding model ID (default: amazon.titan-embed-text-v2:0)
"""
import os
from typing import Optional

from langchain_aws import BedrockEmbeddings, ChatBedrockConverse

DEFAULT_CHAT_PLACEHOLDER = "bedrock-default"
DEFAULT_EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"
DEFAULT_REGION = "ap-northeast-2"


def get_region() -> str:
    return os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or DEFAULT_REGION


def resolve_chat_model_id(model: Optional[str] = None) -> str:
    if model and model != DEFAULT_CHAT_PLACEHOLDER:
        return model
    model_id = os.getenv("BEDROCK_CHAT_MODEL_ID")
    if not model_id:
        raise ValueError(
            "BEDROCK_CHAT_MODEL_ID is not set. Set it to a Bedrock model ID or "
            "inference profile ID, e.g. export BEDROCK_CHAT_MODEL_ID=<id from "
            "'aws bedrock list-inference-profiles'>"
        )
    return model_id


def get_chat_model(
    model: Optional[str] = None,
    temperature: float = 0.1,
    name: Optional[str] = None,
) -> ChatBedrockConverse:
    kwargs = {}
    if name:
        kwargs["name"] = name
    return ChatBedrockConverse(
        model=resolve_chat_model_id(model),
        region_name=get_region(),
        temperature=temperature,
        **kwargs,
    )


def get_embeddings(model: Optional[str] = None) -> BedrockEmbeddings:
    model_id = os.getenv("BEDROCK_EMBED_MODEL_ID") or model or DEFAULT_EMBED_MODEL_ID
    return BedrockEmbeddings(model_id=model_id, region_name=get_region())
