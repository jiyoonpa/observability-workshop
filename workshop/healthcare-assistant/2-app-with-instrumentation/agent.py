"""LangGraph agent for the call-summary assistant."""
import asyncio
import inspect
import json
import threading
import uuid
from typing import Annotated, Any, List, Dict, Optional, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langchain_core.runnables import RunnableConfig                  # 추가 - Control 1

from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from config import TOOLS_DIR, load_config, load_system_prompt
from llm_factory import get_chat_model
from rag import create_rag_tool
from tools import logic as tools_logic
from helpers.agent_control_helpers import (                          # 추가 - Contorl
    build_agent_control_steps,
    ensure_trace_started,
    finalize_trace,
    format_blocked_message,
    init_agent_control,
    infer_control_step_name,
    make_controlled_tool,
    notify_control_block,
    uses_internal_sql_control,
)


import os
# 추가 1
from splunk_ao import splunk_ao_context

#추가 2
from splunk_ao.handlers.langchain import SplunkAOAsyncCallback

from splunk_ao.utils.log_config import enable_console_logging
import logging

enable_console_logging(level=logging.INFO)

# 추가 Control - 
from agent_control import ControlSteerError, ControlViolationError, control

# Control의 step name을 이 값으로 지정해야 LLM 출력에 적용됩니다.
LLM_STEP_NAME = "Call Summary Assistant"
MAX_STEER_RETRIES = 3

class State(TypedDict):
    messages: Annotated[list, add_messages]


_bg_loop: Optional[asyncio.AbstractEventLoop] = None
_bg_loop_lock = threading.Lock()


def _get_background_loop() -> asyncio.AbstractEventLoop:
    """Return a process-wide event loop running on a dedicated daemon thread.

    Streamlit calls process_query() synchronously on every interaction. Using
    asyncio.run() per call creates and then *closes* a new loop each time, but
    modern langchain-openai/openai reuse a cached async HTTP client across calls,
    so its connection pool outlives the loop that created it. When a later request
    reaps a keep-alive connection bound to that closed loop, httpx raises
    "RuntimeError: Event loop is closed". Running every coroutine on one stable,
    long-lived loop keeps the shared client and its pool on a single loop.
    """
    global _bg_loop
    if _bg_loop is not None and not _bg_loop.is_closed():
        return _bg_loop
    with _bg_loop_lock:
        if _bg_loop is None or _bg_loop.is_closed():
            loop = asyncio.new_event_loop()
            threading.Thread(
                target=loop.run_forever,
                name="agent-async-loop",
                daemon=True,
            ).start()
            _bg_loop = loop
    return _bg_loop


def _run_async(coro):
    """Run an async coroutine from sync code (e.g. Streamlit) on the shared loop."""
    loop = _get_background_loop()
    return asyncio.run_coroutine_threadsafe(coro, loop).result()

# 추가 Control - Helper 함수 추가 
def _message_content_text(message: BaseMessage) -> str:
    # Responses API는 content가 블록 리스트라서 .text를 우선 사용
    text = getattr(message, "text", None)
    if isinstance(text, str) and text:
        return text
    content = getattr(message, "content", "")
    return content if isinstance(content, str) else str(content)


def _build_steering_retry_messages(messages, original_output: AIMessage, steer_error: ControlSteerError):
    instructions = steer_error.steering_context or steer_error.message or str(steer_error)
    original_text = _message_content_text(original_output)
    retry = list(messages)
    # tool_calls는 이어붙이지 않습니다 (대응하는 tool 응답이 없어 거부됨)
    retry.append(AIMessage(content=original_text))
    retry.append(HumanMessage(content=(
        "Your previous response was flagged by Agent Control and needs to be revised.\n\n"
        f"Your previous response:\n{original_text}\n\n"
        "Follow these instructions to generate a corrected response:\n"
        f"{instructions}"
    )))
    return retry

class CallSummaryAgent:
    """LangGraph call-summary assistant."""

    def __init__(
        self,
        session_id: str | None = None,
        model_override: Optional[str] = None,
    ):
        self.config = load_config()
        self.session_id = session_id or str(uuid.uuid4())
        self.model_override = model_override
        self.system_prompt = load_system_prompt()
        self.tools = []
        self.graph: CompiledStateGraph | None = None
        self.langgraph_config = {"configurable": {"thread_id": self.session_id}}
        self._control_steps: list[dict] | None = None

    def _init_agent_control(self, splunk_ao_logger) -> None:
        if splunk_ao_logger is None or self._control_steps is None:
            return
        splunk_ao_logger.enable_agent_control()
        init_agent_control(
            splunk_ao_logger,
            project_name=os.getenv("SPLUNK_AO_PROJECT", ""),
            agent_stream=os.getenv("SPLUNK_AO_AGENT_STREAM", ""),
            agent_description="Call summary assistant demo agent",
            steps=self._control_steps,
        )

    def load_tools(self) -> None:
        tool_schema_path = TOOLS_DIR / "schema.json"
        with tool_schema_path.open(encoding="utf-8") as f:
            tool_schema = json.load(f)

        tool_names = [s.get("name") for s in tool_schema if s.get("name")]
        self._control_steps = build_agent_control_steps(LLM_STEP_NAME, tool_names)

        self.tools = []
        for tool_func in tools_logic.TOOLS:
            func_name = tool_func.__name__
            if not uses_internal_sql_control(func_name) and not getattr(
                tool_func, "_agent_control_step", None
            ):
                step_name = infer_control_step_name(func_name)
                tool_func = make_controlled_tool(tool_func, step_name)
                print(f"   🛡️ Agent Control step '{step_name}' → {func_name}")

            tool_schema_dict = next(
                (schema for schema in tool_schema if schema.get("name") == tool_func.__name__),
                None,
            )
            tool_kwargs = {
                "name": tool_func.__name__,
                "description": (
                    tool_schema_dict.get("description")
                    if tool_schema_dict
                    else tool_func.__doc__ or f"Tool: {tool_func.__name__}"
                ),
                "args_schema": tool_schema_dict.get("parameters") if tool_schema_dict else None,
            }
            if inspect.iscoroutinefunction(tool_func):
                langchain_tool = StructuredTool.from_function(coroutine=tool_func, **tool_kwargs)
            else:
                langchain_tool = StructuredTool.from_function(func=tool_func, **tool_kwargs)
            self.tools.append(langchain_tool)

        rag_config = self.config.get("rag", {})
        if rag_config.get("enabled", False):
            top_k = rag_config.get("top_k", 5)
            model_config = self.config.get("model", {})
            effective_model = (
                self.model_override
                or model_config.get("default_model")
                or model_config.get("model_name")
            )
            rag_tool = create_rag_tool(top_k, model_name=effective_model)
            self.tools.append(rag_tool)

        print(f"✓ Loaded {len(self.tools)} tools")

    def _build_graph(self) -> CompiledStateGraph:
        if not self.tools:
            raise ValueError("Tools not loaded. Call load_tools() first.")

        model_config = self.config.get("model", {})
        effective_model = (
            self.model_override
            or model_config.get("default_model")
            or model_config.get("model_name")
        )
        temperature = model_config.get("temperature", 0.1)

        llm_with_tools = get_chat_model(
            model=effective_model,
            temperature=temperature,
            name=LLM_STEP_NAME,
        ).bind_tools(self.tools)

        last_llm_output: Dict[str, Any] = {"message": None}

        @control(step_name=LLM_STEP_NAME) #추가 Control - @control을 붙인 invoke llm 호출로 변경 
        async def _invoke_llm(msgs, config: RunnableConfig):
            result = await llm_with_tools.ainvoke(msgs, config)
            last_llm_output["message"] = result
            return result

        async def invoke_chatbot(state, config: RunnableConfig):
            messages = list(state["messages"])
            if self.system_prompt:
                messages = [SystemMessage(content=self.system_prompt)] + messages

            llm_messages = messages
            last_llm_output["message"] = None
            message = None
            # 추가 Control - ControlViolationError가 나면, format_blocked_message가 만든 안내문구로 응답하고
            for attempt in range(MAX_STEER_RETRIES):
                try:
                    message = await _invoke_llm(llm_messages, config)
                    break
                except ControlViolationError as e:
                    notify_control_block(e, step_name=LLM_STEP_NAME)
                    message = AIMessage(content=format_blocked_message(e, step_name=LLM_STEP_NAME))
                    break
                # ControlSteerError가 나면 3번까지 다시 호출하도록 수정
                except ControlSteerError as e:
                    notify_control_block(e, step_name=LLM_STEP_NAME, guardrail_result="steered")
                    if attempt >= MAX_STEER_RETRIES - 1 or last_llm_output["message"] is None:
                        message = AIMessage(
                            content=format_blocked_message(e, step_name=LLM_STEP_NAME, steered=True)
                        )
                        break
                    llm_messages = _build_steering_retry_messages(
                        llm_messages, last_llm_output["message"], e
                    )
                    last_llm_output["message"] = None
                except RuntimeError as e:
                    # Control 평가 서버 오류 시 앱이 죽지 않고 안내 메시지로 대체
                    print(f"  ⚠️  Agent Control server error ({LLM_STEP_NAME}): {e}")
                    message = AIMessage(
                        content=(
                            "I'm sorry, I'm unable to process your request right now due to "
                            f"a control evaluation error: {e}"
                        )
                    )
                    break

            if message is None:
                message = AIMessage(content="No response generated")
            return {"messages": [message]}

        graph_builder = StateGraph(State)
        graph_builder.add_node("chatbot", invoke_chatbot)
        graph_builder.add_node("tools", ToolNode(tools=self.tools))
        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_conditional_edges("chatbot", tools_condition)
        graph_builder.add_edge("tools", "chatbot")
        return graph_builder.compile()

    async def _process_query_async(self, messages: List[Dict[str, str]]) -> str:
        if not self.tools:
            self.load_tools()
        self.graph = self._build_graph()

        langchain_messages: List[BaseMessage] = []
        for msg in messages:
            if msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_messages.append(AIMessage(content=msg["content"]))
    
        # 추가 3- 컨텍스트, 세션 지정
        with splunk_ao_context(
            project=os.getenv("SPLUNK_AO_PROJECT"),
            agent_stream=os.getenv("SPLUNK_AO_AGENT_STREAM"),
        ):
            # Workaround for an intermittent hang (SDK deadlock): start_session() lazily resolves the
            # project/agent stream from *inside* the SDK's own event-loop thread pool, and that nested
            # sync call can pick the same busy thread and wait on itself forever. Resolving the IDs
            # here, on a non-pool thread, makes start_session() skip that nested call.
            splunk_ao_logger = splunk_ao_context.get_logger_instance()
            try:
                splunk_ao_logger._ensure_session_crud_client()
            except Exception as e:  # tracing must never block answering
                print(f"[WARN] Splunk AO warm-up failed: {e}")

            # agent_stream_id는 세션 시작 후에야 확정되며 Agent Control 초기화에 필요합니다.
            splunk_ao_context.start_session(external_id=self.session_id)
            self._init_agent_control(splunk_ao_logger)
            ensure_trace_started(splunk_ao_logger, langchain_messages, trace_name="Run Agent")

            response = "No response generated"
            try:
                # 콜백 연결 - 콜백이 LangGraph 실행 중에 LLM 호출, 도구 호출, RAG를 자동으로 span으로 기록하는데
                # 이 기록이 evaluator와 Signals가 분석하는 원본 데이터가 됨
                # start_new_trace=False: ensure_trace_started()가 만든 trace 아래에 중첩
                callback = SplunkAOAsyncCallback(
                    splunk_ao_logger,
                    start_new_trace=False,
                    flush_on_chain_end=False,
                )
                run_config = {**self.langgraph_config, "callbacks": [callback]}

                result = await self.graph.ainvoke(
                    {"messages": langchain_messages},
                    run_config,
                )
                if result["messages"]:
                    # Responses API returns a list of content blocks; .text joins them into a str.
                    response = str(result["messages"][-1].text)
                return response
            finally:
                finalize_trace(splunk_ao_logger, response)

    def process_query(self, messages: List[Dict[str, str]]) -> str:
        try:
            return _run_async(self._process_query_async(messages))
        except ControlViolationError as e:
            return format_blocked_message(e, step_name=LLM_STEP_NAME)
        except ControlSteerError as e:
            return format_blocked_message(e, step_name=LLM_STEP_NAME, steered=True)
        except Exception as e:
            print(f"[ERROR] Error processing query: {e}")
            import traceback

            traceback.print_exc()
            return f"Error processing your request: {str(e)}"