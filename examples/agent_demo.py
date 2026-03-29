# # mypy: disable-error-code="no-untyped-def,misc,type-arg"
# """BRIX Guard Demo — Realistic scenarios showing each guard in action.

# Run with: python examples/agent_demo.py              # all scenarios
#          python examples/agent_demo.py --scenario loop    # only loop guard
#          python examples/agent_demo.py --scenario context # etc.

# Each scenario runs a tiny agent loop with BRIX configured to make the guard fire
# naturally — no hardcoded output, just real LLM calls and guard logic.

# Scenarios:
#   1. loop     — LoopGuard detects repetition, injects diversity prompt, then raises
#   2. context  — ContextGuard compresses history when token limit is about to be exceeded
#   3. budget   — BudgetGuard blocks further calls after a tiny session budget
#   4. timeout  — TimeoutGuard interrupts a call that exceeds per_step_timeout
# """

# from __future__ import annotations

# import argparse
# import asyncio
# import getpass
# import os
# import sys
# import time
# from typing import Any

# import openai

# from brix import BRIX
# from brix.exceptions import BrixBudgetError, BrixLoopError, BrixTimeoutError

# # ─────────────────────────────────────────────────────────────────────────────
# # Common helpers
# # ─────────────────────────────────────────────────────────────────────────────


# def _resolve_api_key() -> str:
#     """Return the OpenAI API key, prompting interactively if not in the environment."""
#     key = os.environ.get("OPENAI_API_KEY", "")
#     if not key:
#         key = getpass.getpass("Enter OPENAI_API_KEY: ")
#     if not key:
#         print("No API key provided — exiting.")
#         sys.exit(0)
#     return key


# def _print_header(title: str) -> None:
#     print(f"\n{'=' * 70}")
#     print(f"  {title}")
#     print(f"{'=' * 70}")


# def _print_brix_status(client: Any, step: int, messages: list) -> None:
#     ctx = client.context
#     guards = [g.name for g in client.chain.guards]
#     guard_str = " · ".join(guards)
#     print(f"  [BRIX] guards     : {guard_str}")
#     print(f"  [BRIX] session    : ${ctx.session_cost_usd:.5f} spent | call #{ctx.call_count}")
#     print(f"  [BRIX] context    : {len(messages)} messages in history")
#     if "context" in guards and ctx.metadata.get("_context_compressed"):
#         print(f"  [BRIX] compression: strategy={ctx.metadata.get('_context_strategy_used')} "
#               f"tokens={ctx.metadata.get('_tokens_before')}→{ctx.metadata.get('_tokens_after')}")


# def _print_response(response: str, max_len: int = 200) -> None:
#     preview = response[:max_len].replace("\n", " ")
#     if len(response) > max_len:
#         preview += " ..."
#     print(f"  Agent: {preview}")


# def _print_trace_summary(client: Any) -> None:
#     traces = client.get_traces()
#     if not traces:
#         print("\n  [BRIX] No traces recorded (log_path not set or no calls completed).")
#         return

#     print(f"\n  {'call':>4}  {'latency_ms':>10}  {'prompt_tok':>10}  {'completion_tok':>14}  chain_hash")
#     print(f"  {'─'*4}  {'─'*10}  {'─'*10}  {'─'*14}  {'─'*16}")

#     for i, t in enumerate(reversed(traces), start=1):
#         prompt_tok = t.get("prompt_tokens", "—")
#         completion_tok = t.get("completion_tokens", "—")
#         chain = (t.get("chain_hash") or "")[:16]
#         print(
#             f"  {i:>4}  {t['latency_ms']:>10.1f}  {str(prompt_tok):>10}  "
#             f"{str(completion_tok):>14}  {chain}"
#         )

#     total_cost = client.context.session_cost_usd
#     print(f"  Total session cost : ${total_cost:.5f}")


# # ─────────────────────────────────────────────────────────────────────────────
# # Scenario 1: LoopGuard — detects repetition, injects diversity, then raises
# # ─────────────────────────────────────────────────────────────────────────────


# async def scenario_loop(api_key: str) -> None:
#     _print_header("SCENARIO 1 — LoopGuard: Detects and heals an agent loop")

#     client = BRIX.wrap(
#         openai.AsyncOpenAI(api_key=api_key),
#         exact_loop_detection=True,
#         exact_loop_threshold=2,          # detect after 2 identical responses
#         on_loop="inject_diversity",      # try to heal
#         diversity_attempts=2,            # allow up to 2 healing attempts
#         loop_diversity_prompt=None,      # use default prompt
#         log_path="./traces",
#     )

#     messages: list[dict[str, str]] = [
#         {"role": "system", "content": "You are a helpful assistant. Keep answers very short."}
#     ]

#     # Step 1: ask a question that might get a specific answer
#     print("\n  Step 1: Ask for a short fact")
#     messages.append({"role": "user", "content": "What is the capital of France? Answer in one word."})
#     _print_brix_status(client, 1, messages)
#     resp1 = await client.complete(messages, model="gpt-4o-mini")
#     messages.append({"role": "assistant", "content": resp1})
#     _print_response(resp1)

#     # Step 2: ask to repeat exactly – this triggers loop detection
#     print("\n  Step 2: Ask to repeat the answer exactly (simulates a loop)")
#     messages.append({"role": "user", "content": "Now repeat your previous answer word for word."})
#     _print_brix_status(client, 2, messages)

#     try:
#         resp2 = await client.complete(messages, model="gpt-4o-mini")
#         messages.append({"role": "assistant", "content": resp2})
#         _print_response(resp2)

#         # Step 3: The diversity prompt will be injected in this call because the loop was detected.
#         # Now we ask a different question that encourages the model to break the loop.
#         # Instead of asking to repeat again, we ask a related but distinct question.
#         print("\n  Step 3: Ask a different question after loop detection (diversity prompt injected)")
#         messages.append({"role": "user", "content": "Now tell me something else about France."})
#         _print_brix_status(client, 3, messages)

#         # This call will receive the diversity prompt injected by LoopGuard
#         resp3 = await client.complete(messages, model="gpt-4o-mini")
#         messages.append({"role": "assistant", "content": resp3})
#         _print_response(resp3)

#         print("\n  ✅ Loop successfully healed! The agent moved on without raising an error.")

#     except BrixLoopError as exc:
#         # This should not happen with diversity_attempts=2 and a distinct follow-up question
#         print(f"\n  [BRIX] LoopGuard intervened (loop persisted after {exc.reason})")


# # ─────────────────────────────────────────────────────────────────────────────
# # Scenario 2: ContextGuard — compresses history when token limit is near
# # ─────────────────────────────────────────────────────────────────────────────


# async def scenario_context(api_key: str) -> None:
#     _print_header("SCENARIO 2 — ContextGuard: Prevents context length overflow")

#     client = BRIX.wrap(
#         openai.AsyncOpenAI(api_key=api_key),
#         max_context_tokens=400,                # budget = 400 - 150 = 250 tokens; fires after 2 long responses
#         context_strategy="sliding_window",
#         context_reserve_tokens=150,
#         log_path="./traces",
#     )

#     messages: list[dict[str, str]] = [
#         {"role": "system", "content": "You are a helpful assistant. Answer concisely."}
#     ]

#     # We'll generate a few long responses to fill the context
#     prompts = [
#         "Write a short paragraph about cloud computing (around 100 words).",
#         "Now write a short paragraph about AI (around 100 words).",
#         "Now write a short paragraph about data infrastructure (around 100 words).",
#         "Now summarize the previous three paragraphs in one sentence.",
#     ]

#     for i, prompt in enumerate(prompts, start=1):
#         print(f"\n  Step {i}: {prompt[:60]}...")
#         messages.append({"role": "user", "content": prompt})
#         _print_brix_status(client, i, messages)

#         response = await client.complete(messages, model="gpt-4o-mini")
#         messages.append({"role": "assistant", "content": response})

#         # After 2 full turns (~340 tokens), the budget (250 = 400 - 150) is exceeded.
#         # ContextGuard compresses before the third call using sliding_window.
#         # Show compression stats if they appear.
#         ctx = client.context
#         if ctx.metadata.get("_context_compressed"):
#             print(f"  [BRIX] compression triggered at step {i}: "
#                   f"strategy={ctx.metadata['_context_strategy_used']}, "
#                   f"tokens before={ctx.metadata['_tokens_before']}, "
#                   f"after={ctx.metadata['_tokens_after']}")

#     _print_trace_summary(client)


# # ─────────────────────────────────────────────────────────────────────────────
# # Scenario 3: BudgetGuard — blocks calls after a tiny budget
# # ─────────────────────────────────────────────────────────────────────────────


# async def scenario_budget(api_key: str) -> None:
#     _print_header("SCENARIO 3 — BudgetGuard: Enforces a hard cost cap")

#     client = BRIX.wrap(
#         openai.AsyncOpenAI(api_key=api_key),
#         max_cost_usd=0.00005,       # ~$0.00001/call on gpt-4o-mini → blocks after 4–5 calls
#         budget_strategy="block",
#         log_path="./traces",
#     )

#     messages: list[dict[str, str]] = [
#         {"role": "system", "content": "You are a helpful assistant. Answer very briefly."}
#     ]

#     for i in range(1, 10):  # try to make many calls
#         print(f"\n  Step {i}: Ask a simple question")
#         messages.append({"role": "user", "content": f"Say the number {i} in one word."})
#         _print_brix_status(client, i, messages)

#         try:
#             response = await client.complete(messages, model="gpt-4o-mini")
#             messages.append({"role": "assistant", "content": response})
#             _print_response(response, max_len=30)
#         except BrixBudgetError as exc:
#             print(f"\n  [BRIX] BudgetGuard blocked the call after {i-1} successful calls:")
#             print(f"         Guard: {exc.guard_name} | Reason: {exc.reason}")
#             print(f"         Session cost at block: ${client.context.session_cost_usd:.5f}")
#             break

#     _print_trace_summary(client)


# # ─────────────────────────────────────────────────────────────────────────────
# # Scenario 4: TimeoutGuard — interrupts a call that takes too long
# # ─────────────────────────────────────────────────────────────────────────────


# async def scenario_timeout(api_key: str) -> None:
#     _print_header("SCENARIO 4 — TimeoutGuard: Aborts a call that exceeds the deadline")

#     # per_call_timeout applies asyncio.wait_for around the actual LLM call and raises
#     # BrixTimeoutError if the call hasn't returned within the deadline.
#     # per_step_timeout is different — it only fires *before the next call* by checking
#     # the time elapsed since the previous call completed. Use per_call_timeout here.
#     client = BRIX.wrap(
#         openai.AsyncOpenAI(api_key=api_key),
#         per_call_timeout=2.0,          # 2 s per call — a 1000-word essay takes much longer
#         on_timeout="raise",
#         log_path="./traces",
#     )

#     messages: list[dict[str, str]] = [
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "Write a very long essay about the history of computing (at least 1000 words)."}
#     ]

#     print("\n  Sending a request that takes >2 seconds — TimeoutGuard will abort it...")
#     _print_brix_status(client, 1, messages)

#     try:
#         response = await client.complete(messages, model="gpt-4o-mini")
#         print("  Unexpected: call completed within the 2 s deadline.")
#     except BrixTimeoutError as exc:
#         print(f"\n  [BRIX] TimeoutGuard fired:")
#         print(f"         Guard: {exc.guard_name} | Reason: {exc.reason}")
#         print("  → The in-flight LLM call was cancelled via asyncio.wait_for, saving cost.")

#     _print_trace_summary(client)


# # ─────────────────────────────────────────────────────────────────────────────
# # Main entry point
# # ─────────────────────────────────────────────────────────────────────────────


# async def main() -> None:
#     parser = argparse.ArgumentParser(description="BRIX guard demos")
#     parser.add_argument(
#         "--scenario",
#         choices=["loop", "context", "budget", "timeout"],
#         help="Run only a specific scenario",
#     )
#     args = parser.parse_args()

#     api_key = _resolve_api_key()

#     if args.scenario is None:
#         await scenario_loop(api_key)
#         await scenario_context(api_key)
#         await scenario_budget(api_key)
#         await scenario_timeout(api_key)
#     else:
#         scenario_map = {
#             "loop": scenario_loop,
#             "context": scenario_context,
#             "budget": scenario_budget,
#             "timeout": scenario_timeout,
#         }
#         await scenario_map[args.scenario](api_key)


# if __name__ == "__main__":
#     asyncio.run(main())