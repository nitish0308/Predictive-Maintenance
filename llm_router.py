"""LLM-driven routing between the prediction models in models.py, via a local Ollama model.

The four functions in models.py each answer a different kind of question (machine
failure, process-temperature forecast, cumulative tool wear, rotational-speed
feasibility). Which one applies to a given natural-language question is NOT decided by
keyword/regex matching in this file -- it's decided by the local LLM reading each tool's
description and the question, and choosing which tool (if any) to call. This module
only executes whatever the model decides and reports back.

Requires Ollama running locally (`ollama serve`) with MODEL pulled (`ollama pull qwen3:4b`).
"""

import json

import ollama

import models

MODEL = "qwen3:4b"  # local model confirmed to support Ollama's tool-calling format

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "predict_machine_failure",
            "description": (
                "Predicts whether a machine will experience an overall failure/defect "
                "(binary + probability) from a full sensor reading, and diagnoses which "
                "specific failure mode (tool wear / heat dissipation / power / overstrain) "
                "the reading sits closest to. Use for questions asking whether a machine "
                "will fail, is defective, or should be flagged for maintenance, given a "
                "specific set of readings."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "product_type": {"type": "string", "enum": ["L", "M", "H"],
                                      "description": "Product quality variant. If not stated in the question, omit it -- 'M' will be assumed and that assumption reported."},
                    "air_temp_K": {"type": "number"},
                    "process_temp_K": {"type": "number"},
                    "rotational_speed_rpm": {"type": "number"},
                    "torque_Nm": {"type": "number"},
                    "tool_wear_min": {"type": "number"},
                },
                "required": ["air_temp_K", "process_temp_K", "rotational_speed_rpm", "torque_Nm", "tool_wear_min"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_process_temperature",
            "description": (
                "Forecasts the resulting process temperature (Kelvin) from air temperature "
                "(and optionally rotational speed / torque, though empirically neither "
                "affects process temperature in this dataset). Use for questions asking what "
                "process temperature will result from given operating conditions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "air_temp_K": {"type": "number"},
                    "rotational_speed_rpm": {"type": "number"},
                    "torque_Nm": {"type": "number"},
                },
                "required": ["air_temp_K"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "estimate_tool_wear",
            "description": (
                "Estimates cumulative tool wear after a given number of production cycles, "
                "using the empirical per-cycle wear rate, and reports the 200-240 minute "
                "tool-wear-failure replacement window the tool would hit well before large "
                "cycle counts. Use for questions about cumulative or expected tool wear over "
                "many cycles/operations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "n_cycles": {"type": "number"},
                    "product_type": {"type": "string", "enum": ["L", "M", "H"]},
                },
                "required": ["n_cycles"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "estimate_rotational_speed_for_target_temp",
            "description": (
                "Attempts to determine the rotational speed needed to reach a target process "
                "temperature, given air temperature (and optionally torque). Empirically, "
                "rotational speed has ~0 correlation with process temperature in this "
                "dataset, so this tool typically reports infeasibility with an explanation "
                "rather than a fabricated rpm value. Use for questions asking what rotational "
                "speed setting is needed to achieve a target process temperature."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target_process_temp_K": {"type": "number"},
                    "air_temp_K": {"type": "number"},
                    "torque_Nm": {"type": "number"},
                },
                "required": ["target_process_temp_K", "air_temp_K"],
            },
        },
    },
]

SYSTEM_PROMPT = (
    "You are the routing layer for a predictive-maintenance system built on the AI4I "
    "2020 milling-machine dataset. You have four specialized tools, each backed by a "
    "model or empirical relationship derived from that dataset. Read the user's "
    "question, decide which ONE tool actually answers it, extract the parameters the "
    "question gives, call that tool, then explain the result in plain, concise language "
    "grounded in the numbers the tool returned -- including any caveats or limitations "
    "the tool reports (for example, when a parameter turns out to have no real effect). "
    "If the question is missing a needed parameter, make the most reasonable assumption "
    "and say so explicitly in your answer."
)


def _execute_tool(name, tool_input):
    if name == "predict_machine_failure":
        return models.predict_machine_failure(
            air_temp=tool_input["air_temp_K"],
            process_temp=tool_input["process_temp_K"],
            rpm=tool_input["rotational_speed_rpm"],
            torque=tool_input["torque_Nm"],
            tool_wear=tool_input["tool_wear_min"],
            product_type=tool_input.get("product_type", "M"),
        )
    if name == "predict_process_temperature":
        return models.predict_process_temperature(
            air_temp=tool_input["air_temp_K"],
            rpm=tool_input.get("rotational_speed_rpm"),
            torque=tool_input.get("torque_Nm"),
        )
    if name == "estimate_tool_wear":
        return models.estimate_tool_wear(
            n_cycles=tool_input["n_cycles"],
            product_type=tool_input.get("product_type"),
        )
    if name == "estimate_rotational_speed_for_target_temp":
        return models.estimate_rotational_speed_for_target_temp(
            target_process_temp=tool_input["target_process_temp_K"],
            air_temp=tool_input["air_temp_K"],
            torque=tool_input.get("torque_Nm"),
        )
    raise ValueError(f"Unknown tool: {name}")


def answer_question(question, max_rounds=4):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    trace = []

    for _ in range(max_rounds):
        response = ollama.chat(model=MODEL, messages=messages, tools=TOOLS)
        messages.append(response.message)

        if not response.message.tool_calls:
            return {"answer": response.message.content, "trace": trace}

        for call in response.message.tool_calls:
            name = call.function.name
            tool_input = dict(call.function.arguments)
            try:
                result = _execute_tool(name, tool_input)
                trace.append({"tool": name, "input": tool_input, "result": result})
                messages.append({"role": "tool", "content": json.dumps(result), "name": name})
            except Exception as exc:
                trace.append({"tool": name, "input": tool_input, "error": str(exc)})
                messages.append({"role": "tool", "content": f"Error: {exc}", "name": name})

    return {"answer": "Could not resolve an answer after several tool calls.", "trace": trace}
