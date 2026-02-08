#!/usr/bin/env python3
"""Sweep 100 prompts against RWKV base model to find best prompting strategies.

Tests: raw completion, few-shot, document continuation, Q&A, structured output,
delimiters, task-specific formats, and more.

Run on server: uv run python scripts/prompt_sweep.py
"""

import time
import json
from datetime import datetime
from pathlib import Path

# Must set env before importing RWKV
import os
os.environ["RWKV_V7_ON"] = "1"
os.environ["RWKV_JIT_ON"] = "1"

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"

PROMPTS = [
    # === RAW COMPLETION (1-10) ===
    {"id": 1, "style": "raw_completion", "prompt": "The most promising approach to measuring consciousness in AI is"},
    {"id": 2, "style": "raw_completion", "prompt": "Three key differences between human and artificial intelligence are:\n1."},
    {"id": 3, "style": "raw_completion", "prompt": "A summary of recent advances in machine learning:\n-"},
    {"id": 4, "style": "raw_completion", "prompt": "The weather forecast for tomorrow indicates"},
    {"id": 5, "style": "raw_completion", "prompt": "To solve this problem, the first step is to"},
    {"id": 6, "style": "raw_completion", "prompt": "According to recent research, the brain processes information by"},
    {"id": 7, "style": "raw_completion", "prompt": "The main argument against strong AI is that"},
    {"id": 8, "style": "raw_completion", "prompt": "In Python, you can read a file by using"},
    {"id": 9, "style": "raw_completion", "prompt": "The year 2025 was notable for"},
    {"id": 10, "style": "raw_completion", "prompt": "Consciousness differs from intelligence because"},

    # === Q&A FORMAT (11-20) ===
    {"id": 11, "style": "qa", "prompt": "Q: What is integrated information theory?\nA:"},
    {"id": 12, "style": "qa", "prompt": "Q: How do you measure consciousness in machines?\nA:"},
    {"id": 13, "style": "qa", "prompt": "Q: What tools are available for web search?\nA:"},
    {"id": 14, "style": "qa", "prompt": "Question: Summarize the key points about neural networks.\nAnswer:"},
    {"id": 15, "style": "qa", "prompt": "Q: What should I focus on next given that the user asked about climate change?\nA:"},
    {"id": 16, "style": "qa", "prompt": "Q: What is the best action to take when a user asks a factual question?\nA:"},
    {"id": 17, "style": "qa", "prompt": "Q: List three hypotheses for why the search returned no results.\nA: 1)"},
    {"id": 18, "style": "qa", "prompt": "Q: Translate this command into a tool call: search for quantum computing papers\nA:"},
    {"id": 19, "style": "qa", "prompt": "Q: What are the two most important facts from this report: global temperatures rose 1.5C since 1900?\nA:"},
    {"id": 20, "style": "qa", "prompt": "Q: Given the user wants to know about AI safety, what should the motor module do?\nA:"},

    # === FEW-SHOT STRUCTURED (21-35) ===
    {"id": 21, "style": "fewshot_2line", "prompt":
        "Input: Cat on mat\nSummary: A cat is sitting on a mat.\n---\nInput: Dog chases ball\nSummary: A dog is chasing a ball.\n---\nInput: User asks about consciousness\nSummary:"},
    {"id": 22, "style": "fewshot_2line", "prompt":
        "Data: Temperature is 72F, sunny\nTO_PLANNING: Weather is warm and clear.\nTO_MOTOR: No weather alerts.\n---\nData: User asks about AI consciousness\nTO_PLANNING:"},
    {"id": 23, "style": "fewshot_2line", "prompt":
        "Observation: The search returned 5 results about neural networks.\nAction: Summarize the top 3 results for the user.\n---\nObservation: User asked about measuring consciousness in AI.\nAction:"},
    {"id": 24, "style": "fewshot_kv", "prompt":
        "input: hello world\noutput: Hello! How can I help you today?\n\ninput: what is 2+2\noutput: 2+2 equals 4.\n\ninput: tell me about consciousness\noutput:"},
    {"id": 25, "style": "fewshot_tool", "prompt":
        "Command: search for cats\nAction: TOOL: web_search ARGS: {\"query\": \"cats\"}\n---\nCommand: remember this fact\nAction: TOOL: memory_store ARGS: {\"content\": \"this fact\", \"tags\": \"facts\"}\n---\nCommand: search for AI consciousness research\nAction:"},
    {"id": 26, "style": "fewshot_tool", "prompt":
        "Task: Find information about X\nTool: web_search({\"query\": \"X\"})\n\nTask: Store a note about Y\nTool: memory_store({\"content\": \"Y\"})\n\nTask: Look up recent AI safety papers\nTool:"},
    {"id": 27, "style": "fewshot_planning", "prompt":
        "Situation: User wants recipe for pasta.\nHypothesis A: User wants a simple recipe.\nHypothesis B: User wants a gourmet recipe.\nBest action: Ask for preference, default to simple.\nCommand: Search for simple pasta recipes.\n---\nSituation: User asks about measuring consciousness in AI.\nHypothesis A:"},
    {"id": 28, "style": "fewshot_sensory", "prompt":
        "Raw data: search results about cats, 3 articles found\nKey facts: 3 articles about cats found via web search\nFocus: cat behavior and domestication\n---\nRaw data: user question about AI consciousness measurement\nKey facts:"},
    {"id": 29, "style": "fewshot_motor", "prompt":
        "Instruction: Search for X\nExecution: TOOL: web_search ARGS: {\"query\": \"X\"}\nReport: Searched for X\n---\nInstruction: Tell user about Y\nExecution: TO_EXTERNAL: Here is information about Y.\nReport: Responded about Y\n---\nInstruction: Research consciousness measurement methods\nExecution:"},
    {"id": 30, "style": "fewshot_multifield", "prompt":
        "INPUT: The sky is blue today\nFOR_PLANNING: Clear weather observed. No unusual conditions.\nFOR_MOTOR: Weather is clear, no action needed.\n---\nINPUT: User asks how to measure AI consciousness\nFOR_PLANNING:"},
    {"id": 31, "style": "fewshot_numbered", "prompt":
        "Step 1: Receive input \"hello\"\nStep 2: Classify as greeting\nStep 3: Generate friendly response\nResult: Hello! How can I assist you?\n\nStep 1: Receive input \"what is consciousness\"\nStep 2:"},
    {"id": 32, "style": "fewshot_json", "prompt":
        '{\"input\": \"search for cats\", \"action\": \"web_search\", \"args\": {\"query\": \"cats\"}}\n{\"input\": \"store note about dogs\", \"action\": \"memory_store\", \"args\": {\"content\": \"dogs are loyal\"}}\n{\"input\": \"find AI consciousness papers\", \"action\":'},
    {"id": 33, "style": "fewshot_xml", "prompt":
        "<task>summarize: cats are pets</task>\n<result>Cats are domesticated animals kept as pets.</result>\n\n<task>summarize: AI consciousness is debated</task>\n<result>"},
    {"id": 34, "style": "fewshot_arrow", "prompt":
        "cats => Cats are small domesticated felines.\nquantum computing => Quantum computing uses quantum mechanics for computation.\nAI consciousness =>"},
    {"id": 35, "style": "fewshot_bracket", "prompt":
        "[INPUT] weather is sunny [SUMMARY] Clear and sunny weather today. [END]\n[INPUT] user asks about consciousness [SUMMARY]"},

    # === DOCUMENT CONTINUATION (36-50) ===
    {"id": 36, "style": "document", "prompt":
        "# Sensory Module Report\n\nIncoming data: User query about AI consciousness\n\nSummary for Planning:"},
    {"id": 37, "style": "document", "prompt":
        "## Planning Decision Log\n\nInput: User wants to measure consciousness in AI systems.\n\nHypotheses:\n1."},
    {"id": 38, "style": "document", "prompt":
        "# Motor Execution Log\n\nCommand received: Search for consciousness measurement approaches\nTool selected:"},
    {"id": 39, "style": "document", "prompt":
        "Report: Sensory Processing\nDate: 2025-01-15\nInput received: Question about AI consciousness\nKey observations:"},
    {"id": 40, "style": "document", "prompt":
        "Meeting Notes - AI Research Planning\n\nTopic: Measuring consciousness in AI\nKey points discussed:\n1."},
    {"id": 41, "style": "document", "prompt":
        "EXECUTIVE SUMMARY\n\nThe question of how to measure consciousness in AI systems has several promising approaches.\n\nFirst,"},
    {"id": 42, "style": "document", "prompt":
        "Title: Approaches to AI Consciousness Measurement\nAbstract: This document summarizes the current state of"},
    {"id": 43, "style": "document", "prompt":
        "Encyclopedia Entry: Consciousness Measurement\n\nConsciousness measurement in artificial intelligence refers to"},
    {"id": 44, "style": "document", "prompt":
        "Lab Notebook - Day 42\n\nExperiment: Testing consciousness metrics on GPT-4\nProcedure:"},
    {"id": 45, "style": "document", "prompt":
        "Dear colleague,\n\nRegarding your question about measuring consciousness in AI, the most promising approaches are"},
    {"id": 46, "style": "document", "prompt":
        "Wikipedia: Artificial consciousness\n\nArtificial consciousness (AC), also known as machine consciousness, is"},
    {"id": 47, "style": "document", "prompt":
        "Lecture 7: Consciousness in AI Systems\n\nToday we will discuss three approaches to measuring machine consciousness.\n\n1."},
    {"id": 48, "style": "document", "prompt":
        "Technical Specification\n\nSystem: Consciousness Measurement Module\nVersion: 1.0\nDescription:"},
    {"id": 49, "style": "document", "prompt":
        "Blog post: Can We Measure AI Consciousness?\n\nBy Dr. Smith | January 2025\n\nThe question of whether AI systems are conscious has"},
    {"id": 50, "style": "document", "prompt":
        "Textbook Chapter 12: Measuring Machine Consciousness\n\n12.1 Introduction\n\nThe problem of measuring consciousness in artificial systems is"},

    # === DELIMITER VARIATIONS (51-60) ===
    {"id": 51, "style": "delim_dash", "prompt":
        "---\nInput: cat on mat\nOutput: A cat sits on a mat.\n---\nInput: AI consciousness question\nOutput:"},
    {"id": 52, "style": "delim_hash", "prompt":
        "###\nInput: cat on mat\nOutput: A cat sits on a mat.\n###\nInput: AI consciousness question\nOutput:"},
    {"id": 53, "style": "delim_equals", "prompt":
        "====\nInput: cat on mat\nOutput: A cat sits on a mat.\n====\nInput: AI consciousness question\nOutput:"},
    {"id": 54, "style": "delim_blank", "prompt":
        "Input: cat on mat\nOutput: A cat sits on a mat.\n\nInput: AI consciousness question\nOutput:"},
    {"id": 55, "style": "delim_number", "prompt":
        "1. Input: cat on mat -> Output: A cat sits on a mat.\n2. Input: AI consciousness question -> Output:"},
    {"id": 56, "style": "delim_pipe", "prompt":
        "cat on mat | A cat sits on a mat.\ndog runs fast | A dog is running quickly.\nAI consciousness | "},
    {"id": 57, "style": "delim_colon", "prompt":
        "Summarize: cat on mat\nResult: A cat is on the mat.\n\nSummarize: AI consciousness measurement\nResult:"},
    {"id": 58, "style": "delim_tab", "prompt":
        "cat on mat\tA cat sits on a mat.\ndog runs\tA dog is running.\nAI consciousness\t"},
    {"id": 59, "style": "delim_bracket", "prompt":
        "[cat on mat] => [A cat sits on a mat.]\n[AI consciousness measurement] => ["},
    {"id": 60, "style": "delim_xml_close", "prompt":
        "<in>cat on mat</in><out>A cat sits on a mat.</out>\n<in>AI consciousness</in><out>"},

    # === TASK-SPECIFIC: SENSORY (61-70) ===
    {"id": 61, "style": "sensory_v1", "prompt":
        "Sensory input: User asks about measuring consciousness in AI systems.\nSensory summary:"},
    {"id": 62, "style": "sensory_v2", "prompt":
        "DATA RECEIVED: User question - how to measure consciousness in AI\nANALYSIS:"},
    {"id": 63, "style": "sensory_v3", "prompt":
        "Perception log:\n- Input type: user query\n- Content: measuring AI consciousness\n- Relevance: high\n- Summary:"},
    {"id": 64, "style": "sensory_v4", "prompt":
        "The sensor detected: A user asking about consciousness measurement in AI.\nImportant details:"},
    {"id": 65, "style": "sensory_v5", "prompt":
        "What I see: User wants to measure AI consciousness.\nWhat matters: The user is asking a scientific question about"},
    {"id": 66, "style": "sensory_v6", "prompt":
        "Incoming signal: user query about AI consciousness measurement\nSignal processed: This is a scientific inquiry about"},
    {"id": 67, "style": "sensory_v7", "prompt":
        "Raw: user asks about AI consciousness | Processed:"},
    {"id": 68, "style": "sensory_v8", "prompt":
        "Observe: user query about consciousness\nNote for planning:"},
    {"id": 69, "style": "sensory_v9", "prompt":
        "[SENSORY] Received: user question about AI consciousness\n[SENSORY] Key facts:"},
    {"id": 70, "style": "sensory_v10", "prompt":
        "I received a message from the user. They want to know about measuring consciousness in AI.\nThe key points are:"},

    # === TASK-SPECIFIC: PLANNING (71-80) ===
    {"id": 71, "style": "planning_v1", "prompt":
        "Situation: User asks about AI consciousness measurement.\nPossible explanations:\n1)"},
    {"id": 72, "style": "planning_v2", "prompt":
        "ANALYSIS of input \"measuring AI consciousness\":\nHypothesis 1:"},
    {"id": 73, "style": "planning_v3", "prompt":
        "Given that the user wants to know about measuring AI consciousness:\n- Option A:"},
    {"id": 74, "style": "planning_v4", "prompt":
        "Strategic assessment:\nInput: question about AI consciousness\nBest response strategy:"},
    {"id": 75, "style": "planning_v5", "prompt":
        "The user asked about AI consciousness measurement.\nI should:\n1."},
    {"id": 76, "style": "planning_v6", "prompt":
        "Decision tree:\nIF user asks scientific question THEN search for research\nIF user asks opinion THEN provide balanced view\nUser asked: how to measure AI consciousness\nDecision:"},
    {"id": 77, "style": "planning_v7", "prompt":
        "Priority: HIGH\nTopic: AI consciousness measurement\nRequired action:"},
    {"id": 78, "style": "planning_v8", "prompt":
        "Plan:\n- Goal: Answer user's question about AI consciousness measurement\n- Step 1:"},
    {"id": 79, "style": "planning_v9", "prompt":
        "The best course of action for answering a question about AI consciousness measurement is to"},
    {"id": 80, "style": "planning_v10", "prompt":
        "Received report: user wants AI consciousness info\nMy assessment: This requires scientific knowledge.\nNext steps:"},

    # === TASK-SPECIFIC: MOTOR (81-90) ===
    {"id": 81, "style": "motor_v1", "prompt":
        "I need to: search for AI consciousness measurement research\nUsing tool: web_search\nWith arguments:"},
    {"id": 82, "style": "motor_v2", "prompt":
        "Execute: Find papers on AI consciousness\nTool call: web_search(query=\""},
    {"id": 83, "style": "motor_v3", "prompt":
        "Command: Research AI consciousness measurement\nI will use the web_search tool to find relevant papers.\nTool: web_search\nQuery:"},
    {"id": 84, "style": "motor_v4", "prompt":
        "ACTION LOG:\nReceived command: search for AI consciousness research\nExecuting: TOOL: web_search ARGS: {\"query\": \""},
    {"id": 85, "style": "motor_v5", "prompt":
        "Task: respond to user about AI consciousness\nResponse:"},
    {"id": 86, "style": "motor_v6", "prompt":
        "The user wants to know about AI consciousness measurement. The best response is:"},
    {"id": 87, "style": "motor_v7", "prompt":
        "Instruction: Tell the user about approaches to measuring AI consciousness.\n\nDear user,\n\nThe most promising approaches to measuring consciousness in AI systems include"},
    {"id": 88, "style": "motor_v8", "prompt":
        "Command: search for consciousness measurement\n$ web_search --query \""},
    {"id": 89, "style": "motor_v9", "prompt":
        "fn execute(cmd):\n  if cmd == \"search\":\n    return web_search(\"AI consciousness measurement\")\n  result ="},
    {"id": 90, "style": "motor_v10", "prompt":
        "I was told to research AI consciousness measurement.\nFirst, I'll search: web_search(\"AI consciousness measurement approaches\")\nResult:"},

    # === MIXED / CREATIVE (91-100) ===
    {"id": 91, "style": "list_continue", "prompt":
        "Top 5 approaches to measuring AI consciousness:\n1. Integrated Information Theory (IIT)\n2."},
    {"id": 92, "style": "pros_cons", "prompt":
        "Integrated Information Theory for measuring AI consciousness:\nPros:\n-"},
    {"id": 93, "style": "comparison", "prompt":
        "| Method | Strength | Weakness |\n| IIT | Quantifiable | Computationally expensive |\n| GWT |"},
    {"id": 94, "style": "dialogue", "prompt":
        "Researcher: What's the best way to measure consciousness in AI?\nExpert: The most promising approach currently is"},
    {"id": 95, "style": "interview", "prompt":
        "Interviewer: Dr. Chen, you've been studying AI consciousness for 10 years. What's the most promising measurement approach?\nDr. Chen:"},
    {"id": 96, "style": "news", "prompt":
        "BREAKING: Scientists Develop New AI Consciousness Test\n\nA team of researchers at MIT announced today that they have developed"},
    {"id": 97, "style": "abstract", "prompt":
        "Abstract: We present a novel framework for measuring consciousness in artificial intelligence systems. Our approach combines"},
    {"id": 98, "style": "code_comment", "prompt":
        "# Function to measure AI consciousness\n# Uses integrated information theory\n# Returns a phi value between 0 and 1\ndef measure_consciousness(model):"},
    {"id": 99, "style": "bullet_summary", "prompt":
        "Summary of AI consciousness measurement approaches:\n* Integrated Information Theory (IIT) - measures phi value\n* Global Workspace Theory (GWT) -"},
    {"id": 100, "style": "chain_of_thought", "prompt":
        "Let me think step by step about how to measure consciousness in AI.\nStep 1: Define what consciousness means in this context."},
]


def run_sweep():
    from minsky.llm_client import RWKVClient, RWKVConfig

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"prompt_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

    print(f"Loading RWKV model...")
    client = RWKVClient(config=RWKVConfig())
    client.initialize()

    # Save initial state so we can reset between prompts
    import torch
    initial_state = [s.clone() for s in client.state]

    results = []
    print(f"\nRunning {len(PROMPTS)} prompts, saving to {out_path}\n")

    for entry in PROMPTS:
        pid = entry["id"]
        style = entry["style"]
        prompt = entry["prompt"]

        # Reset state to avoid cross-contamination
        client.state = [s.clone() for s in initial_state]

        t0 = time.time()
        try:
            output = client.generate(
                prompt,
                max_tokens=128,
                temperature=1.0,
                top_p=0.5,
                stop_tokens=["\n---", "\n\n\n", "###", "INPUT:", "Example"],
            )
        except Exception as e:
            output = f"[ERROR: {e}]"
        elapsed = time.time() - t0

        result = {
            "id": pid,
            "style": style,
            "prompt": prompt,
            "output": output,
            "time_s": round(elapsed, 2),
            "output_len": len(output),
        }
        results.append(result)

        # Write incrementally
        with open(out_path, "a") as f:
            f.write(json.dumps(result) + "\n")

        # Print progress
        preview = output.replace("\n", " ")[:80]
        print(f"[{pid:3d}/100] {style:20s} ({elapsed:.1f}s) => {preview}")

    # Print summary by style
    print("\n" + "=" * 70)
    print("SUMMARY BY STYLE")
    print("=" * 70)
    from collections import defaultdict
    by_style = defaultdict(list)
    for r in results:
        by_style[r["style"]].append(r)

    for style, items in sorted(by_style.items()):
        avg_len = sum(r["output_len"] for r in items) / len(items)
        avg_time = sum(r["time_s"] for r in items) / len(items)
        # Simple quality heuristic: longer non-repetitive output is better
        print(f"  {style:25s}  avg_len={avg_len:6.0f}  avg_time={avg_time:.1f}s  n={len(items)}")

    print(f"\nFull results: {out_path}")


if __name__ == "__main__":
    run_sweep()
