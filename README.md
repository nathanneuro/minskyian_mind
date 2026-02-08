# Minsky: Society of Mind

A cognitive architecture inspired by Marvin Minsky's "Society of Mind" — three
specialized rooms (Sensory, Planning, Motor) communicate via short messages,
coordinated by an orchestrator. A frozen LLM generates text; a small trainable
T5 model learns to edit it based on judge feedback.

```
                            ┌─────────────────────────────┐
                            │       EXTERNAL USER         │
                            │   (real or fake_user bot)    │
                            └──────────┬──────────────────┘
                                       │ unbounded messages
                                       │
              ┌────────────────────────┼────────────────────────┐
              │                  ORCHESTRATOR                    │
              │  routes messages, runs judges & summarizers      │
              │                                                 │
              │  ┌─────────┐   ┌──────────┐   ┌─────────────┐  │
              │  │ Judges   │   │Summarize │   │  Forecast   │  │
              │  │(DeepSeek)│   │(DeepSeek)│   │   (RWKV)    │  │
              │  └─────────┘   └──────────┘   └─────────────┘  │
              └──┬──────────────────┬───────────────────┬───────┘
                 │                  │                   │
        ~256 char messages   ~256 char messages  ~256 char messages
                 │                  │                   │
    ┌────────────▼───┐   ┌─────────▼────┐   ┌─────────▼──────┐
    │    SENSORY      │   │   PLANNING    │   │     MOTOR      │
    │                 │   │               │   │                │
    │ ┌─────────────┐ │   │ ┌───────────┐ │   │ ┌────────────┐ │
    │ │ Agent L / R  │ │   │ │ Agent L/R │ │   │ │ Agent L / R│ │
    │ │ (free-form)  │ │   │ │(free-form)│ │   │ │ (free-form)│ │
    │ │  unbounded   │ │   │ │ unbounded │ │   │ │  unbounded │ │
    │ └──────┬──────┘ │   │ └─────┬─────┘ │   │ └─────┬──────┘ │
    │        ▼         │   │       ▼       │   │       ▼        │
    │ ┌─────────────┐ │   │ ┌───────────┐ │   │ ┌────────────┐ │
    │ │ Agent R / L  │ │   │ │ Agent R/L │ │   │ │ Agent R / L│ │
    │ │ (structured) │ │   │ │(structured│ │   │ │(structured)│ │
    │ └─────────────┘ │   │ └───────────┘ │   │ └────────────┘ │
    │                 │   │               │   │                │
    │ Observes world, │   │ Hypothesizes, │   │ Calls tools,   │
    │ summarizes,     │   │ picks best    │   │ sends messages │
    │ directs         │   │ action,       │   │ to external    │
    │ attention       │   │ commands      │   │ user           │
    └─────────────────┘   └───────────────┘   └────────────────┘
                                                     │
                                              ┌──────▼───────┐
                                              │    TOOLS     │
                                              │ web_search   │
                                              │ memory_*     │
                                              │ scratchpad_* │
                                              └──────────────┘

    ┌──────────────────────────────────────────────────────────┐
    │                   GPU PIPELINE                           │
    │                                                          │
    │   GPU 0: Frozen LLM (Qwen 8B or RWKV 7B)               │
    │          generates raw text for each room agent          │
    │                                                          │
    │   GPU 1: T5Gemma 270M (trainable edit model)            │
    │          edits raw LLM output into structured messages   │
    │          learns from judge counterfactuals               │
    └──────────────────────────────────────────────────────────┘
```

## How It Works

**Each global step:**

1. The orchestrator delivers queued messages to each room
2. Each room runs two LLM calls (dual-agent):
   - First agent (randomly L or R) does free-form analysis (unbounded)
   - Second agent reads the analysis and produces structured output
3. Raw LLM output passes through T5 for editing/cleanup
4. Structured output is parsed into short (~256 char) between-room messages
5. Messages are routed to target rooms for the next step
6. Motor can send unbounded messages to the external user via `TO_EXTERNAL`

**Feedback loop:**

- Judges (DeepSeek API) score each room's output and generate counterfactuals
- Counterfactuals become training targets: T5 learns `raw_output -> improved_output`
- Training pairs accumulate in `data/train_data/` as JSONL files
- A fake user (DeepSeek API) simulates conversation to keep the loop running

## Setup

```bash
uv sync
cp .env.example .env   # add INF_API_KEY for judges/summarizers/fake_user
```

## Usage

```bash
# Run the main experiment
uv run python main.py                        # uses config.toml
uv run python main.py --config other.toml    # custom config

# Train T5 on accumulated judge data
uv run python scripts/train_t5.py            # train on pending data
uv run python scripts/train_t5.py --stats    # show training stats
uv run python scripts/train_t5.py --replay 5 # replay last 5 used files
uv run python scripts/train_t5.py --epochs 3 # multiple passes
uv run python scripts/train_t5.py --rollback # revert to previous checkpoint
```

## Configuration

All parameters live in `config.toml`:

```toml
[run]
max_steps = 200
prompt = "Your initial prompt here"

[run.features]
llm = true          # frozen LLM generation
t5 = true           # T5 edit model
summarizers = true   # periodic room summarization
judges = true        # judge evaluation + training pairs
forecasts = true     # sensory predictions
fake_user = true     # simulated user responses

[llm]
backend = "hf"                  # "hf" (Qwen) or "rwkv"
model_name = "Qwen/Qwen3-8B"
device = "cuda:0"

[t5]
device = "cuda:1"

[training]
learning_rate = 1e-4
batch_size = 4
epochs = 1
```

## Logs

Each run produces two log files in `outputs/logs/`:

- `YYYYMMDD_HHMMSS.log` — full internal log (all room messages, judges, summaries)
- `YYYYMMDD_HHMMSS_external.log` — external view only (user + assistant messages)

## Deploy

```bash
./scripts/deploy.sh    # rsync to Lambda training server
# then on server:
uv sync
./scripts/launch_experiment.sh
```

## Project Structure

```
main.py                     # entry point, logging, callbacks
config.toml                 # single source of truth for all parameters
src/minsky/
  orchestrator.py           # message routing, cycle management
  rooms.py                  # room processors (dual-agent logic)
  types.py                  # Message, RoomState, RoomType
  prompts/
    rooms.py                # LLM prompts for each room (RWKV + chat)
    t5.py                   # T5 edit model prompts per room
    summarizer.py           # summarizer prompt
    forecast.py             # forecast prompt
    judges.py               # judge evaluation prompts
  edit_model.py             # T5Gemma model, trainer, checkpoints
  judges.py                 # judge/summarizer/fake_user (DeepSeek API)
  tools.py                  # web_search, memory_*, scratchpad_*
  llm_client.py             # HF and RWKV backends
  memory.py                 # FSRS-6 long-term memory system
scripts/
  train_t5.py               # standalone T5 training
  deploy.sh                 # rsync to training server
  download_model.py         # download RWKV/T5 models
  launch_experiment.sh      # start run on server
```
