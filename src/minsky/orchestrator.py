"""Orchestrator for the Society of Mind architecture.

The orchestrator coordinates message passing between rooms and manages
the multi-cycle communication loop. Uses room processors from rooms.py
for actual message processing logic.
"""

import re
from dataclasses import dataclass, field
from typing import Callable, Any

from minsky.types import Message, RoomType, RoomState, MessageType
from minsky.rooms import create_room_state, ROOM_PROCESSORS
from minsky.judges import JudgeInput, JudgeOutput, judge_batch, summarize_batch, fake_user_respond, AgentConfig, configure_agents
from minsky.edit_model import TrainingPair, save_training_pairs
from minsky.prompts.summarizer import SUMMARIZER_PROMPT_TEMPLATE
from minsky.prompts.forecast import FORECAST_PROMPT_TEMPLATE
from minsky.prompts.t5 import format_t5_prompt


@dataclass
class LLMWrapper:
    """Wraps LLM client to provide llm_fn interface for room processors.

    Delegates to either HFClient or RWKVClient based on config.backend.
    """

    client: Any = None
    config: Any = None
    _initialized: bool = False

    def initialize(self, config=None) -> None:
        """Load the LLM model (HF or RWKV based on config.backend)."""
        if self._initialized:
            return

        from minsky.llm_client import LLMConfig, create_llm_client

        llm_config = config if isinstance(config, LLMConfig) else LLMConfig()
        self.client = create_llm_client(llm_config)
        self.client.initialize()
        self.config = llm_config
        self._initialized = True

    @property
    def use_chat_template(self) -> bool:
        """Whether the active backend uses chat-style prompts."""
        if self.config:
            return self.config.use_chat_template
        return False

    def __call__(self, prompt: str) -> str:
        """Generate text from a prompt."""
        if not self._initialized:
            self.initialize()

        return self.client.generate(
            prompt,
            max_tokens=self.config.max_tokens if self.config else 256,
            temperature=self.config.temperature if self.config else 0.7,
            top_p=self.config.top_p if self.config else 0.9,
            stop_tokens=["\n---", "\n\n\n", "###", "INPUT:", "Example"],
        )

    def save_state(self, metadata: dict | None = None) -> None:
        """Save LLM state/metadata to disk."""
        if self._initialized and self.client:
            self.client.save_state(metadata)

    def get_saved_metadata(self) -> dict:
        """Get metadata loaded from the state file."""
        if self._initialized and self.client:
            return self.client.saved_metadata
        return {}


# Backward-compat alias
RWKVWrapper = LLMWrapper


@dataclass
class T5EditWrapper:
    """Wraps T5 edit model to provide edit_fn interface for room processors."""

    model: Any = None
    processor: Any = None
    config: Any = None
    _initialized: bool = False

    def initialize(self, config=None) -> None:
        """Load the T5Gemma model."""
        if self._initialized:
            return

        from transformers import AutoProcessor, AutoModelForSeq2SeqLM
        import torch
        from pathlib import Path

        # Try local path first, then HuggingFace
        local_path = Path(__file__).parent.parent.parent / "data" / "models" / "t5gemma"
        if local_path.exists():
            model_name = str(local_path)
            print(f"Loading T5 from local: {model_name}")
        else:
            model_name = config.model_name if config else "google/t5gemma-2-270m-270m"
            print(f"Loading T5 from HuggingFace: {model_name}")

        device = config.device if config else "cuda:1"
        # T5 works better with float32
        dtype = config.dtype if config else torch.float32

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
        ).to(dtype=dtype, device=device)  # Explicit dtype cast ensures all params/buffers match
        self.config = config
        self._initialized = True
        print(f"T5 edit model loaded on {device} (dtype={dtype})")

    def __call__(self, text: str, context: str = "", task_prefix: str = "edit") -> str:
        """Edit text with context (edit_fn interface)."""
        if not self._initialized:
            self.initialize()

        import torch

        prompt = format_t5_prompt(text, context, task_prefix)

        # Tokenize
        inputs = self.processor(
            text=prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )

        return self.processor.decode(outputs[0], skip_special_tokens=True)


@dataclass
class PendingForecast:
    """A forecast awaiting resolution against reality."""
    cycle: int
    raw_forecast: str      # RWKV raw output (becomes training `original`)
    context: str           # What forecast was based on


@dataclass
class Orchestrator:
    """Coordinates communication between rooms.

    Uses room processors from rooms.py for message processing.
    Each room processor handles its own prompts and response parsing.

    Global step flow:
    1. Route messages to target rooms
    2. Each room processor generates prompts, calls LLM, parses responses
    3. Collect outgoing messages and route to next targets
    4. Every N steps, run summarizers (RWKV only, no T5)
    """

    sensory_state: RoomState = field(default_factory=lambda: create_room_state(RoomType.SENSORY))
    planning_state: RoomState = field(default_factory=lambda: create_room_state(RoomType.PLANNING))
    motor_state: RoomState = field(default_factory=lambda: create_room_state(RoomType.MOTOR))

    message_queue: list[Message] = field(default_factory=list)
    message_log: list[Message] = field(default_factory=list)
    current_cycle: int = 0  # This is the global step counter
    max_cycles: int = 5

    # Model wrappers (provide llm_fn and edit_fn interfaces)
    rwkv: LLMWrapper = field(default_factory=LLMWrapper)
    t5_edit: T5EditWrapper = field(default_factory=T5EditWrapper)
    use_llm: bool = False
    use_edit: bool = False

    # What becomes the "accepted truth" routed between rooms:
    #   "raw"      - raw LLM output (skip T5 even if loaded)
    #   "edited"   - T5-edited output (default, current behavior)
    #   "improved" - judge counterfactual replaces routed messages each cycle
    room_output_source: str = "edited"

    # Summarizer settings (RWKV only, no T5)
    use_summarizers: bool = False
    summarizer_interval: int = 10  # Run summarizers every N global steps
    room_summaries: dict = field(default_factory=dict)  # Store summaries per room

    # Optional action function for Motor
    action_fn: Callable[[str], str] | None = None

    # Judge settings
    use_judges: bool = False
    judge_interval: int = 1  # Run judges every N global steps
    save_training_pairs: bool = True  # Save to data/train_data/
    training_pairs: list[TrainingPair] = field(default_factory=list)
    # (room_type, t5_edited, raw_output, context, history)
    pending_evaluations: list[tuple[RoomType, str, str, str, list[tuple[str, str, str]] | None]] = field(default_factory=list)

    # Forecast settings
    use_forecasts: bool = False
    pending_forecasts: list[PendingForecast] = field(default_factory=list)

    # Fake user: DeepSeek simulates user responses to TO_EXTERNAL messages
    use_fake_user: bool = False

    # Callbacks for visualization/debugging
    on_message: Callable[[Message], None] | None = None
    on_cycle_start: Callable[[int], None] | None = None
    on_cycle_end: Callable[[int, list[Message]], None] | None = None
    on_summarize: Callable[[str, str], None] | None = None  # (room, summary)
    on_judge: Callable[[JudgeOutput], None] | None = None  # Called when judge evaluates
    on_fake_user: Callable[[str], None] | None = None  # Called with fake user reply

    def restore_from_saved_state(self) -> None:
        """Restore orchestrator state (e.g. global_step) from LLM saved metadata."""
        meta = self.rwkv.get_saved_metadata()
        if meta.get("global_step"):
            self.current_cycle = meta["global_step"]
            print(f"Restored global step: {self.current_cycle}")

    def get_state(self, room_type: RoomType) -> RoomState:
        """Get state for a specific room."""
        match room_type:
            case RoomType.SENSORY:
                return self.sensory_state
            case RoomType.PLANNING:
                return self.planning_state
            case RoomType.MOTOR:
                return self.motor_state
            case _:
                raise ValueError(f"Unknown room type: {room_type}")

    def set_state(self, room_type: RoomType, state: RoomState) -> None:
        """Update state for a specific room."""
        match room_type:
            case RoomType.SENSORY:
                self.sensory_state = state
            case RoomType.PLANNING:
                self.planning_state = state
            case RoomType.MOTOR:
                self.motor_state = state

    def _extract_history(self, room_type: RoomType, max_messages: int = 8) -> list[tuple[str, str, str]]:
        """Extract recent message history from a room's state as tuples."""
        state = self.get_state(room_type)
        recent = state.get_recent_messages(max_messages)
        return [
            (msg.source.value, msg.message_type.value, msg.content)
            for msg in recent
        ]

    def inject_message(self, message: Message) -> None:
        """Inject a message into the system."""
        message.cycle = self.current_cycle
        self.message_queue.append(message)
        self.message_log.append(message)
        if self.on_message:
            self.on_message(message)

    def run_cycle(self) -> list[Message]:
        """Run a single cycle using room processors from rooms.py.

        Each room processor handles:
        - Building prompts internally
        - Calling llm_fn for generation
        - Parsing structured output (TO_PLANNING, TO_MOTOR, etc.)
        - Creating outgoing messages with proper length limits
        - Tool execution (Motor only)
        """
        if self.on_cycle_start:
            self.on_cycle_start(self.current_cycle)

        external_outputs: list[Message] = []
        next_queue: list[Message] = []

        # Group messages by target room
        messages_by_room: dict[RoomType, list[Message]] = {
            RoomType.SENSORY: [],
            RoomType.PLANNING: [],
            RoomType.MOTOR: [],
        }

        for msg in self.message_queue:
            if msg.target == RoomType.EXTERNAL:
                external_outputs.append(msg)
            elif msg.target in messages_by_room:
                messages_by_room[msg.target].append(msg)

        # Get llm_fn and edit_fn based on settings and room_output_source
        llm_fn = self.rwkv if self.use_llm else None
        if self.room_output_source == "raw":
            # Skip T5 even if loaded — rooms get raw LLM output
            edit_fn = None
        else:
            edit_fn = self.t5_edit if self.use_edit else None

        # Process each room using its processor from rooms.py
        for room_type, messages in messages_by_room.items():
            if not messages:
                continue

            state = self.get_state(room_type)
            processor = ROOM_PROCESSORS[room_type]

            # Call the room processor (it handles prompts, LLM, parsing internally)
            if room_type == RoomType.MOTOR:
                # Motor processor has extra action_fn parameter
                new_state, outgoing, raw_output = processor(
                    state, messages, llm_fn, edit_fn, self.action_fn
                )
            else:
                new_state, outgoing, raw_output = processor(state, messages, llm_fn, edit_fn)

            self.set_state(room_type, new_state)

            # Log and queue outgoing messages
            for out_msg in outgoing:
                out_msg.cycle = self.current_cycle
                self.message_log.append(out_msg)
                if self.on_message:
                    self.on_message(out_msg)

                if out_msg.target == RoomType.EXTERNAL:
                    if out_msg.content:  # Only add non-empty external messages
                        external_outputs.append(out_msg)
                else:
                    next_queue.append(out_msg)

            # Queue for judge evaluation (all three rooms)
            if self.use_judges:
                # Pick first non-empty output as representative for this room
                representative = next(
                    (m.content for m in outgoing if m.content and m.target != RoomType.EXTERNAL),
                    None,
                )
                if representative:
                    history = self._extract_history(room_type)
                    self.pending_evaluations.append(
                        (room_type, representative, raw_output, new_state.current_context, history)
                    )

        # "improved" mode: run judges immediately and replace queued messages
        if self.room_output_source == "improved" and self.use_judges and self.pending_evaluations:
            judge_outputs = self._run_judges()
            for jo in judge_outputs:
                if jo.counterfactual != jo.original:
                    for msg in next_queue:
                        if msg.source == jo.room_type and msg.content == jo.original:
                            msg.content = jo.counterfactual[:256]
                            break

        self.message_queue = next_queue
        self.current_cycle += 1

        # Run summarizers every N global steps
        if self.use_summarizers and self.current_cycle % self.summarizer_interval == 0:
            self._run_summarizers()

        # Run judges every N global steps (skip if "improved" already ran them)
        if self.room_output_source != "improved" and self.use_judges and self.current_cycle % self.judge_interval == 0:
            self._run_judges()

        # Forecasts: resolve old, generate new
        if self.use_forecasts and self.use_llm:
            self._resolve_forecasts()
            self._generate_forecast()

        # Fake user: respond to each TO_EXTERNAL message individually
        # to ensure strict alternation (assistant → user → assistant → …)
        if self.use_fake_user:
            for ext_msg in external_outputs:
                self._run_fake_user(ext_msg)

        if self.on_cycle_end:
            self.on_cycle_end(self.current_cycle - 1, external_outputs)

        return external_outputs

    def _run_summarizers(self) -> None:
        """Run summarizer agents for each room via DeepSeek API.

        Summarizers compress the message history of each room into
        a concise summary. All rooms are summarized concurrently.
        """
        room_types = []
        prompts = []

        for room_type in [RoomType.SENSORY, RoomType.PLANNING, RoomType.MOTOR]:
            state = self.get_state(room_type)
            recent_messages = state.get_recent_messages(20)

            if not recent_messages:
                continue

            history = "\n".join([
                f"[{m.message_type.value}] {m.content[:200]}"
                for m in recent_messages
            ])

            prompt = SUMMARIZER_PROMPT_TEMPLATE.format(
                room_type=room_type.value,
                history=history,
            )
            room_types.append(room_type)
            prompts.append(prompt)

        if not prompts:
            return

        summaries = summarize_batch(prompts)

        for room_type, summary in zip(room_types, summaries):
            self.room_summaries[room_type.value] = summary
            if self.on_summarize:
                self.on_summarize(room_type.value, summary)

    def _run_judges(self) -> list[JudgeOutput]:
        """Run judges on pending evaluations via agent API.

        Judges evaluate room outputs and generate counterfactuals
        that become training targets for the T5 edit model.
        All API calls run concurrently.

        Returns:
            List of JudgeOutput (empty if nothing to evaluate).
        """
        if not self.pending_evaluations:
            return []

        # Build judge inputs (judge evaluates the t5_edited output)
        judge_inputs = [
            JudgeInput(
                room_type=room_type,
                room_output=t5_edited,
                context=context,
                message_history=history,
            )
            for room_type, t5_edited, _raw, context, history in self.pending_evaluations
        ]

        # Build raw_output lookup keyed by index
        raw_outputs = [raw for _, _, raw, _, _ in self.pending_evaluations]

        # Run all evaluations concurrently via agent API
        judge_outputs = judge_batch(judge_inputs)

        # Convert to training pairs and store
        for i, judge_output in enumerate(judge_outputs):
            # Only create training pair if counterfactual differs from original
            if judge_output.counterfactual != judge_output.original:
                pair = TrainingPair(
                    raw=raw_outputs[i],
                    t5_edited=judge_output.original,
                    improved=judge_output.counterfactual,
                    task_prefix=f"edit_{judge_output.room_type.value}",
                    score=judge_output.score,
                )
                self.training_pairs.append(pair)

            # Callback
            if self.on_judge:
                self.on_judge(judge_output)

        # Clear pending evaluations
        self.pending_evaluations = []

        # Save training pairs to disk if enabled
        if self.save_training_pairs and self.training_pairs:
            filepath = save_training_pairs(self.training_pairs)
            print(f"Saved {len(self.training_pairs)} training pairs to {filepath}")
            self.training_pairs = []

        return judge_outputs

    def _run_fake_user(self, external_msg: Message) -> None:
        """Generate a simulated user response to a single TO_EXTERNAL message.

        Called once per external message to ensure strict alternation.
        Uses the judge model (DeepSeek) to simulate a curious user.
        Injects the response as a new perception into the next cycle.
        """
        assistant_text = external_msg.content
        if not assistant_text or not assistant_text.strip():
            return

        # Build brief conversation context from recent external messages
        recent_external = [
            msg for msg in self.message_log[-20:]
            if msg.target == RoomType.EXTERNAL or msg.source == RoomType.EXTERNAL
        ]
        context_parts = []
        for msg in recent_external[-6:]:
            role = "User" if msg.source == RoomType.EXTERNAL else "Assistant"
            context_parts.append(f"{role}: {msg.content[:150]}")
        conversation_context = "\n".join(context_parts)

        user_reply = fake_user_respond(assistant_text, conversation_context)
        if user_reply:
            print(f"  FAKE USER: {user_reply}")
            if self.on_fake_user:
                self.on_fake_user(user_reply)
            self.inject_message(Message(
                content=user_reply,
                source=RoomType.EXTERNAL,
                target=RoomType.SENSORY,
                message_type=MessageType.PERCEPTION,
            ))

    def _generate_forecast(self) -> None:
        """Generate a sensory forecast using RWKV (raw, no T5 edit).

        Gathers recent Motor commands and external events, then asks
        RWKV to predict what Sensory will observe next cycle.
        """
        # Gather recent events from all rooms
        event_parts = []
        for room_type in [RoomType.MOTOR, RoomType.PLANNING, RoomType.SENSORY]:
            state = self.get_state(room_type)
            recent = state.get_recent_messages(4)
            for msg in recent:
                if msg.content:
                    event_parts.append(f"{msg.source.value}->{msg.target.value}: {msg.content[:80]}")

        if not event_parts:
            return

        recent_events = " | ".join(event_parts[-6:])  # Last 6 events, capped
        prompt = FORECAST_PROMPT_TEMPLATE.format(recent_events=recent_events)
        raw_forecast = self.rwkv(prompt)

        # Strip closing tag if RWKV generated it
        raw_forecast = re.sub(r'</forecast>.*', '', raw_forecast, flags=re.DOTALL).strip()

        if raw_forecast:
            self.pending_forecasts.append(PendingForecast(
                cycle=self.current_cycle,
                raw_forecast=raw_forecast,
                context=recent_events,
            ))

    def _resolve_forecasts(self) -> None:
        """Resolve pending forecasts against actual reality.

        For forecasts at least 1 cycle old, compare the prediction
        against what actually happened and create training pairs.
        """
        if not self.pending_forecasts:
            return

        resolved = []
        kept = []

        for forecast in self.pending_forecasts:
            if self.current_cycle - forecast.cycle < 1:
                kept.append(forecast)
                continue

            # Gather actual events from Sensory since the forecast
            sensory_recent = self.sensory_state.get_recent_messages(6)
            actual_parts = []
            for msg in sensory_recent:
                if msg.cycle >= forecast.cycle and msg.content:
                    actual_parts.append(msg.content[:100])

            if actual_parts:
                actual_events = " ".join(actual_parts)
                pair = TrainingPair(
                    raw=forecast.raw_forecast,
                    t5_edited="",
                    improved=actual_events,
                    task_prefix="forecast_sensory",
                    context=forecast.context,
                )
                self.training_pairs.append(pair)
                resolved.append(forecast)
            elif self.current_cycle - forecast.cycle > 3:
                # Too old, discard
                resolved.append(forecast)
            else:
                kept.append(forecast)

        self.pending_forecasts = kept

        # Save forecast training pairs alongside judge pairs
        if self.save_training_pairs and self.training_pairs:
            filepath = save_training_pairs(self.training_pairs)
            print(f"Saved {len(self.training_pairs)} training pairs (incl. forecasts) to {filepath}")
            self.training_pairs = []

    def get_training_pairs(self, clear: bool = True) -> list[TrainingPair]:
        """Get accumulated training pairs for T5.

        Args:
            clear: If True, clear the pairs after returning.

        Returns:
            List of TrainingPair for T5 training.
        """
        pairs = self.training_pairs.copy()
        if clear:
            self.training_pairs = []
        return pairs

    def run_until_output(self, initial_input: str) -> list[Message]:
        """Run cycles until Motor produces output or max_cycles reached."""
        # Inject initial perception
        self.inject_message(Message(
            content=initial_input,
            source=RoomType.EXTERNAL,
            target=RoomType.SENSORY,
            message_type=MessageType.PERCEPTION,
        ))

        # Also send to Planning
        self.inject_message(Message(
            content=initial_input,
            source=RoomType.SENSORY,
            target=RoomType.PLANNING,
            message_type=MessageType.PERCEPTION,
        ))

        all_outputs = []

        while self.current_cycle < self.max_cycles:
            outputs = self.run_cycle()
            all_outputs.extend(outputs)

            # With fake_user, the loop keeps going until max_cycles
            if not self.use_fake_user:
                if outputs and not self.message_queue:
                    break
                if not self.message_queue:
                    break

        return all_outputs

    def get_conversation_log(self) -> str:
        """Get formatted log of all messages."""
        lines = []
        for msg in self.message_log:
            lines.append(f"[Cycle {msg.cycle}] {msg}")
        return "\n".join(lines)

    def shutdown(self) -> None:
        """Gracefully shutdown, saving LLM state and any remaining training pairs."""
        # Flush remaining training pairs to disk
        if self.save_training_pairs and self.training_pairs:
            filepath = save_training_pairs(self.training_pairs)
            print(f"Saved {len(self.training_pairs)} remaining training pairs to {filepath}")
            self.training_pairs = []

        if self.use_llm:
            self.rwkv.save_state({"global_step": self.current_cycle})
            print("LLM state saved.")


def run_cycle(orchestrator: Orchestrator, input_text: str | None = None) -> list[Message]:
    """Convenience function to run a cycle with optional new input."""
    if input_text:
        orchestrator.inject_message(Message(
            content=input_text,
            source=RoomType.EXTERNAL,
            target=RoomType.SENSORY,
            message_type=MessageType.PERCEPTION,
        ))
    return orchestrator.run_cycle()
