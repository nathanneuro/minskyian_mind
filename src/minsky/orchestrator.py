"""Orchestrator for the Society of Mind architecture.

The orchestrator coordinates message passing between rooms and manages
the multi-cycle communication loop. All rooms process in parallel as
a batch through RWKV, then through T5 edit models.
"""

from dataclasses import dataclass, field
from typing import Callable, Any

from minsky.types import Message, RoomType, RoomState, MessageType
from minsky.rooms import create_room_state
from minsky.judges import JudgeInput, JudgeOutput, build_judge_batch, parse_judge_batch
from minsky.edit_model import TrainingPair, save_training_pairs


@dataclass
class BatchedLLM:
    """Handles inference using the official rwkv pip package.

    Uses CUDAGraph acceleration for fast token generation.
    Processes prompts sequentially (RWKV doesn't batch well due to state).
    """

    client: Any = None
    config: Any = None
    _initialized: bool = False

    def initialize(self, config=None) -> None:
        """Load the shared RWKV model."""
        if self._initialized:
            return

        from minsky.llm_client import RWKVClient, RWKVConfig

        rwkv_config = RWKVConfig()
        if config and hasattr(config, 'model_path'):
            rwkv_config.model_path = config.model_path
        if config and hasattr(config, 'strategy'):
            rwkv_config.strategy = config.strategy

        self.client = RWKVClient(config=rwkv_config)
        self.client.initialize()
        self.config = config
        self._initialized = True

    def generate_batch(
        self,
        prompts: list[str],
        max_tokens: int = 256,
        temperature: float = 1.0,
        noise: float = 0.0,  # Not used, kept for API compatibility
    ) -> list[str]:
        """Generate responses for multiple prompts.

        Note: RWKV processes sequentially due to stateful nature.
        CUDAGraph acceleration makes each generation fast.

        Args:
            prompts: List of prompts to process.
            max_tokens: Max tokens per response.
            temperature: Sampling temperature.
            noise: Ignored (kept for API compatibility).

        Returns:
            List of generated responses (same order as prompts).
        """
        if not self._initialized:
            self.initialize()

        if not prompts:
            return []

        results = []
        for prompt in prompts:
            response = self.client.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.0,  # Greedy by default
            )
            results.append(response)

        return results


@dataclass
class BatchedEditModel:
    """Handles batched T5 edit inference."""

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

        model_name = config.model_name if config else "google/t5gemma-2-270m-270m"
        device = config.device if config else "cuda:1"
        dtype = config.dtype if config else torch.bfloat16

        print(f"Loading batched T5: {model_name} on {device}")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
        ).to(device)
        self.config = config
        self._initialized = True
        print("Batched T5 loaded.")

    def edit_batch(
        self,
        texts: list[str],
        task_prefixes: list[str],
        contexts: list[str],
        max_new_tokens: int = 256,
    ) -> list[str]:
        """Edit multiple texts in parallel.

        Args:
            texts: List of texts to edit.
            task_prefixes: Task prefix for each text.
            contexts: Context for each edit.
            max_new_tokens: Max tokens per output.

        Returns:
            List of edited texts.
        """
        if not self._initialized:
            self.initialize()

        if not texts:
            return []

        import torch

        # Format inputs
        prompts = []
        for text, prefix, ctx in zip(texts, task_prefixes, contexts):
            if ctx:
                prompts.append(f"{prefix}: context: {ctx} text: {text}")
            else:
                prompts.append(f"{prefix}: {text}")

        # Tokenize batch
        inputs = self.processor(
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        # Decode
        results = []
        for output in outputs:
            text = self.processor.decode(output, skip_special_tokens=True)
            results.append(text)

        return results


@dataclass
class Orchestrator:
    """Coordinates communication between rooms with batched processing.

    Global step flow:
    1. Batch all room prompts through RWKV (GPU 0)
    2. Batch all outputs through T5 edit model (GPU 1)
    3. Route edited outputs to target rooms
    4. Every N steps, run summarizers (RWKV only, no T5)
    """

    sensory_state: RoomState = field(default_factory=lambda: create_room_state(RoomType.SENSORY))
    planning_state: RoomState = field(default_factory=lambda: create_room_state(RoomType.PLANNING))
    motor_state: RoomState = field(default_factory=lambda: create_room_state(RoomType.MOTOR))

    message_queue: list[Message] = field(default_factory=list)
    message_log: list[Message] = field(default_factory=list)
    current_cycle: int = 0  # This is the global step counter
    max_cycles: int = 5

    # Batched processors
    batched_llm: BatchedLLM = field(default_factory=BatchedLLM)
    batched_edit: BatchedEditModel = field(default_factory=BatchedEditModel)
    use_llm: bool = False
    use_edit: bool = False

    # Summarizer settings (RWKV only, no T5)
    use_summarizers: bool = False
    summarizer_interval: int = 10  # Run summarizers every N global steps
    room_summaries: dict = field(default_factory=dict)  # Store summaries per room

    # Legacy per-room LLM functions (for backward compatibility)
    sensory_llm: Callable[[str], str] | None = None
    planning_llm: Callable[[str], str] | None = None
    motor_llm: Callable[[str], str] | None = None

    # Optional action function for Motor
    action_fn: Callable[[str], str] | None = None

    # Judge settings
    use_judges: bool = False
    judge_interval: int = 1  # Run judges every N global steps
    save_training_pairs: bool = True  # Save to data/train_data/
    training_pairs: list[TrainingPair] = field(default_factory=list)
    pending_evaluations: list[tuple[RoomType, str, str]] = field(default_factory=list)  # (room, output, context)

    # Callbacks for visualization/debugging
    on_message: Callable[[Message], None] | None = None
    on_cycle_start: Callable[[int], None] | None = None
    on_cycle_end: Callable[[int, list[Message]], None] | None = None
    on_summarize: Callable[[str, str], None] | None = None  # (room, summary)
    on_judge: Callable[[JudgeOutput], None] | None = None  # Called when judge evaluates

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

    def inject_message(self, message: Message) -> None:
        """Inject a message into the system."""
        message.cycle = self.current_cycle
        self.message_queue.append(message)
        self.message_log.append(message)
        if self.on_message:
            self.on_message(message)

    def run_cycle(self) -> list[Message]:
        """Run a single cycle with batched processing."""
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

        # Collect prompts that need LLM processing
        llm_requests: list[tuple[RoomType, str, Message]] = []

        for room_type, messages in messages_by_room.items():
            state = self.get_state(room_type)
            for msg in messages:
                state.add_message(msg)
                prompt = self._build_prompt(room_type, msg, state)
                if prompt:
                    llm_requests.append((room_type, prompt, msg))
            self.set_state(room_type, state)

        # Batch process through RWKV
        if llm_requests and self.use_llm:
            prompts = [req[1] for req in llm_requests]
            responses = self.batched_llm.generate_batch(prompts)

            # Batch process through T5 edit
            if self.use_edit:
                task_prefixes = [f"edit_{req[0].value}" for req in llm_requests]
                contexts = [req[2].content[:100] for req in llm_requests]
                responses = self.batched_edit.edit_batch(responses, task_prefixes, contexts)

            # Create outgoing messages
            for (room_type, prompt, msg), response in zip(llm_requests, responses):
                outgoing = self._create_response(room_type, msg, response)
                for out_msg in outgoing:
                    out_msg.cycle = self.current_cycle
                    self.message_log.append(out_msg)
                    if self.on_message:
                        self.on_message(out_msg)
                    next_queue.extend(outgoing)

                # Queue for judge evaluation
                if self.use_judges:
                    self.pending_evaluations.append((room_type, response, prompt))
        else:
            # Stub mode - process without LLM
            for room_type, _, msg in llm_requests:
                response = f"[{room_type.value} stub response]"
                outgoing = self._create_response(room_type, msg, response)
                for out_msg in outgoing:
                    out_msg.cycle = self.current_cycle
                    self.message_log.append(out_msg)
                    if self.on_message:
                        self.on_message(out_msg)
                next_queue.extend(outgoing)

        self.message_queue = next_queue
        self.current_cycle += 1

        # Run summarizers every N global steps
        if self.use_summarizers and self.current_cycle % self.summarizer_interval == 0:
            self._run_summarizers()

        # Run judges every N global steps
        if self.use_judges and self.current_cycle % self.judge_interval == 0:
            self._run_judges()

        if self.on_cycle_end:
            self.on_cycle_end(self.current_cycle - 1, external_outputs)

        return external_outputs

    def _run_summarizers(self) -> None:
        """Run summarizer agents for each room (RWKV only, no T5).

        Summarizers compress the message history of each room into
        a concise summary that can be used as context.
        """
        if not self.use_llm:
            return

        # Build summarization prompts for each room
        summarize_prompts = []
        room_types = []

        for room_type in [RoomType.SENSORY, RoomType.PLANNING, RoomType.MOTOR]:
            state = self.get_state(room_type)
            recent_messages = state.get_recent_messages(20)

            if not recent_messages:
                continue

            # Build message history string
            history = "\n".join([
                f"[{m.message_type.value}] {m.content[:200]}"
                for m in recent_messages
            ])

            prompt = (
                f"Summarize the following {room_type.value} room activity in 2-3 sentences. "
                f"Focus on key observations, decisions, and actions.\n\n"
                f"Activity:\n{history}\n\n"
                f"Summary:"
            )
            summarize_prompts.append(prompt)
            room_types.append(room_type)

        if not summarize_prompts:
            return

        # Batch through RWKV only (no T5 edit for summarizers)
        summaries = self.batched_llm.generate_batch(
            summarize_prompts,
            max_tokens=128,
            temperature=0.5,
            noise=0.3,
        )

        # Store summaries
        for room_type, summary in zip(room_types, summaries):
            self.room_summaries[room_type.value] = summary
            if self.on_summarize:
                self.on_summarize(room_type.value, summary)

    def _run_judges(self) -> None:
        """Run judges on pending evaluations (RWKV only, no T5).

        Judges evaluate room outputs and generate counterfactuals
        that become training targets for the T5 edit model.
        """
        if not self.use_llm or not self.pending_evaluations:
            return

        # Build judge inputs
        judge_inputs = [
            JudgeInput(
                room_type=room_type,
                room_output=output,
                context=context,
            )
            for room_type, output, context in self.pending_evaluations
        ]

        # Build prompts and run through RWKV (no T5 edit for judges)
        prompts = build_judge_batch(judge_inputs)
        raw_outputs = self.batched_llm.generate_batch(
            prompts,
            max_tokens=256,
            temperature=0.3,  # Lower temp for more consistent judgments
            noise=0.2,
        )

        # Parse outputs
        judge_outputs = parse_judge_batch(raw_outputs, judge_inputs)

        # Convert to training pairs and store
        for judge_output in judge_outputs:
            # Only create training pair if counterfactual differs from original
            if judge_output.counterfactual != judge_output.original:
                pair = TrainingPair(
                    original=judge_output.original,
                    edited=judge_output.counterfactual,
                    task_prefix=f"edit_{judge_output.room_type.value}",
                    score=judge_output.score,
                )
                self.training_pairs.append(pair)

            # Callback
            if self.on_judge:
                self.on_judge(judge_output)

        # Clear pending evaluations
        self.pending_evaluations = []

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

    def _build_prompt(self, room_type: RoomType, msg: Message, state: RoomState) -> str | None:
        """Build prompt for LLM based on room type and message."""
        match room_type:
            case RoomType.SENSORY:
                if msg.message_type == MessageType.ATTENTION_REQUEST:
                    return (
                        f"Focus attention on: {msg.content}\n"
                        f"Current context: {state.current_context}\n"
                        f"What do you observe?"
                    )
            case RoomType.PLANNING:
                if msg.message_type == MessageType.PERCEPTION:
                    return (
                        f"Given this perception: {msg.content}\n\n"
                        f"1. Generate at least 2 hypotheses explaining this situation.\n"
                        f"2. For each hypothesis, generate a plan.\n"
                        f"3. Estimate the expected value (0-1) of each plan.\n"
                        f"4. Select the highest EV plan.\n\n"
                        f"What is your highest EV plan, stated as a clear instruction?"
                    )
                elif msg.message_type == MessageType.ATTENTION_RESPONSE:
                    return (
                        f"Additional sensory information: {msg.content}\n"
                        f"Previous context: {state.current_context}\n\n"
                        f"Generate the highest EV plan as a clear instruction for Motor."
                    )
            case RoomType.MOTOR:
                if msg.message_type == MessageType.MOTOR_COMMAND:
                    sensory_context = state.metadata.get("sensory_context", "")
                    return (
                        f"Instruction from Planning: {msg.content}\n"
                        f"Sensory context: {sensory_context}\n\n"
                        f"Translate this into a concrete action or response."
                    )
        return None

    def _create_response(self, room_type: RoomType, msg: Message, response: str) -> list[Message]:
        """Create outgoing messages based on room response."""
        outgoing = []

        match room_type:
            case RoomType.SENSORY:
                if msg.message_type == MessageType.ATTENTION_REQUEST:
                    outgoing.append(Message(
                        content=response,
                        source=RoomType.SENSORY,
                        target=RoomType.PLANNING,
                        message_type=MessageType.ATTENTION_RESPONSE,
                    ))

            case RoomType.PLANNING:
                state = self.get_state(RoomType.PLANNING)
                if not state.metadata.get("attention_requested"):
                    # First, request attention
                    outgoing.append(Message(
                        content=f"What are the key details about: {msg.content[:50]}?",
                        source=RoomType.PLANNING,
                        target=RoomType.SENSORY,
                        message_type=MessageType.ATTENTION_REQUEST,
                    ))
                    state.metadata["attention_requested"] = True
                    self.set_state(RoomType.PLANNING, state)
                else:
                    # Send plan to Motor
                    outgoing.append(Message(
                        content=response,
                        source=RoomType.PLANNING,
                        target=RoomType.MOTOR,
                        message_type=MessageType.MOTOR_COMMAND,
                    ))

            case RoomType.MOTOR:
                # Execute action
                if self.action_fn:
                    result = self.action_fn(response)
                else:
                    result = f"[Action completed: {response[:50]}]"

                outgoing.extend([
                    Message(
                        content=response,
                        source=RoomType.MOTOR,
                        target=RoomType.SENSORY,
                        message_type=MessageType.ACTION,
                    ),
                    Message(
                        content=result,
                        source=RoomType.MOTOR,
                        target=RoomType.PLANNING,
                        message_type=MessageType.ACTION_RESULT,
                    ),
                    Message(
                        content=response,
                        source=RoomType.MOTOR,
                        target=RoomType.EXTERNAL,
                        message_type=MessageType.ACTION,
                    ),
                ])

        return outgoing

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
