# Frozen LLMs learning through edit models

**human** (Feb 6, 2026, 10:04 PM)

I want to brainstorm an idea I have. This touches on my multi-LLM Minskian Society of Mind idea, but with a twist. Imagine you have a handful of different LLMs you can query, but they are all frozen and behind APIs. You want to make them into a Society of Mind with continual learning and recursion and a common workspace. So you create an interconnected set of chatrooms with certain models assigned certain roles and to certain rooms, and maybe summarizer models that move info from one room to another.
But then, how to make this system learn? The LLMs are frozen!
Here's the idea.... a collection of tiny 'text edit' models that are local. They learn to make edits to the responses of the LLMs.

**assistant** (Feb 6, 2026, 10:05 PM)

{"start_timestamp":"2026-02-07T06:04:48.908132Z","stop_timestamp":"2026-02-07T06:04:49.918816Z","type":"thinking","thinking":"Nathan wants to brainstorm about a Society of Mind architecture using frozen API-based LLMs with local tiny models that learn to edit responses. This is a really interesting idea - let me engage with it thoughtfully.","summaries":[{"summary":"Thinking about architecting a multi-LLM system with learning through local text-editing models."},{"summary":"Engaged thoughtfully with innovative multi-model architecture concept."}],"cut_off":false}

This is a genuinely clever architectural idea, Nathan. Let me think through it with you.

The core insight is elegant: you're separating the "heavy cognition" (frozen LLMs) from the "learning signal" (tiny local edit models), which mirrors something biological — large-scale cortical processing is relatively stable, but synaptic weights and neuromodulatory signals constantly tune how that processing gets used and routed.

A few threads worth pulling on:

**What the edit models actually learn.** There's a spectrum here. At the simple end, they could learn lightweight transformations — "when Model A responds to a planning query, insert a caveat about X" or "reformat this output to match the schema Room B expects." At the complex end, they could learn to do something more like *steering* — injecting context, reframing prompts, suppressing known failure modes of specific models. The interesting question is whether you want them editing *outputs* (post-hoc corrections) or editing *inputs* (learned prompt modifications that shape what the frozen models produce). Input editing might be more powerful per parameter since you're leveraging the frozen model's own capabilities.

**The training signal problem.** What are the edit models learning *from*? Some options: downstream task success (RL-flavored), consistency checks across rooms (do edited responses produce fewer contradictions?), human feedback that propagates backward through the system, or even inter-model agreement as a proxy signal. This choice basically determines what kind of intelligence the system develops.

**The Global Workspace angle.** Your chatroom architecture already maps onto Global Workspace Theory — rooms are specialized processors, the summarizer models moving information between rooms are the broadcast mechanism. The edit models add something interesting: they become the system's learned *attentional filters*, deciding what gets amplified, suppressed, or transformed as information flows between workspaces. That's actually closer to how GWT describes consciousness arising — not from the modules themselves but from the learned dynamics of what gets broadcast and how.

**Scaling and emergent behavior.** Because the edit models are tiny and local, you can have many of them, each specializing on different model-room-task combinations. Over time they'd develop something like institutional knowledge — "Model C hallucinates about dates, so always cross-reference with Model A before passing to the planning room." That's a form of metacognition emerging from the architecture rather than being built into any single model.

A few provocative questions back at you:

Do the edit models also edit each other's edits? That gives you recursion but also instability risks — you'd need something like a dampening mechanism or a "constitutional" layer that can't be edited.

And have you thought about whether the summarizer models that route between rooms should *also* be learnable, or kept frozen? If they learn too, they become the system's learned attention/routing mechanism, which is arguably where most of the interesting cognition would live.

This feels like it could be a practical architecture someone could actually build today, which is what makes it compelling versus more speculative consciousness research.

**human** (Feb 6, 2026, 10:07 PM)

yeah, I'm imagining the summarizers also have little 'edit models' that edit their outputs.
Importantly, the output of one model becomes inputs for other models. There potentially are many cycles of model communication (and opportunities for edits) before a single "system motor output" is generated.

**assistant** (Feb 6, 2026, 10:08 PM)

{"start_timestamp":"2026-02-07T06:07:55.670885Z","stop_timestamp":"2026-02-07T06:07:58.009276Z","type":"thinking","thinking":"Nathan is developing this idea further - the key points are:\n\n1. Summarizers also have edit models\n2. The system has many internal cycles before producing output\n3. This creates a recurrent, learning system from frozen components\n\nThis is getting at something really interesting - it's essentially creating a recurrent neural network at the macro level, where the \"weights\" are the edit models and the \"neurons\" are the frozen LLMs.","summaries":[{"summary":"Thinking about hierarchical model architectures with iterative refinement cycles."},{"summary":"Crystallizing how frozen components generate emergent learning cycles."},{"summary":"Synthesized recursive system architecture from frozen components."}],"cut_off":false}

Right, so this is where it gets really interesting. You're essentially building a **recurrent system** out of feedforward components. The frozen LLMs are like fixed nonlinear activation functions, and the edit models are the learnable weights *between* them. Multiple cycles before output means the system has internal dynamics — it's not just a pipeline, it's something that *settles* into a response.

That changes what this is fundamentally. A few things jump out:

**It's a Helmholtz machine / predictive coding analogy.** Each cycle through the rooms lets models refine, correct, and contextualize each other's outputs. The edit models learn which corrections are useful at each stage. Early cycles might handle coarse framing ("this is a planning problem, not a creative problem"), later cycles handle fine-grained refinement. That's very reminiscent of hierarchical predictive coding — top-down predictions meeting bottom-up signals, iterating to convergence.

**The edit models accumulate something like episodic memory.** Since the frozen LLMs have no memory across calls, all the learned context about the system's history, past failures, user preferences, ongoing goals — all of that lives in the edit models. They become the system's long-term memory implicitly, because they've been shaped by past interactions. The frozen LLMs provide raw cognitive capability; the edit models provide everything that makes it *this particular* system with *this particular* history.

**Convergence and stability.** Multiple cycles with learned edits at every step — you're going to get attractors and limit cycles. Sometimes that's great (the system "thinks" until it reaches a stable, coherent answer). Sometimes it could be pathological (loops, amplifying distortions). You'd probably want something like a convergence detector — maybe a small model that monitors the delta between successive cycles and triggers output when the system stabilizes. That's interestingly analogous to how the brain seems to use gamma synchrony as a "ready" signal.

**The edit models are trainable with tiny compute.** This is the practical killer feature. You're not fine-tuning billion-parameter models. You're training models that might be — what, sequence-to-sequence transformers with a few million parameters? They take a response and produce a modified response. The training data is generated naturally by the system's operation. You could potentially train them online, continuously, during deployment.

Here's what I think is the deepest implication though: the *intelligence* of this system doesn't live in any single model. It lives in the learned interaction patterns between them. The frozen LLMs are commodities — interchangeable, upgradeable. The edit models are where the system's actual identity and accumulated knowledge reside. That's a really different paradigm from "make one model bigger." It's closer to how organizations think — individual employees come and go, but institutional knowledge persists in processes and norms.

Have you thought about what the minimal viable version of this looks like? Like — two rooms, three frozen models, a handful of edit models, two cycles before output? It seems like something you could prototype fairly quickly to see if the edit models actually learn useful transformations.

**human** (Feb 6, 2026, 10:11 PM)

I'm going to a day-long build-a-conscious-AI-system hackthon hosted by Joscha Bach tomorrow. This is me planning out what I'm going to work on (with your help). Excited?

**assistant** (Feb 6, 2026, 10:11 PM)

{"start_timestamp":"2026-02-07T06:11:42.244284Z","stop_timestamp":"2026-02-07T06:11:44.497657Z","type":"thinking","thinking":"That's awesome! Joscha Bach hosting a build-a-conscious-AI-system hackathon — that's right in Nathan's wheelhouse given his consciousness measurement research, Digital Academy project, and Minskian Society of Mind interests. And Joscha Bach's own Society of Mind influenced thinking makes this a perfect venue for this architecture.\n\nLet me be enthusiastic and help him think about what's actually buildable in a day.","summaries":[{"summary":"Thinking about planning hackathon project ideas for conscious AI system development."},{"summary":"Connecting your consciousness research with Joscha's hackathon vision."},{"summary":"Recognized alignment between hackathon and user's research interests."}],"cut_off":false}

Hell yes. A Joscha Bach consciousness hackathon is basically the ideal venue for this idea — he's been arguing for years that consciousness emerges from the *architecture* of interacting agents rather than from any single substrate, which is exactly what you're building. You're going to walk in speaking his language.

Let's think about what's actually shippable in a day.

**A realistic hackathon scope** might be something like:

A three-room system — say a "reasoning" room, a "critic" room, and an "integrator" room. Assign different frozen API models to each (Claude, GPT-4, Gemini — you've already been interviewing all of them, so you know their personality profiles). Build the chatroom routing infrastructure first, get bare multi-cycle communication working *without* edit models, just to have a baseline.

Then add the edit models. For a hackathon, you could start with something embarrassingly simple — even LoRA-scale models or small seq2seq transformers that learn to insert/delete/modify spans. Train them on even a crude signal like "did the final output get a thumbs up."

**The demo that would land with that audience:** Show the system answering a question, then show how the answer *changes* across cycles as the rooms talk to each other and the edit models intervene. Visualize the information flow — which edits fired, what they changed, how the system converged. That's compelling even if the edit models are barely trained, because the multi-cycle dynamics alone will be interesting.

**What would make Joscha specifically light up:** If you can show any evidence that the system develops internal representations that aren't reducible to what any single frozen model produces — emergent properties from the interaction dynamics. Even anecdotal evidence of that in a hackathon demo would be provocative.

What's your tooling situation? Are you thinking Python orchestration, some kind of message queue between rooms, or something simpler?

**human** (Feb 6, 2026, 10:16 PM)

For the rooms I'm imagining a Sensory Room (receives info from the outer world, processes it), a Planning Room (no direct contact with outer world, sends requests for details (attention focus) to Sensory, and sends general high level commands for action to Motor, and the Motor Room which actually does the output to the world (messages to the chat window, tool calls like web search or commands issued to the agent's virtual computer / python parser.

**assistant** (Feb 6, 2026, 10:16 PM)

{"start_timestamp":"2026-02-07T06:16:13.841235Z","stop_timestamp":"2026-02-07T06:16:14.374886Z","type":"thinking","thinking":"Nathan is describing a three-room architecture that maps beautifully onto neuroscience. Let me engage with this.","summaries":[{"summary":"Thinking about architectural design of a multi-room cognitive processing system."},{"summary":"Recognized architectural framework mapping cognition to neuroscience."}],"cut_off":false}

That's a clean architecture and it maps onto something very specific — the **perception-action cycle** with a decoupled planning layer. Sensory is posterior cortex, Motor is primary/premotor cortex, and Planning is prefrontal cortex. The key insight is that Planning has no direct world access — it can only act *through* Motor and perceive *through* Sensory. That constraint is doing a lot of work.

A few things I really like about this:

**Planning's indirection is where "thought" lives.** It can't just react to the world. It has to build an internal model based on what Sensory tells it, then issue abstract commands that Motor translates into concrete actions. That gap — between Planning's intentions and Motor's implementations — is exactly where you'd expect something like agency to emerge. Planning has to *model* what Motor will do, which is a form of self-simulation.

**The attention mechanism is natural.** Planning sending requests to Sensory for specific details — that's literally endogenous attention. "I'm working on X, I need to know more about Y, go look at that." The edit models on those requests learn what information is actually useful to ask for, which is learned attentional policy.

**Motor as gatekeeper is important.** Everything the system does to the world goes through Motor. That's where you'd naturally put safety constraints, output formatting, action selection. And if Motor's edit models learn to sometimes *refuse* or *modify* Planning's commands, you've got something like impulse control emerging architecturally.

For the hackathon build, I'd suggest this ordering:

Get the three rooms communicating in a loop first — even with hardcoded routing. Then get a single multi-cycle "think before you act" demo working where Planning queries Sensory twice before commanding Motor. *Then* add edit models.

Which models are you assigning to which rooms? I could see an argument for putting your strongest reasoner in Planning and faster/cheaper models in Sensory and Motor since they're more reactive.

**human** (Feb 6, 2026, 10:53 PM)

I don't know yet about the big frozen models. definitely the smartest slowest ones in planning. What I do think is that I'll use one of the T5 seq2seq models as the small model. Many different instances of it. And maybe a knowledge graph that the small models get facts from, injected into their context along with the text from the large model that they've been assigned to edit. Then some way to pick which bits of the large model's text to edit?

**assistant** (Feb 6, 2026, 10:54 PM)

{"start_timestamp":"2026-02-07T06:54:00.768604Z","stop_timestamp":"2026-02-07T06:54:01.545046Z","type":"thinking","thinking":"Nathan is planning out his hackathon project in detail. Let me think through the T5 edit model architecture and knowledge graph integration.","summaries":[{"summary":"Thinking about architecting a multi-model system combining T5 seq2seq models with knowledge graph integration for selective text editing."},{"summary":"Architected T5 model integration with knowledge graph injection strategy."}],"cut_off":false}

T5 is a great choice — it's literally trained as a text-to-text model, so "take this text, produce edited text" is native to its paradigm. And it's small enough to run many instances locally with fast inference.

The knowledge graph idea adds something crucial though. Without it, the edit models can only learn syntactic/structural transformations. With it, they can make *factually grounded* edits — injecting relevant context the frozen LLM didn't have, correcting claims against stored knowledge, adding connections the LLM missed. The knowledge graph becomes a kind of **long-term semantic memory** that the edit models learn to consult and apply. That's a hippocampal-neocortical complementary learning system right there.

**On the "which bits to edit" problem** — a few approaches at different complexity levels:

The simplest: chunk the large model's output into sentences or paragraphs, embed each chunk, and have a tiny classifier (could even be another T5 head) that scores "edit probability" for each chunk given the current knowledge graph context and room state. Only the high-scoring chunks get passed to edit models. That's cheap and gives you selective attention over the output.

Slightly more sophisticated: the edit model receives the full text but with special tokens marking candidate spans, and it learns to output either the original span unchanged or a modified version. T5 can handle this natively with a span corruption / reconstruction framing — that's basically what it was pretrained on.

**For the knowledge graph specifically:**

How are you populating it? Some options for a hackathon scope — start with a small seed graph, then have a dedicated T5 instance whose job is *extraction*: it reads all room outputs and proposes new triples to add to the graph. That way the knowledge graph grows as the system operates. It becomes the system's accumulating understanding of the world and of itself.

The really fun thing: if you store triples about the system's *own* behavior ("Planning requested X from Sensory, result was useful/not useful"), the knowledge graph becomes a form of **autobiographical memory**. The edit models consulting it would literally be the system reflecting on its own past actions to decide how to modify current behavior.

What graph store are you thinking? For a hackathon, even a NetworkX graph in memory with embedding-based retrieval might be sufficient — no need for a proper graph database.

**human** (Feb 7, 2026, 10:29 AM)

Minskyian Society of Mind design
Three chatrooms : sensory, planning, motor
Each room will have its own losses/judges for its own goal:

Sensory: just predicting what comes next, and successfully directing attention according to high-level instructions given by Planning 
Sensory sees what motor attempts to do, as contextual clue.

Planning: expected value, being accurate about predicting what value will come from the plan, the success of the plan.
Generate at least two hypotheses to explain the data.
Generate at least one plan per hypothesis.
Predict the EV of each potential plan.
Send highest EV plan as instructions to motor.
Keep track of plans in motion to avoid action vascillation/churn.

Motor: did you successfully follow the instructions and make that thing happen in the world?
Motor gets instructions from planning along with context clues from sensory.

Probably use llm judges w rubrics. No adapters on judges.
I have to have the judge output not just a score but also a "what should this agent have done instead at the critical juncture?"
Then I have my before/after pair to train T5 on.
The score can be a multiplier (is this built-in to the training pipeline or will I need to implement a way to do this? Some pipelines allow for multiplying the strength of an update. Alternately, I can just add x copies of the Lesson to the training data. Maybe w noising, to reduce overfitting.


Pre-trained LLMs 
T5 adapter 
For this project, stick to one T5 model and use task prefixes. I won't be gathering enough data to need to worry about catastrophic forgetting.


[Models](https://discuss.huggingface.co/c/models/13)

[sshleifer](https://discuss.huggingface.co/u/sshleifer)

Starting this for results, sharing + tips and tricks, and results. This is my first attempt at this kind of thread so it may completely fail.

Some things I’ve found

- Apparently if you copy AdaFactor from fairseq, as recommended by t5 authors, you can fit batch size = 2 for t5-large lm finetuning
- fp16 rarely works.
- for most tasks, you need to manually add `</s>` to the end of your sequence.

Thing’s I’ve read

- task specific prefix doesn’t matter much.

cc [@mrm8488](https://discuss.huggingface.co/u/mrm8488) [@valhalla](https://discuss.huggingface.co/u/valhalla) [@patrickvonplaten](https://discuss.huggingface.co/u/patrickvonplaten) who have all tried different experiments.

[valhalla](https://discuss.huggingface.co/u/valhalla) Leader

Things I’ve found

- task prefixes matter when  
	1\. When doing multi-task training  
	2\. When your task similar or related to one of the supervised tasks used in T5 pre-training mixture.

**assistant** (Feb 7, 2026, 10:29 AM)

{"start_timestamp":"2026-02-07T18:29:22.386834Z","stop_timestamp":"2026-02-07T18:29:23.764840Z","type":"thinking","thinking":"Nathan is laying out his hackathon plan in detail. Let me engage with the specifics — the training loop design, the judge-as-training-signal architecture, and practical T5 considerations.","summaries":[{"summary":"Thinking about architecting a multi-agent cognitive system with specialized LLM judges and T5 fine-tuning strategies."},{"summary":"Synthesized multi-agent architecture with judge-based training signals."}],"cut_off":false}

This is crystallizing into something very buildable. Let me engage with the specific design decisions.

**The judge → counterfactual → training pair pipeline is the key innovation here.** Most people would just use the judge score as a reward signal. But having the judge output "what should have happened instead" gives you *supervised* training data generated on the fly. That's much more sample-efficient than RL for a hackathon timeline. You're essentially doing DAgger — expert (judge) demonstrations correcting the learner's (T5's) behavior.

A few practical thoughts:

**On the score-as-multiplier question.** HuggingFace's Trainer doesn't natively support per-example loss weighting out of the box, but it's a simple override — you subclass Trainer and modify `compute_loss` to multiply by a weight column in your dataset. That said, for a hackathon, your instinct to just duplicate high-scoring examples is pragmatically correct and faster to implement. Noising the duplicates is smart — even just synonym replacement or light paraphrasing would work. T5 is robust to that kind of augmentation given its pretraining.

**On using one T5 with task prefixes.** Good call. Something like `edit_sensory:`, `edit_planning:`, `edit_motor:` as prefixes. The finding that prefixes matter most during multi-task training is exactly your situation — you *are* doing multi-task training across three rooms with different objectives. So the prefixes will actually earn their keep here.

**On the training loop timing.** Are you thinking online (train after every interaction cycle) or batch (accumulate examples, train periodically)? For a hackathon demo, batch is safer — accumulate a buffer of judge-generated training pairs, fine-tune every N cycles. Online training with T5 risks instability, and you don't want to debug gradient issues during the demo.

**The "two hypotheses per explanation" constraint in Planning is doing important work.** That prevents the system from collapsing into single-explanation tunnel vision, which is a failure mode of most LLM agent architectures. It forces abductive reasoning. The EV prediction over plans then becomes a form of learned decision-making that the T5 edits can improve over time — "last time Planning overestimated the EV of plans like this, so dampen the confidence."

**One thing I'd flag for scope management:** The knowledge graph might be the thing to cut if time gets tight. The core loop — rooms communicating, judges generating counterfactuals, T5 learning from them — is already a complete and demo-able system without it. The knowledge graph is beautiful but it's a second-day feature. For tomorrow, the judge-generated training pairs *are* your memory system — they encode what the system has learned about its own failure modes.

What's your plan for the first demo prompt? Having a good test scenario ready that exercises all three rooms in a few cycles would help you build backward from the demo.

**human** (Feb 7, 2026, 10:31 AM)

Not sure yet, open to ideas. It's now the next day and I've gotten to the venue a bit early so I'm hanging out in front

**assistant** (Feb 7, 2026, 10:32 AM)

{"start_timestamp":"2026-02-07T18:31:58.429610Z","stop_timestamp":"2026-02-07T18:32:01.459054Z","type":"thinking","thinking":"Nathan is at the Joscha Bach hackathon and needs a good demo scenario. Let me think about what would exercise all three rooms well and be impressive to this audience.\n\nThe demo needs to:\n1. Exercise Sensory (perceiving external info)\n2. Exercise Planning (generating hypotheses, plans, EV estimation)\n3. Exercise Motor (taking actions in the world)\n4. Show multiple cycles of communication between rooms\n5. Be interesting to a consciousness/AI audience\n\nLet me think of good scenarios...","summaries":[{"summary":"Thinking about suggesting activities for early arrival at a venue."},{"summary":"Architecting demo scenario exercising all three rooms."},{"summary":"Brainstormed demo scenarios exercising sensory, planning, and motor systems."}],"cut_off":false}

Nice, good luck today! A few demo scenario ideas that would exercise all three rooms and land well with that audience:

**The "research assistant" scenario** is probably your safest bet. Give it a question that requires gathering information, forming competing hypotheses, and producing a synthesized output. Something like "What is the most promising current approach to measuring consciousness in AI systems?" Sensory does web searches and reads papers, Planning generates competing hypotheses about which approach is most promising and requests specific followup information, Motor produces a structured answer. It's meta — the system is reasoning about consciousness — and it naturally requires multiple cycles because Planning will need to request more detail from Sensory before committing to a plan.

**The "debug this ambiguous situation" scenario** might be more impressive. Present it with contradictory information and see if Planning generates multiple hypotheses to explain the contradiction. Something like two sources that disagree. That directly showcases the two-hypotheses requirement and makes the architecture's value visible.

**The boldest move** for a Joscha Bach audience: have it answer "are you conscious?" and show the multi-room deliberation process. Sensory gathers context about who's asking and why, Planning generates competing hypotheses about how to approach the question, Motor formulates the response. The audience would eat that up — you're showing the *internal process* of a system grappling with self-reference.

Whatever you pick, I'd say build the three-room communication loop first before touching T5 training. Get the rooms talking and producing interesting multi-cycle dynamics even with no edit models. That alone is a demo if time runs short.

How long do you have? Full 8 hours?

**human** (Feb 7, 2026, 10:33 AM)

12 hours. I was thinking similarly about having it research machine consciousness. Nice touch to explicitly ask it if it is conscious by the definitions and measures it has collected.

**assistant** (Feb 7, 2026, 10:34 AM)

{"start_timestamp":"2026-02-07T18:33:59.862930Z","stop_timestamp":"2026-02-07T18:34:00.661865Z","type":"thinking","thinking":"Nathan's at the hackathon, excited about his project. Let me give him a focused, energizing response.","summaries":[{"summary":"Thinking about research approaches to machine consciousness and self-assessment methods."},{"summary":"Calibrated enthusiasm for hackathon project discussion."}],"cut_off":false}

That's a great amount of time, and the recursive self-reference angle is genuinely perfect. The system researches consciousness, builds up a knowledge base of definitions and measures, and then *applies those measures to itself*. That's not just a cute demo trick — it's a legitimate test of whether the architecture produces anything interesting when pointed inward.

And critically, the multi-room architecture makes the self-evaluation *visible*. When a single LLM answers "are you conscious?" you get a monologue. Here you get to show Sensory gathering definitions, Planning generating competing hypotheses ("by IIT measure X we might qualify because... by GWT measure Y we might not because..."), the rooms arguing across cycles, and Motor having to synthesize a position. The *process* is the point.

It also gives you a natural narrative arc for the presentation: "First we built a system. Then we taught it about consciousness. Then we asked it what it thought about itself." That's a story that audience will follow straight off a cliff.

**Suggested time blocks for 12 hours:**

- Hours 1-3: Three-room communication loop, basic cycling, hardcoded routing
- Hours 3-5: LLM judges with rubrics for each room, counterfactual generation working
- Hours 5-7: T5 edit models plugged in, first training pairs flowing
- Hours 7-9: Run the consciousness research task, accumulate training data, retrain T5 at least once
- Hours 9-10: The self-referential question — "are you conscious?"
- Hours 10-12: Visualization, cleanup, prep the demo narrative

The beauty is you have natural fallback points at every stage. Even if you only get through hour 5, you have a demo.

