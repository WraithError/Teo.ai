# TEO — Architecture

> Architecture before AI. Before UI. Before personality.
> If we can't explain the flow clearly, we don't write the code.

Status: **Phase 1 complete (chat pipeline, FastAPI + HTML/JS). LSTM integrated into nanonet, gradient-checked. Tokenizer + next-token training loop in progress. Phase 2 (Memory) not started.**

---

## 1. System Map

```
TEO
├── Frontend
├── Backend
├── Database
├── AI Layer
├── Memory Layer
├── Project Layer
├── Security Layer
└── Future Modules
```

---

## 2. Message Flow — target, full system

```
User
  ↓
Frontend
  ↓
API
  ↓
Memory Retrieval
  ↓
Context Assembly        ← decides what to pass forward:
  ├── relevant memories   memories, active tool context,
  ├── active module       and which module Teo is routing to
  └── tool context        (chat / code / image / docs / search / ...)
  ↓
Prompt Builder
  ↓
LLM
  ↓
Response Processor
  ↓
Frontend
  ↓
User
```

This is the destination. We do not build this on day one — see Build Order below.

---

## 3. AI Layer — decision: RESOLVED

There are two different things both called "AI" in this project. Keeping them separate avoids a tangle later.

**Decision (made 2026-06-30): pure from-scratch, no external LLM API, for both the language engine and the auxiliary models.** The LSTM in `ai/nanonet/nanonet/layers/lstm.py` is the conversation engine. nanonet handles everything — there is no Claude/GPT/Ollama call anywhere in this stack.

This is a deliberate divergence from one recommendation in the project's review history: the mentor-sourced advice in the audit doc suggested an LLM API for conversation with nanonet reserved for narrow auxiliary jobs (mood classification, security scoring), reasoning that it ships a working product faster. That's a legitimate path — it's just not the one chosen. The goal here isn't "ship Teo fastest," it's "become the kind of engineer who has built a language model from the ground up." Worth a real conversation with the mentor about this explicitly, since the recommendation was made for a reason and deserves more than a one-line override — but the decision as it stands, and the one this document and the rest of the codebase now build toward, is full from-scratch.

**What "the AI Layer" means concretely:**

- **Language Engine** — the LSTM, trained as a next-token-prediction character-level language model on `datasets/train/*.jsonl`, producing Teo's actual conversational output. This replaces the `LLM` box in the Section 2 flow — same position in the pipeline, different implementation. Personality (the 11-step format, mood system, constitution — see `docs/TEO_VOICE.md`) is supplied as conditioning during training and/or prompt-style framing at inference, not bolted on after.
- **nanonet auxiliary models** — same framework, smaller dedicated heads for narrow jobs once Phase 4 (Security) and later phases need them: mood/sentiment classification, code-quality scoring, log anomaly detection. Not decided in detail yet — revisit when Phase 4 starts, not before.

**Known cost of this decision, stated plainly so it doesn't surprise anyone later:** a from-scratch char-LSTM trained on hundreds (eventually thousands) of dialogue pairs, on CPU, will not produce GPT-level fluency. Early output will be repetitive and often ungrammatical. That's expected, not a sign something's broken — it's where every from-scratch language model starts. The payoff is depth of understanding, not speed to a polished demo.

---

## 4. Build Order (locked)

**Phase 0 — This document.**
No code until the flow above can be explained in one sentence, by both of us, without hand-waving.

**Phase 1 — Chat Pipeline only.**
```
User → Teo → AI → Response
```
No memory. No mood. No project management. No deployment. No security audits. One message in, one message out. Nothing else exists yet.

**Phase 2 — Memory Layer.**
```
User → Store Message → Retrieve Context → AI → Response
```

**Phase 3 — Project Layer.**
```
Project
├── Files
├── Notes
├── Memories
└── Settings
```

**Phase 4 — Security Layer.**
This is where Teo stops being "a chatbot" and becomes the thing the reference doc describes:
- Hardcoded secret detection
- `.env` exposure detection
- Debug-mode detection
- Dangerous configuration detection
- Missing error-handling detection

**Phase 5 — Personality.**
Moods, insults, weather reactions, emotional responses, the 11-step format. Added last, on purpose. Personality is the paint. Architecture, chat, memory, and project understanding are the engine — the engine has to run before the paint matters.

**Phase 6+ — Deployment system, dashboard, remote SSH/nginx/TLS.**
(From the Telegram spec.) Out of scope until Phase 1–5 are solid.

---

## 5. Rules

- No production code before this document is finalized and both of us can explain Section 2 in one breath.
- Each phase must work end-to-end before the next one starts. No skipping ahead because a later phase is more fun.
- `main` stays stable-only. Feature work happens on branches.
- Every deletion must be recoverable. Git history is the safety net this time — not a Desktop folder.
