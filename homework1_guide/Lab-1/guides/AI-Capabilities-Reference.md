# AI Vision and Multimodal Capabilities Reference

**From Week 2 Lecture: Vision and Multimodal I/O**

Use this reference during brainstorming to identify the specific AI capability that connects to your problem. "AI could help" is too vague. Name the capability.

---

## Reliable Capabilities (Build On These)

These work well enough today to put in a production application.

| Capability | What It Does | Example Use Cases |
|-----------|-------------|-------------------|
| **Object and scene recognition** | Identifies objects, people, settings in photos | Inventory counting, damage detection, quality inspection |
| **OCR and text extraction** | Reads printed text from images | Receipt scanning, document digitization, sign reading |
| **Document layout understanding** | Understands structure of documents (headings, tables, forms, sections) | Invoice processing, form extraction, contract analysis |
| **Image captioning and description** | Generates rich text descriptions of images | Accessibility tools, content cataloging, photo organization |
| **Visual question answering** | Answers specific questions about image content | "What brand is this?" "Is this form signed?" "What color is the roof?" |
| **Chart and graph comprehension** | Extracts data from bar, line, and pie charts | Report analysis, data extraction from presentations, research summaries |
| **Medical image description** | Describes medical images (does NOT diagnose) | Patient record annotation, second-opinion flagging (always with human review) |
| **PDF and document processing** | Reads multi-page PDFs natively (Gemini 3.1) | Textbook analysis, legal document review, academic paper summarization |

## Unreliable Capabilities (Design Around These)

These fail often enough that your app MUST handle failures gracefully.

| Capability | Why It Fails | What This Means for Your App |
|-----------|-------------|------------------------------|
| **Precise spatial coordinates** | Cannot reliably give pixel positions | Do not build features that depend on exact object locations |
| **Counting many items** | Degrades beyond roughly 10 items | Show estimates with confidence ranges, not precise counts |
| **Reading small or stylized text** | Blurry, watermarked, or decorative fonts fail | Pre-screen image quality; ask user to retake if unreadable |
| **3D depth estimation** | Inherently ambiguous from 2D images | Avoid features requiring distance or depth measurement |
| **Subtle emotion recognition** | Cultural context and subtlety routinely missed | Do not build features that rely on detecting emotions from faces |
| **Diagnosing or prescribing** | Always a liability | Route to professionals; never present AI output as medical or legal advice |
| **Consistent identity across images** | Cannot reliably say "this is the same person" | Do not build facial recognition or re-identification features |

---

## Key Models for Student Projects (March 2026)

| Model | Best For | Free Tier? | OpenRouter String |
|-------|---------|-----------|-------------------|
| **Gemini 3.1 Flash** | Prototyping, high-volume vision, PDF processing | Yes (1500 req/day) | `google/gemini-3.1-flash` |
| **GPT-5.2** | Complex document extraction, structured reasoning over images | Paid only | `openai/gpt-5.2` |
| **Claude Opus 4.6** | Legal documents, nuanced visual reasoning, safety-critical analysis | Paid only | `anthropic/claude-opus-4-6` |

**Lab strategy:** Start with Gemini 3.1 Flash (free). Escalate to paid models only when Flash cannot meet your quality bar.

---

## Using This Reference for Brainstorming

When filling in your AI ANGLE field, ask:

1. Which capability from the "Reliable" list most directly addresses my problem?
2. Will my app also depend on any "Unreliable" capabilities? If so, what is my fallback when it fails?
3. Which model should I prototype with? (Almost always: Gemini 3.1 Flash first.)

**Good AI ANGLE example:** "Document layout understanding to extract vendor name, line items, and total from varied invoice formats. Confidence scoring per field. Fallback to manual entry for LOW confidence fields."

**Weak AI ANGLE example:** "AI vision to help with documents."

The first one tells you exactly what to build. The second one tells you nothing.
