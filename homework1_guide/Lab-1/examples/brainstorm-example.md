# Example: Completed Problem Brainstorm

**This is an EXAMPLE for calibration. Do not copy these problems. Generate your own.**

**Name:** Example Student
**GitHub Username:** example-student
**Date:** 13 March 2026

---

## H1: Hassle

*What process in your daily life is tedious, error-prone, or needlessly manual?*

**WHO has this problem?**
Small restaurant owners in Tbilisi who keep paper receipts from suppliers and manually enter them into accounting spreadsheets every weekend. Specifically, owners of restaurants with 5-15 employees who do not have a dedicated bookkeeper.

**PROBLEM in one sentence:**
Entering data from 50-100 paper receipts per week into a spreadsheet takes 3-4 hours and produces frequent errors in amounts and vendor names.

**CURRENT SOLUTION:**
They type everything manually on Sunday evenings, or hire a part-time bookkeeper for 200-400 GEL/month. Some photograph receipts and store them in phone galleries, but never actually extract the data. When tax season arrives, they spend a full weekend reconciling errors.

**AI ANGLE:**
Document layout understanding combined with OCR text extraction. A phone app photographs a receipt, extracts vendor name, date, line items, and total using Gemini 3.1 Flash (free tier). Confidence scoring per field is critical: handwritten receipts and faded thermal paper will produce LOW confidence results, which must route to manual entry rather than guessing.

**WHY IT MATTERS:**
3-4 hours per week recovered. That is 150+ hours per year that an owner could spend on the restaurant instead of data entry. Fewer errors means fewer tax reconciliation problems. Lower barrier to keeping clean financial records, which helps small businesses qualify for loans and survive audits.

---

## H2: Hardship

*What problem do you see in your community or country that affects real people?*

**WHO has this problem?**
Elderly patients in rural Georgian villages who receive handwritten prescriptions from visiting doctors. Specifically, patients over 65 in villages more than 30 minutes from a pharmacy, who live alone or with a spouse who also cannot read the handwriting.

**PROBLEM in one sentence:**
Patients leave clinic visits with prescriptions they cannot read and no way to verify what medication they need, what dose to take, or whether it interacts with something they already take.

**CURRENT SOLUTION:**
They call a family member in the city, who tries to decipher the handwriting over a video call. Or they show the paper to the local pharmacy staff who sometimes guess. Occasionally they do not fill the prescription at all.

**AI ANGLE:**
Vision-based OCR for handwritten text recognition, with a critical caveat: handwriting recognition (especially non-Latin scripts) is on the UNRELIABLE list. This means the system design must center on the fallback path. Confidence scoring per recognized word. Any LOW confidence field triggers a pharmacist verification step before displaying results. The app would also cross-reference recognized medication names against a structured drug database to flag potential interactions, but always with a disclaimer and human verification gate.

**WHY IT MATTERS:**
Medication errors from misread prescriptions are a real patient safety issue. Even a partial solution that catches obvious misreads (e.g., "10mg" vs "100mg") could prevent harm. For isolated elderly patients, reducing dependence on family members for basic healthcare navigation preserves dignity and independence.

---

## H3: Horizon

*What new capability does multimodal AI unlock that was impossible 2 years ago?*

**WHO has this problem?**
International students arriving in Georgia (approximately 8,000 per year at Georgian universities) who cannot read Georgian script on street signs, menus, government forms, or utility bills.

**PROBLEM in one sentence:**
Navigating daily life in a country where you cannot read the dominant script means constant dependence on others for basic tasks like reading a menu, understanding a utility bill, or finding a street address.

**CURRENT SOLUTION:**
Google Translate camera mode (inconsistent with Georgian script, especially handwritten or stylized text). Asking friends or classmates. Avoiding situations that require reading, which limits independence and integration. Some students hire other students to help with paperwork.

**AI ANGLE:**
Multimodal pipeline combining real-time vision (point camera at text) with document layout understanding (upload a utility bill or government form). Unlike basic OCR followed by translation, a vision-language model can understand the SEMANTIC LAYOUT of a document: "this field is the due date, this is the amount owed, this is the account number." Georgian script is a specific testing challenge because models trained primarily on Latin and Cyrillic text underperform on Mkhedruli. The app would need to be tested extensively with real Georgian documents, not just English-language benchmarks.

**WHY IT MATTERS:**
Independence. An international student who can read their own utility bill, navigate to an appointment using street signs, and understand a government form without help is a student who integrates faster, stays enrolled longer, and has a better university experience. At scale, this is a retention tool for Georgian universities competing for international students.

---

## Self-Assessment

**Which has the most specific WHO?** H1. I can name 8 restaurant owners personally who deal with this exact problem weekly.

**Which has the strongest AI angle?** H3. The combination of real-time vision, document understanding, and Georgian script handling is genuinely impossible without multimodal AI. No traditional software approach can do this.

**Which would I fight hardest for?** H1, because I have direct access to users and can validate the problem immediately. H3 is more exciting technically but finding 5+ international students to interview would take more effort.

**Discarded idea:** "AI-powered study notes from lecture recordings." Dropped it because Otter.ai and similar tools already do this well. A student team would not meaningfully improve on existing solutions in 15 weeks. Failed the AI Differentiation filter (score 1).
