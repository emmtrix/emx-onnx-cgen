const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.3 x 7.5
pres.author = "emmtrix Technologies";
pres.title = "Lessons learned from building an AOT ONNX-to-C compiler";

const W = 13.333, H = 7.5;

// ---- Palette (embedded / compiler theme) ----
const INK   = "0E2A47"; // deep navy
const DEEP  = "0A1F38"; // darker navy
const TEAL  = "16A39A"; // accent
const TEAL2 = "0E7C77";
const AMBER = "E8A13A"; // sharp accent
const SLATE = "33475B"; // body text on light
const MUTED = "6B7C8F";
const LIGHT = "FFFFFF";
const CARD  = "F2F6F8"; // card tint
const CARDLN= "DBE6EA";
const CODEBG= "0C2138";
const CODEFG= "D7E4EC";

const SERIF = "Cambria";
const SANS  = "Calibri";
const MONO  = "Consolas";

const sh = (o = {}) => Object.assign({ type: "outer", color: "0A1F38", blur: 9, offset: 3, angle: 90, opacity: 0.14 }, o);

function footer(slide, n) {
  slide.addText("emx-onnx-cgen  ·  ONNX Community Day", {
    x: 0.6, y: H - 0.45, w: 7, h: 0.3, fontFace: SANS, fontSize: 9, color: MUTED, align: "left", margin: 0,
  });
  slide.addText(String(n), {
    x: W - 1.1, y: H - 0.45, w: 0.5, h: 0.3, fontFace: SANS, fontSize: 9, color: MUTED, align: "right", margin: 0,
  });
}

// kicker label (small teal chip motif repeated across slides)
function kicker(slide, text) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 0.55, w: 0.22, h: 0.22, rectRadius: 0.05, fill: { color: TEAL } });
  slide.addText(text.toUpperCase(), { x: 0.92, y: 0.5, w: 9, h: 0.33, fontFace: SANS, fontSize: 12, bold: true, color: TEAL2, charSpacing: 2, margin: 0, valign: "middle" });
}

function title(slide, text, y = 0.95) {
  slide.addText(text, { x: 0.6, y, w: 12.1, h: 0.95, fontFace: SERIF, fontSize: 30, bold: true, color: INK, margin: 0, valign: "middle" });
}

function codeBox(slide, code, opt = {}) {
  const o = Object.assign({ x: 0.6, y: 2.2, w: 6.0, h: 2.4, fontSize: 12.5 }, opt);
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: o.x, y: o.y, w: o.w, h: o.h, rectRadius: 0.06, fill: { color: CODEBG }, shadow: sh() });
  slide.addText(code, { x: o.x + 0.2, y: o.y + 0.12, w: o.w - 0.4, h: o.h - 0.24, fontFace: MONO, fontSize: o.fontSize, color: CODEFG, align: "left", valign: "top", margin: 0, lineSpacingMultiple: 1.05 });
}

// content card with header + bullets
function card(slide, x, y, w, h, header, bullets, opt = {}) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.07, fill: { color: opt.fill || CARD }, line: { color: CARDLN, width: 1 }, shadow: sh({ opacity: 0.10 }) });
  let cy = y + 0.22;
  if (header) {
    slide.addText(header, { x: x + 0.28, y: cy, w: w - 0.5, h: 0.4, fontFace: SANS, fontSize: 15, bold: true, color: opt.headColor || TEAL2, margin: 0, valign: "middle" });
    cy += 0.5;
  }
  if (bullets && bullets.length) {
    const runs = bullets.map((b, i) => ({ text: b, options: { bullet: { code: "2022", indent: 14 }, color: SLATE, breakLine: true, paraSpaceAfter: 5 } }));
    slide.addText(runs, { x: x + 0.28, y: cy, w: w - 0.5, h: h - (cy - y) - 0.2, fontFace: SANS, fontSize: opt.fontSize || 12.5, color: SLATE, align: "left", valign: "top", margin: 0 });
  }
}

// numbered lesson badge
function badge(slide, x, y, label) {
  slide.addShape(pres.shapes.OVAL, { x, y, w: 0.62, h: 0.62, fill: { color: TEAL }, shadow: sh({ opacity: 0.18 }) });
  slide.addText(label, { x, y, w: 0.62, h: 0.62, fontFace: SERIF, fontSize: 20, bold: true, color: LIGHT, align: "center", valign: "middle", margin: 0 });
}

/* ============================ SLIDE 1 — TITLE ============================ */
{
  const s = pres.addSlide();
  s.background = { color: INK };
  // motif: faint code-flow line ONNX -> C on dark
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.85, y: 1.0, w: 0.28, h: 0.28, rectRadius: 0.06, fill: { color: TEAL } });
  s.addText("LESSONS LEARNED", { x: 1.25, y: 0.95, w: 9, h: 0.4, fontFace: SANS, fontSize: 14, bold: true, color: TEAL, charSpacing: 3, margin: 0, valign: "middle" });

  s.addText("Building an AOT\nONNX-to-C Compiler", { x: 0.85, y: 1.7, w: 11.6, h: 2.2, fontFace: SERIF, fontSize: 52, bold: true, color: LIGHT, margin: 0, lineSpacingMultiple: 1.0 });

  s.addText("emx-onnx-cgen — deterministic, portable C for embedded and resource-constrained systems", {
    x: 0.87, y: 3.95, w: 11, h: 0.6, fontFace: SANS, fontSize: 18, color: "B9CEDD", margin: 0,
  });

  // ONNX -> C flow chips
  const fy = 5.0;
  const chips = [["ONNX model", DEEP], ["emx-onnx-cgen", TEAL2], ["clean C", DEEP], ["embedded target", DEEP]];
  let cx = 0.87;
  chips.forEach((c, i) => {
    const wdt = 0.35 + c[0].length * 0.115;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: cx, y: fy, w: wdt, h: 0.5, rectRadius: 0.1, fill: { color: c[1] }, line: { color: TEAL, width: i === 1 ? 1.5 : 0.75 } });
    s.addText(c[0], { x: cx, y: fy, w: wdt, h: 0.5, fontFace: MONO, fontSize: 12.5, color: i === 1 ? LIGHT : "C7D8E5", align: "center", valign: "middle", margin: 0 });
    cx += wdt;
    if (i < chips.length - 1) {
      s.addText("→", { x: cx, y: fy, w: 0.45, h: 0.5, fontFace: SANS, fontSize: 18, color: TEAL, align: "center", valign: "middle", margin: 0 });
      cx += 0.45;
    }
  });

  s.addText("ONNX Community Day  ·  emmtrix Technologies", { x: 0.87, y: 6.45, w: 11, h: 0.4, fontFace: SANS, fontSize: 13, color: "8AA3B5", margin: 0 });

  s.addNotes("Frame the talk as lessons learned, not a product pitch. The focus is not only what the compiler supports, but what the project taught us about ONNX as input to a deterministic AOT compiler. emx-onnx-cgen is an open-source ahead-of-time compiler that turns ONNX models into portable, deterministic C for deeply embedded, safety-critical and resource-constrained systems.");
}

/* ====================== SLIDE 2 — WHY WE STARTED ====================== */
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  kicker(s, "History  ·  motivation");
  title(s, "Why we started");

  card(s, 0.6, 2.05, 5.85, 4.55, "We first used onnx2c", [
    "It proved that compiling ONNX graphs to C is genuinely useful",
    "It gave a concrete reference for what generated C could look like",
    "It helped surface real requirements for embedded deployment",
  ], { headColor: TEAL2 });

  card(s, 6.85, 2.05, 5.85, 4.55, "But it was not enough for us", [
    "Generated-code quality was hard for our target use case",
    "Not enough control over the structure of the emitted C",
    "Dynamic-dimension support was a major pain point",
    "Embedded C must be readable, deterministic and reviewable",
  ], { fill: "FBF3E7", headColor: "B5781F" });

  s.addText("For embedded systems, generated code is not an implementation detail — it is an artifact to review, analyze, certify, debug and further optimize.", {
    x: 0.6, y: 6.75, w: 12.1, h: 0.5, fontFace: SANS, fontSize: 13, italic: true, color: MUTED, margin: 0,
  });
  footer(s, 2);
  s.addNotes("Start with the practical history. This was not initially a 'build a complete ONNX compiler' project. The first driver was control over generated C. onnx2c demonstrated the value of ONNX-to-C, but generated-code quality and control were insufficient, and dynamic dimensions were the central irritation.");
}

/* ================== SLIDE 3 — DYNAMIC-DIMENSION TRIGGER ================== */
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  kicker(s, "The trigger");
  title(s, "The dynamic-dimension idea: VLA parameters");

  card(s, 0.6, 2.05, 5.4, 3.0, "The pain point", [
    "onnx2c's weak spot was dynamic dimensions",
    "Idea: represent some dynamic dims as C99 VLA parameters",
    "Rank stays static; extents become runtime arguments",
  ]);

  card(s, 0.6, 5.2, 5.4, 1.95, "Why N-D types matter", [
    "float x[N][C] keeps rank + inner extent in the type",
    "float *x leaves structure only as an offset convention",
    "Typed arrays aid readability, review and analysis",
  ], { fill: CARD, fontSize: 12 });

  codeBox(s,
`void model(int N, int C,
           const float x[restrict N][C],
           float       y[restrict N][C]);

/* extents in the ABI, rank known at compile time,
   loop bounds use explicit N, C parameters       */`,
    { x: 6.4, y: 2.05, w: 6.3, h: 2.55, fontSize: 13 });

  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 6.4, y: 4.85, w: 6.3, h: 2.3, rectRadius: 0.07, fill: { color: "FBF3E7" }, line: { color: "EAD9BC", width: 1 } });
  s.addText("What VLAs do NOT solve", { x: 6.65, y: 5.0, w: 5.8, h: 0.35, fontFace: SANS, fontSize: 14, bold: true, color: "B5781F", margin: 0 });
  s.addText([
    { text: "Static memory bounds for temporaries and buffers", options: { bullet: { code: "2022" }, breakLine: true, paraSpaceAfter: 4 } },
    { text: "Dynamic rank", options: { bullet: { code: "2022" }, breakLine: true, paraSpaceAfter: 4 } },
    { text: "Unbounded sequence length and string size", options: { bullet: { code: "2022" }, breakLine: true, paraSpaceAfter: 4 } },
    { text: "Targets that lack VLA support", options: { bullet: { code: "2022" } } },
  ], { x: 6.65, y: 5.45, w: 5.85, h: 1.6, fontFace: SANS, fontSize: 12, color: SLATE, margin: 0, valign: "top" });

  footer(s, 3);
  s.addNotes("VLA parameters are a representation technique for known rank with runtime extents. They keep the ABI explicit and preserve tensor structure in the type — accessing beyond a declared array object is undefined behaviour even when the next row is contiguous, so typed arrays carry more information than flat pointers. But representation is not semantics: VLAs do not bound memory, do not solve dynamic rank, and do not solve unbounded sequences or strings. This experiment justified building a new implementation.");
}

/* ============== SLIDE 4 — FROM EXPERIMENT TO COVERAGE ============== */
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  kicker(s, "The turning point");
  title(s, "From experiment to coverage project");

  const steps = [
    ["1", "Proof of concept", "VLA-based C representation worked on early models"],
    ["2", "Goal expanded", "From prototype to broad ONNX operator & model coverage"],
    ["3", "Became a compiler", "Pass-based AOT pipeline with verification & coverage reports"],
  ];
  const bw = 3.95, gap = 0.28, x0 = 0.6, y0 = 2.25, bh = 3.0;
  steps.forEach((st, i) => {
    const x = x0 + i * (bw + gap);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: y0, w: bw, h: bh, rectRadius: 0.08, fill: { color: CARD }, line: { color: CARDLN, width: 1 }, shadow: sh({ opacity: 0.10 }) });
    badge(s, x + 0.3, y0 + 0.3, st[0]);
    s.addText(st[1], { x: x + 0.3, y: y0 + 1.1, w: bw - 0.6, h: 0.5, fontFace: SANS, fontSize: 17, bold: true, color: INK, margin: 0 });
    s.addText(st[2], { x: x + 0.3, y: y0 + 1.65, w: bw - 0.6, h: 1.1, fontFace: SANS, fontSize: 13, color: SLATE, margin: 0, valign: "top" });
    if (i < steps.length - 1) s.addText("→", { x: x + bw - 0.02, y: y0, w: gap + 0.04, h: bh, fontFace: SANS, fontSize: 22, color: TEAL, align: "center", valign: "middle", margin: 0 });
  });

  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 5.65, w: 12.1, h: 1.05, rectRadius: 0.07, fill: { color: INK } });
  s.addText("A high-coverage ONNX compiler cannot be a pile of special cases — it needs registries, structured lowering, stable IR objects and a verification strategy.", {
    x: 0.95, y: 5.65, w: 11.4, h: 1.05, fontFace: SANS, fontSize: 14.5, italic: true, color: "DCE8F0", margin: 0, valign: "middle",
  });
  footer(s, 4);
  s.addNotes("This is the turning point. Once broad coverage became the goal, the project needed real architecture and testing discipline. Import, normalization, lowering and codegen needed clear boundaries; unsupported cases needed explicit, testable diagnostics; coverage reports became part of the daily workflow.");
}

/* =============== SLIDE 5 — WHAT emx-onnx-cgen DOES =============== */
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  kicker(s, "The project");
  title(s, "What emx-onnx-cgen does");

  card(s, 0.6, 2.05, 6.1, 4.55, "Ahead-of-time ONNX → C", [
    "Compiles ONNX models to portable, deterministic C",
    "Targets deeply embedded & resource-constrained systems",
    "No dynamic allocation, no OS dependency, no runtime",
    "No hidden dispatch or callbacks — explicit loops & layout",
    "Auditable, readable, reproducible builds",
  ]);

  s.addText("Intentionally small public API", { x: 7.0, y: 2.1, w: 5.7, h: 0.4, fontFace: SANS, fontSize: 15, bold: true, color: TEAL2, margin: 0 });
  codeBox(s,
`_Bool model_load(const char *path);

void  model(/* ...inputs... */,
            /* ...outputs... */);

/* tensor IO -> C array parameters
   dynamic dims -> C99 VLAs
   weights      -> optional model.bin */`,
    { x: 7.0, y: 2.55, w: 5.7, h: 2.55, fontSize: 13 });

  // emmtrix flow strip
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.0, y: 5.35, w: 5.7, h: 1.25, rectRadius: 0.07, fill: { color: CARD }, line: { color: CARDLN, width: 1 } });
  s.addText("Frontend of the emmtrix embedded-AI flow", { x: 7.2, y: 5.45, w: 5.3, h: 0.35, fontFace: SANS, fontSize: 12.5, bold: true, color: INK, margin: 0 });
  s.addText("ONNX → clean C → Vectorizer → target architecture", { x: 7.2, y: 5.85, w: 5.3, h: 0.6, fontFace: MONO, fontSize: 12.5, color: TEAL2, margin: 0, valign: "top" });

  footer(s, 5);
  s.addNotes("Keep this concise — just enough context before the lessons. AOT ONNX-to-C, portable and deterministic, for embedded targets, avoiding dynamic memory and external runtimes. The generated API is deliberately tiny: a load function and a model function. C is not only the output language; it is the handoff IR into emmtrix's source-to-source vectorizer and safety-critical toolchains (ISO 26262, DO-178C contexts).");
}

/* ============ SLIDE 6 — ARCHITECTURE SHAPED BY TESTING ============ */
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  kicker(s, "Architecture");
  title(s, "100% coverage shaped the architecture");

  s.addText("Goal: 100% test coverage across the compiler pipeline — not just a metric, an architectural constraint.", {
    x: 0.6, y: 1.85, w: 12.1, h: 0.45, fontFace: SANS, fontSize: 14.5, color: SLATE, margin: 0,
  });

  const forced = ["Pass-based architecture", "Explicit IR boundaries", "Deterministic codegen", "Precise diagnostics", "Reference-based verification", "Explicit context, no global state"];
  const discouraged = ["Hidden global state", "Oversized “god” modules", "Implicit shape assumptions", "Broad mutation in passes", "Codegen driven by verify-only inputs"];

  // two columns
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 2.5, w: 6.0, h: 4.1, rectRadius: 0.08, fill: { color: "EAF5F3" }, line: { color: "C9E5E1", width: 1 }, shadow: sh({ opacity: 0.10 }) });
  s.addText("It forced", { x: 0.9, y: 2.7, w: 5.4, h: 0.4, fontFace: SANS, fontSize: 16, bold: true, color: TEAL2, margin: 0 });
  s.addText(forced.map((t) => ({ text: t, options: { bullet: { code: "2713" }, color: SLATE, breakLine: true, paraSpaceAfter: 7 } })), { x: 0.9, y: 3.15, w: 5.4, h: 3.3, fontFace: SANS, fontSize: 13.5, margin: 0, valign: "top" });

  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 6.85, y: 2.5, w: 5.85, h: 4.1, rectRadius: 0.08, fill: { color: "FBEEE9" }, line: { color: "EFD6CC", width: 1 }, shadow: sh({ opacity: 0.10 }) });
  s.addText("It discouraged", { x: 7.15, y: 2.7, w: 5.3, h: 0.4, fontFace: SANS, fontSize: 16, bold: true, color: "B5503A", margin: 0 });
  s.addText(discouraged.map((t) => ({ text: t, options: { bullet: { code: "2715" }, color: SLATE, breakLine: true, paraSpaceAfter: 7 } })), { x: 7.15, y: 3.15, w: 5.3, h: 3.3, fontFace: SANS, fontSize: 13.5, margin: 0, valign: "top" });

  footer(s, 6);
  s.addNotes("Coverage is not just a number — it forced the system to be modular and deterministic. Small functions, clear pass contracts, explicit context objects instead of globals, deterministic output for stable golden tests. The point of the talk is that the coverage target produced better compiler architecture, not merely a high percentage.");
}

/* ================== SLIDE 7 — VERIFICATION LOOP ================== */
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  kicker(s, "Verification");
  title(s, "The verification loop");

  const steps = [
    ["Generate", "Emit C with a testbench"],
    ["Compile & run", "Build and execute the generated C"],
    ["Reference", "Run ONNX Runtime / ONNX Reference on same inputs"],
    ["Compare", "ULP threshold after abs-epsilon; exact for non-float"],
    ["Record", "Coverage + expected failures tracked"],
  ];
  const bw = 2.32, gap = 0.13, x0 = 0.6, y0 = 2.3, bh = 2.35;
  steps.forEach((st, i) => {
    const x = x0 + i * (bw + gap);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: y0, w: bw, h: bh, rectRadius: 0.08, fill: { color: i % 2 ? CARD : "EAF5F3" }, line: { color: CARDLN, width: 1 }, shadow: sh({ opacity: 0.09 }) });
    s.addText(String(i + 1), { x: x + 0.18, y: y0 + 0.16, w: 0.5, h: 0.5, fontFace: SERIF, fontSize: 22, bold: true, color: TEAL, margin: 0 });
    s.addText(st[0], { x: x + 0.2, y: y0 + 0.72, w: bw - 0.4, h: 0.45, fontFace: SANS, fontSize: 14.5, bold: true, color: INK, margin: 0 });
    s.addText(st[1], { x: x + 0.2, y: y0 + 1.18, w: bw - 0.4, h: 1.05, fontFace: SANS, fontSize: 11.5, color: SLATE, margin: 0, valign: "top" });
    if (i < steps.length - 1) s.addText("›", { x: x + bw - 0.05, y: y0, w: gap + 0.1, h: bh, fontFace: SANS, fontSize: 20, bold: true, color: TEAL, align: "center", valign: "middle", margin: 0 });
  });

  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 5.2, w: 12.1, h: 1.5, rectRadius: 0.07, fill: { color: INK } });
  s.addText("Invariant", { x: 0.95, y: 5.35, w: 3, h: 0.4, fontFace: SANS, fontSize: 14, bold: true, color: AMBER, margin: 0 });
  s.addText("Verification-only inputs must never implicitly change generated code. If verify uses extra shape information, it is testing a different compiler path — compile/verify parity keeps the contract honest.", {
    x: 0.95, y: 5.75, w: 11.5, h: 0.85, fontFace: SANS, fontSize: 14, color: "DCE8F0", margin: 0, valign: "top",
  });
  footer(s, 7);
  s.addNotes("Explain why compile/verify parity matters. Bad pattern: verify reads representative input_*.pb files whose values or shapes make codegen succeed, while plain compile would fail or emit different code. If a model needs representative inputs to resolve dynamic shapes, it should fail clearly or require an explicit option that compile also has. Floating outputs are compared with ULP thresholds after an absolute epsilon; non-floating outputs must match exactly.");
}

/* ============ SLIDE 8 — LESSON 1: UNBOUNDED ============ */
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  kicker(s, "Lesson 1");
  badge(s, 0.6, 0.85, "1");
  s.addText("ONNX is effectively unbounded", { x: 1.4, y: 0.85, w: 11.3, h: 0.7, fontFace: SERIF, fontSize: 30, bold: true, color: INK, margin: 0, valign: "middle" });

  const items = [
    ["Dynamic dimensions", "Symbolic/unknown axes give a runtime extent, but no maximum"],
    ["Sequence length", "Containers have no natural capacity — one level above a dynamic dim"],
    ["String size", "String tensors carry no maximum character length"],
  ];
  const bw = 3.95, gap = 0.28, x0 = 0.6, y0 = 1.95, bh = 2.55;
  items.forEach((it, i) => {
    const x = x0 + i * (bw + gap);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: y0, w: bw, h: bh, rectRadius: 0.08, fill: { color: CARD }, line: { color: CARDLN, width: 1 }, shadow: sh({ opacity: 0.10 }) });
    s.addText(it[0], { x: x + 0.25, y: y0 + 0.25, w: bw - 0.5, h: 0.85, fontFace: SANS, fontSize: 16.5, bold: true, color: TEAL2, margin: 0, valign: "top" });
    s.addText(it[1], { x: x + 0.25, y: y0 + 1.1, w: bw - 0.5, h: 1.3, fontFace: SANS, fontSize: 13, color: SLATE, margin: 0, valign: "top" });
  });

  // emx response strip
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 4.75, w: 12.1, h: 1.95, rectRadius: 0.07, fill: { color: "EAF5F3" }, line: { color: "C9E5E1", width: 1 } });
  s.addText("emx-onnx-cgen makes the bound explicit", { x: 0.9, y: 4.88, w: 11.5, h: 0.4, fontFace: SANS, fontSize: 14.5, bold: true, color: TEAL2, margin: 0 });
  s.addText([
    { text: "Sequences → fixed-capacity arrays + count: ", options: { color: SLATE } },
    { text: "T name[EMX_SEQUENCE_MAX_LEN][...] + idx_t name__count", options: { fontFace: MONO, color: INK, breakLine: true } },
    { text: "Strings → fixed slots: ", options: { color: SLATE } },
    { text: "char[EMX_STRING_MAX_LEN]", options: { fontFace: MONO, color: INK } },
    { text: "  (256 default, #ifndef-overridable)", options: { color: MUTED, breakLine: true } },
    { text: "Ragged inputs require ", options: { color: SLATE } },
    { text: "--sequence-element-shape", options: { fontFace: MONO, color: INK } },
    { text: "  ·  otherwise: fail clearly", options: { color: MUTED } },
  ], { x: 0.9, y: 5.3, w: 11.5, h: 1.3, fontFace: SANS, fontSize: 12.5, margin: 0, valign: "top", lineSpacingMultiple: 1.15 });

  footer(s, 8);
  s.addNotes("This is the central AOT lesson. ONNX is intentionally flexible as an interchange format; embedded C needs concrete bounds. General static memory planning is impossible without additional assumptions. A tensor dimension asks 'how large is this axis?'; a sequence asks 'how many tensor objects, and what shape each?' — comfortable for runtimes, hard for static C. emx-onnx-cgen bounds these explicitly via fixed-capacity arrays, fixed string slots, and override macros, and fails clearly when bounds are missing.");
}

/* ============ SLIDE 9 — LESSON 2: TYPE INFERENCE ============ */
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  kicker(s, "Lesson 2");
  badge(s, 0.6, 0.85, "2");
  s.addText("Type & shape inference is not enough", { x: 1.4, y: 0.85, w: 11.3, h: 0.7, fontFace: SERIF, fontSize: 30, bold: true, color: INK, margin: 0, valign: "middle" });

  card(s, 0.6, 1.95, 6.0, 2.55, "Where it falls short", [
    "Inference is incomplete for some operators",
    "Output shape can depend on values, attributes or subgraphs",
    "Container element shapes may simply be missing",
    "Sequence types complicate static analysis & ABI design",
  ]);
  card(s, 0.6, 4.7, 6.0, 2.0, "Compiler response", [
    "Normalize attributes & types on import",
    "Add compiler-side inference where ONNX stops",
    "Missing information becomes an explicit error",
  ], { fill: "EAF5F3", headColor: TEAL2 });

  // duplication problem + vision
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 6.85, y: 1.95, w: 5.85, h: 2.2, rectRadius: 0.08, fill: { color: "FBEEE9" }, line: { color: "EFD6CC", width: 1 }, shadow: sh({ opacity: 0.10 }) });
  s.addText("The duplication problem", { x: 7.15, y: 2.1, w: 5.3, h: 0.4, fontFace: SANS, fontSize: 15, bold: true, color: "B5503A", margin: 0 });
  s.addText("No reliable standalone library infers shapes/types for dynamic models in a way static code generators can reuse — so importers, MLIR frontends, codegens, verifiers and runtimes each reimplement it, and may disagree.", {
    x: 7.15, y: 2.55, w: 5.4, h: 1.55, fontFace: SANS, fontSize: 12.5, color: SLATE, margin: 0, valign: "top",
  });

  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 6.85, y: 4.35, w: 5.85, h: 2.35, rectRadius: 0.08, fill: { color: INK }, shadow: sh({ opacity: 0.12 }) });
  s.addText([{ text: "VISION  ", options: { color: AMBER, bold: true } }, { text: "a shared, extensible inference layer", options: { color: LIGHT, bold: true } }], { x: 7.15, y: 4.5, w: 5.3, h: 0.4, fontFace: SANS, fontSize: 14, margin: 0 });
  s.addText([
    { text: "Preserve symbolic dims, constraints & unresolved facts", options: { bullet: { code: "2022" }, color: "CFE0EC", breakLine: true, paraSpaceAfter: 5 } },
    { text: "Registry for external domains (ORT contrib ops)", options: { bullet: { code: "2022" }, color: "CFE0EC", breakLine: true, paraSpaceAfter: 5 } },
    { text: "Persist results back into ONNX (value_info / metadata)", options: { bullet: { code: "2022" }, color: "CFE0EC", breakLine: true, paraSpaceAfter: 5 } },
    { text: "One DSL → C++ & Python, numeric + symbolic (SymPy)", options: { bullet: { code: "2022" }, color: "CFE0EC" } },
  ], { x: 7.15, y: 4.95, w: 5.4, h: 1.7, fontFace: SANS, fontSize: 12, margin: 0, valign: "top" });

  footer(s, 9);
  s.addNotes("Frame this as a mismatch, not only a flaw: a flexible model format versus a static compiler input. Inference should be separable from code generation and from a fixed operator universe. A shared, extensible inference library that preserves symbolic/dynamic dimensions and constraints — and writes results back into the model — would stop every backend reimplementing it. We are exploring a DSL where a rule is written once and runs from both C++ and Python, numerically for concrete dims and symbolically (e.g. SymPy) for symbolic dims, and could ship inside ONNX models for custom operators. Present as vision/research, not finished work.");
}

/* ============ SLIDE 10 — LESSON 3: ACCURACY ============ */
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  kicker(s, "Lesson 3");
  badge(s, 0.6, 0.85, "3");
  s.addText("Numerical accuracy is underspecified", { x: 1.4, y: 0.85, w: 11.3, h: 0.7, fontFace: SERIF, fontSize: 30, bold: true, color: INK, margin: 0, valign: "middle" });

  card(s, 0.6, 1.95, 6.0, 2.6, "The ambiguity", [
    "Official tests give examples, not full accuracy contracts",
    "Reference impls differ on edge cases & FP behaviour",
    "node tests carry no per-operator tolerance (data.json is for real models)",
    "Accumulation, rounding, approximations left to the backend",
  ]);

  // tolerance comparison mini-table
  s.addText("Everyone picks a different policy", { x: 6.85, y: 1.95, w: 5.85, h: 0.4, fontFace: SANS, fontSize: 15, bold: true, color: TEAL2, margin: 0 });
  const rows = [
    ["ONNX backend", "rtol 1e-3, atol 1e-7"],
    ["ONNX Runtime", "per-sample 1e-3, config.txt"],
    ["PyTorch", "dtype-dependent defaults"],
    ["NumPy / JAX", "atol + rtol·|expected|"],
    ["Polygraphy", "per-output; FP32 too strict for FP16"],
  ];
  let ty = 2.45;
  rows.forEach((r, i) => {
    s.addShape(pres.shapes.RECTANGLE, { x: 6.85, y: ty, w: 5.85, h: 0.42, fill: { color: i % 2 ? "FFFFFF" : CARD }, line: { color: CARDLN, width: 0.75 } });
    s.addText(r[0], { x: 7.0, y: ty, w: 2.3, h: 0.42, fontFace: SANS, fontSize: 12, bold: true, color: INK, margin: 0, valign: "middle" });
    s.addText(r[1], { x: 9.3, y: ty, w: 3.3, h: 0.42, fontFace: MONO, fontSize: 11, color: SLATE, margin: 0, valign: "middle" });
    ty += 0.42;
  });

  // emx approach
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 4.75, w: 6.0, h: 1.95, rectRadius: 0.08, fill: { color: "EAF5F3" }, line: { color: "C9E5E1", width: 1 } });
  s.addText("emx-onnx-cgen: ULP-based, dtype-aware", { x: 0.85, y: 4.88, w: 5.5, h: 0.4, fontFace: SANS, fontSize: 14, bold: true, color: TEAL2, margin: 0 });
  s.addText([
    { text: "Ignore abs diffs up to ", options: { color: SLATE } },
    { text: "atol_eps · eps(dtype)", options: { fontFace: MONO, color: INK, breakLine: true } },
    { text: "Then measure ", options: { color: SLATE } },
    { text: "ULP distance", options: { fontFace: MONO, color: INK } },
    { text: "; non-float must match exactly", options: { color: SLATE, breakLine: true } },
    { text: "CLI: ", options: { color: SLATE } },
    { text: "--atol-eps  --max-ulp  --fp32-accumulation-strategy fp64", options: { fontFace: MONO, color: INK } },
  ], { x: 0.85, y: 5.32, w: 5.55, h: 1.3, fontFace: SANS, fontSize: 12, margin: 0, valign: "top", lineSpacingMultiple: 1.15 });

  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 6.85, y: 4.75, w: 5.85, h: 1.95, rectRadius: 0.08, fill: { color: INK } });
  s.addText("Why ULP, not fixed rtol/atol", { x: 7.1, y: 4.88, w: 5.4, h: 0.4, fontFace: SANS, fontSize: 14, bold: true, color: AMBER, margin: 0 });
  s.addText("float16, float32, float64 have very different spacing between representable values. ULP asks the real question: how many representable values apart are the results? Some reference outputs are high-precision, so a few official tests need fp64 accumulation.", {
    x: 7.1, y: 5.32, w: 5.4, h: 1.3, fontFace: SANS, fontSize: 12, color: "CFE0EC", margin: 0, valign: "top",
  });

  footer(s, 10);
  s.addNotes("Highly relevant to backend authors. Validation is an engineering policy unless the standard defines the contract. A single fixed rtol/atol is not equally meaningful across dtypes — float16/32/64 differ enormously in spacing. emx uses an absolute-epsilon gate then ULP distance, exact for non-float. Some ONNX reference outputs are effectively high precision, which under strict ULP forced 64-bit accumulation on selected tests, exposed as --fp32-accumulation-strategy fp64.");
}

/* ================== SLIDE 11 — CURRENT STATUS ================== */
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  kicker(s, "Status");
  title(s, "Where it stands today");

  const stats = [
    ["Opset 26", "ONNX 1.21.0 target"],
    ["ORT 1.26", "broad Microsoft operator support"],
    ["#2", "after ONNX Reference on the\nBackend Scoreboard (stable)"],
  ];
  const bw = 3.95, gap = 0.28, x0 = 0.6, y0 = 2.1, bh = 1.85;
  stats.forEach((st, i) => {
    const x = x0 + i * (bw + gap);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: y0, w: bw, h: bh, rectRadius: 0.08, fill: { color: INK }, shadow: sh({ opacity: 0.14 }) });
    s.addText(st[0], { x: x + 0.25, y: y0 + 0.2, w: bw - 0.5, h: 0.85, fontFace: SERIF, fontSize: 38, bold: true, color: AMBER, margin: 0, valign: "middle" });
    s.addText(st[1], { x: x + 0.25, y: y0 + 1.0, w: bw - 0.5, h: 0.75, fontFace: SANS, fontSize: 13, color: "CFE0EC", margin: 0, valign: "top" });
  });

  card(s, 0.6, 4.25, 6.0, 2.45, "Coverage is reported, reproducible, visible", [
    "SUPPORT_OPS.md — operator support status",
    "ONNX_SUPPORT.md — official backend model coverage",
    "ONNX_SUPPORT.md — ORT artifact corpus coverage",
    "Unsupported cases documented as expected errors",
  ], { headColor: TEAL2 });

  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 6.85, y: 4.25, w: 5.85, h: 2.45, rectRadius: 0.08, fill: { color: CARD }, line: { color: CARDLN, width: 1 }, shadow: sh({ opacity: 0.10 }) });
  s.addText("emx-ort-test-artifacts", { x: 7.15, y: 4.4, w: 5.3, h: 0.4, fontFace: SANS, fontSize: 15, bold: true, color: INK, margin: 0 });
  s.addText("Exports ORT tests into backend-test-like artifacts (model.onnx + test_data_set_*), reusable outside the ORT C++ harness — a broader, runtime-oriented reality check beyond the compact official node tests.", {
    x: 7.15, y: 4.85, w: 5.4, h: 1.0, fontFace: SANS, fontSize: 12, color: SLATE, margin: 0, valign: "top",
  });
  s.addText("ORT test case → ONNX model + test_data_set_* → backend-runnable", {
    x: 7.15, y: 5.95, w: 5.4, h: 0.6, fontFace: MONO, fontSize: 10.5, color: TEAL2, margin: 0, valign: "top",
  });

  footer(s, 11);
  s.addNotes("Phrase coverage as report-based and corpus-based — don't make exact percentages the point. The scoreboard placement (directly after ONNX Reference in the stable build) is credibility, but it measures backend unit-test execution, not auditability or deterministic AOT constraints. The real message: coverage became systematic, reproducible and visible. emx-ort-test-artifacts extends the same artifact idea to ORT-derived tests; the idea was raised in ORT forums with near-zero response — mention as an underused community opportunity, not a complaint.");
}

/* ============ SLIDE 12 — TAKEAWAYS FOR THE COMMUNITY ============ */
{
  const s = pres.addSlide();
  s.background = { color: INK };
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 0.55, w: 0.22, h: 0.22, rectRadius: 0.05, fill: { color: TEAL } });
  s.addText("FOR THE ONNX COMMUNITY", { x: 0.92, y: 0.5, w: 9, h: 0.33, fontFace: SANS, fontSize: 12, bold: true, color: TEAL, charSpacing: 2, margin: 0, valign: "middle" });
  s.addText("Takeaways", { x: 0.6, y: 0.95, w: 12, h: 0.9, fontFace: SERIF, fontSize: 32, bold: true, color: LIGHT, margin: 0, valign: "middle" });

  const cards = [
    ["Boundedness", "An AOT-friendly ONNX profile should state what must be bounded or static for deterministic codegen."],
    ["Shared inference", "Reliable, reusable shape/type inference for dynamic models — extensible to external domains like ORT contrib."],
    ["Persisted facts", "Inferred shapes/types should be storable in the model; a small DSL could drive C++ & Python, numeric + symbolic."],
    ["Accuracy contracts", "Numerical accuracy requirements should be explicit in specs and official tests."],
    ["Code quality", "Determinism, readability and auditability are backend features — not cosmetics."],
    ["Portable test artifacts", "ORT-exported, backend-test-like artifacts let any backend share a broader corpus."],
  ];
  const bw = 3.95, gap = 0.27, bh = 1.55, x0 = 0.6, y0 = 2.0;
  cards.forEach((c, i) => {
    const col = i % 3, row = Math.floor(i / 3);
    const x = x0 + col * (bw + gap), y = y0 + row * (bh + 0.3);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: bw, h: bh, rectRadius: 0.08, fill: { color: "13355A" }, line: { color: "1E4A78", width: 1 } });
    s.addText(c[0], { x: x + 0.25, y: y + 0.18, w: bw - 0.5, h: 0.4, fontFace: SANS, fontSize: 15, bold: true, color: AMBER, margin: 0 });
    s.addText(c[1], { x: x + 0.25, y: y + 0.6, w: bw - 0.5, h: 0.85, fontFace: SANS, fontSize: 11.5, color: "CFE0EC", margin: 0, valign: "top" });
  });

  s.addText("ONNX is excellent interchange — deterministic AOT compilation just needs a few constraints made explicit.", {
    x: 0.6, y: 6.55, w: 12.1, h: 0.5, fontFace: SANS, fontSize: 14, italic: true, color: "9FC0D6", margin: 0,
  });
  s.addNotes("End constructively. AOT-friendly ONNX usage needs explicit bounds; compiler-oriented infrastructure should include reliable, reusable, extensible shape/type inference whose results can be persisted in the model; numerical accuracy requirements should be more explicit; generated-code quality, determinism and auditability are backend features. The closing question for the room: what would an AOT-friendly ONNX profile need to specify so embedded backends can compile deterministically without hidden assumptions?");
}

/* ================== SLIDE 13 — DISCUSSION ================== */
{
  const s = pres.addSlide();
  s.background = { color: DEEP };
  s.addText("Discussion", { x: 0.85, y: 0.75, w: 11.6, h: 1.0, fontFace: SERIF, fontSize: 44, bold: true, color: LIGHT, margin: 0 });
  s.addText("What would an AOT-friendly ONNX profile need to specify?", {
    x: 0.87, y: 1.75, w: 11.6, h: 0.6, fontFace: SANS, fontSize: 18, color: AMBER, margin: 0,
  });

  const qs = [
    "Should ONNX have a stronger shared shape/type inference artifact for dynamic models?",
    "How should external operator domains register inference rules?",
    "Could custom operators carry portable shape/type rules inside ONNX?",
    "Which operators most need clearer numerical contracts?",
    "How should scoreboards represent deterministic codegen & generated-code quality?",
  ];
  s.addText(qs.map((q) => ({ text: q, options: { bullet: { code: "2192", indent: 22 }, color: "DCE8F0", breakLine: true, paraSpaceAfter: 13 } })), {
    x: 0.9, y: 2.65, w: 11.5, h: 3.4, fontFace: SANS, fontSize: 16.5, margin: 0, valign: "top",
  });

  s.addShape(pres.shapes.LINE, { x: 0.9, y: 6.25, w: 3.5, h: 0, line: { color: TEAL, width: 2 } });
  s.addText("Thank you  ·  emx-onnx-cgen is open source", { x: 0.9, y: 6.45, w: 11, h: 0.5, fontFace: SANS, fontSize: 15, bold: true, color: LIGHT, margin: 0 });
  s.addNotes("Open the floor. Lead with the profile question, then let the room pick threads: shared inference artifact, domain registration, portable custom-op rules, numerical contracts, and how scoreboards could reflect deterministic, auditable code generation rather than only unit-test execution.");
}

pres.writeFile({ fileName: "onnx-community-day-aot-onnx-to-c-v2.pptx" }).then((f) => console.log("WROTE", f));
