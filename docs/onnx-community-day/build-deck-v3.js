const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5
pres.author = "emmtrix Technologies";
pres.title = "Lessons learned from building an AOT ONNX-to-C compiler (master deck)";

const W = 13.333, H = 7.5;

// ---- Palette ----
const INK   = "0E2A47";
const DEEP  = "0A1F38";
const TEAL  = "16A39A";
const TEAL2 = "0E7C77";
const AMBER = "E8A13A";
const SLATE = "33475B";
const MUTED = "6B7C8F";
const LIGHT = "FFFFFF";
const CARD  = "F2F6F8";
const CARDLN= "DBE6EA";
const CODEBG= "0C2138";
const CODEFG= "D7E4EC";
const GREEN = "EAF5F3", GREENLN = "C9E5E1", GREENTX = "0E7C77";
const RUST  = "FBEEE9", RUSTLN  = "EFD6CC", RUSTTX  = "B5503A";
const SAND  = "FBF3E7", SANDLN  = "EAD9BC", SANDTX  = "B5781F";

const SERIF = "Cambria";
const SANS  = "Calibri";
const MONO  = "Consolas";

let pageNo = 0;
const sh = (o = {}) => Object.assign({ type: "outer", color: "0A1F38", blur: 9, offset: 3, angle: 90, opacity: 0.14 }, o);

function footer(slide) {
  pageNo += 1;
  slide.addText("emx-onnx-cgen  ·  ONNX Community Day  ·  master deck", {
    x: 0.6, y: H - 0.42, w: 8, h: 0.3, fontFace: SANS, fontSize: 9, color: MUTED, align: "left", margin: 0 });
  slide.addText(String(pageNo), {
    x: W - 1.1, y: H - 0.42, w: 0.5, h: 0.3, fontFace: SANS, fontSize: 9, color: MUTED, align: "right", margin: 0 });
}

function kicker(slide, text) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 0.52, w: 0.22, h: 0.22, rectRadius: 0.05, fill: { color: TEAL } });
  slide.addText(text.toUpperCase(), { x: 0.92, y: 0.47, w: 11.5, h: 0.33, fontFace: SANS, fontSize: 12, bold: true, color: TEAL2, charSpacing: 2, margin: 0, valign: "middle" });
}

function title(slide, text, y = 0.92) {
  slide.addText(text, { x: 0.6, y, w: 12.1, h: 0.9, fontFace: SERIF, fontSize: 28, bold: true, color: INK, margin: 0, valign: "middle" });
}

function codeBox(slide, code, opt = {}) {
  const o = Object.assign({ x: 0.6, y: 2.2, w: 6.0, h: 2.4, fontSize: 12.5 }, opt);
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: o.x, y: o.y, w: o.w, h: o.h, rectRadius: 0.06, fill: { color: CODEBG }, shadow: sh() });
  if (o.caption) slide.addText(o.caption, { x: o.x + 0.2, y: o.y + 0.08, w: o.w - 0.4, h: 0.3, fontFace: SANS, fontSize: 10.5, bold: true, color: "7FA8C0", margin: 0 });
  slide.addText(code, { x: o.x + 0.2, y: o.y + (o.caption ? 0.42 : 0.12), w: o.w - 0.4, h: o.h - (o.caption ? 0.54 : 0.24), fontFace: MONO, fontSize: o.fontSize, color: CODEFG, align: "left", valign: "top", margin: 0, lineSpacingMultiple: 1.05 });
}

function card(slide, x, y, w, h, header, bullets, opt = {}) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.07, fill: { color: opt.fill || CARD }, line: { color: opt.line || CARDLN, width: 1 }, shadow: sh({ opacity: 0.10 }) });
  let cy = y + 0.2;
  if (header) {
    slide.addText(header, { x: x + 0.26, y: cy, w: w - 0.48, h: 0.4, fontFace: SANS, fontSize: opt.headSize || 15, bold: true, color: opt.headColor || TEAL2, margin: 0, valign: "middle" });
    cy += 0.48;
  }
  if (bullets && bullets.length) {
    const code = opt.bulletCode || "2022";
    const runs = bullets.map((b) => ({ text: b, options: { bullet: { code, indent: 14 }, color: opt.textColor || SLATE, breakLine: true, paraSpaceAfter: opt.gap != null ? opt.gap : 6 } }));
    slide.addText(runs, { x: x + 0.26, y: cy, w: w - 0.48, h: h - (cy - y) - 0.18, fontFace: SANS, fontSize: opt.fontSize || 12.5, align: "left", valign: "top", margin: 0 });
  }
}

function badge(slide, x, y, label, d = 0.62) {
  slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: TEAL }, shadow: sh({ opacity: 0.18 }) });
  slide.addText(label, { x, y, w: d, h: d, fontFace: SERIF, fontSize: d > 0.55 ? 20 : 15, bold: true, color: LIGHT, align: "center", valign: "middle", margin: 0 });
}

function lessonHead(slide, num, text) {
  kicker(slide, "Lesson " + num);
  badge(slide, 0.6, 0.82, String(num));
  slide.addText(text, { x: 1.42, y: 0.82, w: 11.3, h: 0.66, fontFace: SERIF, fontSize: 27, bold: true, color: INK, margin: 0, valign: "middle" });
}

function sectionDivider(num, secTitle, subtitle, items) {
  const s = pres.addSlide();
  s.background = { color: INK };
  s.addText(String(num).padStart(2, "0"), { x: 0.85, y: 1.0, w: 3, h: 1.6, fontFace: SERIF, fontSize: 96, bold: true, color: "1C436B", margin: 0 });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.9, y: 2.95, w: 0.7, h: 0.12, rectRadius: 0.06, fill: { color: TEAL } });
  s.addText(secTitle, { x: 0.85, y: 3.15, w: 11.5, h: 1.0, fontFace: SERIF, fontSize: 40, bold: true, color: LIGHT, margin: 0 });
  if (subtitle) s.addText(subtitle, { x: 0.87, y: 4.2, w: 11, h: 0.6, fontFace: SANS, fontSize: 17, color: "9FC0D6", margin: 0 });
  if (items && items.length) {
    s.addText(items.map((t) => ({ text: t, options: { bullet: { code: "2192", indent: 16 }, color: "CFE0EC", breakLine: true, paraSpaceAfter: 7 } })), {
      x: 0.9, y: 5.0, w: 11.4, h: 1.9, fontFace: SANS, fontSize: 14, margin: 0, valign: "top" });
  }
  return s;
}

/* =================================================================== */
/* SLIDE 1 — TITLE                                                     */
/* =================================================================== */
{
  const s = pres.addSlide();
  s.background = { color: INK };
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.85, y: 0.95, w: 0.28, h: 0.28, rectRadius: 0.06, fill: { color: TEAL } });
  s.addText("LESSONS LEARNED  ·  MASTER DECK", { x: 1.25, y: 0.9, w: 10, h: 0.4, fontFace: SANS, fontSize: 14, bold: true, color: TEAL, charSpacing: 3, margin: 0, valign: "middle" });
  s.addText("Building an AOT\nONNX-to-C Compiler", { x: 0.85, y: 1.6, w: 11.6, h: 2.1, fontFace: SERIF, fontSize: 50, bold: true, color: LIGHT, margin: 0, lineSpacingMultiple: 1.0 });
  s.addText("emx-onnx-cgen — deterministic, portable C for embedded and resource-constrained systems", {
    x: 0.87, y: 3.85, w: 11.2, h: 0.6, fontFace: SANS, fontSize: 18, color: "B9CEDD", margin: 0 });

  const fy = 4.85;
  const chips = [["ONNX model", DEEP], ["emx-onnx-cgen", TEAL2], ["clean C", DEEP], ["embedded target", DEEP]];
  let cx = 0.87;
  chips.forEach((c, i) => {
    const wdt = 0.35 + c[0].length * 0.115;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: cx, y: fy, w: wdt, h: 0.5, rectRadius: 0.1, fill: { color: c[1] }, line: { color: TEAL, width: i === 1 ? 1.5 : 0.75 } });
    s.addText(c[0], { x: cx, y: fy, w: wdt, h: 0.5, fontFace: MONO, fontSize: 12.5, color: i === 1 ? LIGHT : "C7D8E5", align: "center", valign: "middle", margin: 0 });
    cx += wdt;
    if (i < chips.length - 1) { s.addText("→", { x: cx, y: fy, w: 0.45, h: 0.5, fontFace: SANS, fontSize: 18, color: TEAL, align: "center", valign: "middle", margin: 0 }); cx += 0.45; }
  });
  s.addText("This is an extended master deck — pick the slides relevant to your session.", { x: 0.87, y: 5.7, w: 11, h: 0.4, fontFace: SANS, fontSize: 13, italic: true, color: "8AA3B5", margin: 0 });
  s.addText("ONNX Community Day  ·  emmtrix Technologies", { x: 0.87, y: 6.55, w: 11, h: 0.4, fontFace: SANS, fontSize: 13, color: "8AA3B5", margin: 0 });
  s.addNotes("Master deck: a superset of slides covering every theme in depth. Use it as a pool to assemble a 20-minute talk or a longer technical session. emx-onnx-cgen is an open-source ahead-of-time compiler that turns ONNX models into portable, deterministic C for deeply embedded, safety-critical and resource-constrained systems.");
}

/* =================================================================== */
/* SLIDE 2 — AGENDA / MAP                                              */
/* =================================================================== */
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  kicker(s, "How this deck is organised");
  title(s, "Six themes to pick from");
  const secs = [
    ["01", "Origins & motivation", "onnx2c, the VLA idea, why a new compiler"],
    ["02", "What emx-onnx-cgen does", "Output, API, C as handoff IR"],
    ["03", "Architecture & verification", "100% coverage, the verify loop, ORT artifacts"],
    ["04", "ONNX lessons learned", "Boundedness, inference, accuracy, dtypes"],
    ["05", "Status & community", "Coverage reports, scoreboard, takeaways"],
    ["06", "Discussion", "Open questions for an AOT-friendly ONNX"],
  ];
  const bw = 3.95, bh = 1.95, gx = 0.27, gy = 0.32, x0 = 0.6, y0 = 2.0;
  secs.forEach((c, i) => {
    const col = i % 3, row = Math.floor(i / 3);
    const x = x0 + col * (bw + gx), y = y0 + row * (bh + gy);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: bw, h: bh, rectRadius: 0.08, fill: { color: CARD }, line: { color: CARDLN, width: 1 }, shadow: sh({ opacity: 0.10 }) });
    s.addText(c[0], { x: x + 0.26, y: y + 0.22, w: 1.4, h: 0.7, fontFace: SERIF, fontSize: 32, bold: true, color: TEAL, margin: 0 });
    s.addText(c[1], { x: x + 0.26, y: y + 0.95, w: bw - 0.5, h: 0.45, fontFace: SANS, fontSize: 15.5, bold: true, color: INK, margin: 0 });
    s.addText(c[2], { x: x + 0.26, y: y + 1.38, w: bw - 0.5, h: 0.5, fontFace: SANS, fontSize: 12, color: SLATE, margin: 0, valign: "top" });
  });
  footer(s);
  s.addNotes("Map of the deck. Each section starts with a dark divider slide. Sections are independent enough to cherry-pick.");
}

/* =================================================================== */
/* SECTION 01 — ORIGINS                                                */
/* =================================================================== */
sectionDivider(1, "Origins & motivation", "Why we built another ONNX-to-C compiler", [
  "onnx2c proved the value — and showed the gaps",
  "Dynamic dimensions were the central pain point",
  "The VLA-parameter idea triggered a new implementation",
]);

/* SLIDE — Why we started */
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  kicker(s, "History  ·  motivation");
  title(s, "Why we started");
  card(s, 0.6, 1.95, 5.85, 4.5, "onnx2c — what worked", [
    "Proved that compiling ONNX graphs to C is genuinely useful",
    "Gave a concrete reference for what generated C looks like",
    "Helped surface real requirements for embedded deployment",
  ], { fill: GREEN, line: GREENLN, headColor: GREENTX });
  card(s, 6.65, 1.95, 6.05, 4.5, "onnx2c — what was missing for us", [
    "Generated-code quality was hard for our target use case",
    "Not enough control over the structure of the emitted C",
    "Dynamic-dimension support was a major pain point",
    "Embedded C must be readable, deterministic, reviewable",
  ], { fill: SAND, line: SANDLN, headColor: SANDTX });
  s.addText("For embedded systems, generated code is not an implementation detail — it is an artifact to review, analyze, certify, debug and further optimize.", {
    x: 0.6, y: 6.65, w: 12.1, h: 0.5, fontFace: SANS, fontSize: 13, italic: true, color: MUTED, margin: 0 });
  footer(s);
  s.addNotes("Start with practical history. This was not initially a 'build a complete ONNX compiler' project — the first driver was control over generated C. onnx2c demonstrated the value but generated-code quality and control were insufficient, and dynamic dimensions were the central irritation.");
}

/* SLIDE — The search for alternatives */
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  kicker(s, "Landscape");
  title(s, "We looked for existing solutions first");
  const rows = [
    ["onnx2c", "Closest match — direct ONNX-to-C", "Quality & control insufficient; weak on dynamic dims", SANDTX],
    ["Apache TVM", "Mature ML compiler stack", "No straightforward standalone ONNX-to-generic-C path", RUSTTX],
    ["IREE / MLIR", "Strong staged-lowering compiler infra", "Key IR stays inside the stack; not auditable C handoff", RUSTTX],
  ];
  let y = 2.05;
  rows.forEach((r) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y, w: 12.1, h: 1.25, rectRadius: 0.07, fill: { color: CARD }, line: { color: CARDLN, width: 1 }, shadow: sh({ opacity: 0.08 }) });
    s.addText(r[0], { x: 0.85, y: y + 0.1, w: 2.7, h: 1.05, fontFace: MONO, fontSize: 16, bold: true, color: INK, margin: 0, valign: "middle" });
    s.addText([{ text: "Strength  ", options: { bold: true, color: GREENTX } }, { text: r[1], options: { color: SLATE } }], { x: 3.6, y: y + 0.18, w: 4.4, h: 0.9, fontFace: SANS, fontSize: 12.5, margin: 0, valign: "middle" });
    s.addText([{ text: "Gap for us  ", options: { bold: true, color: r[3] } }, { text: r[2], options: { color: SLATE } }], { x: 8.1, y: y + 0.18, w: 4.4, h: 0.9, fontFace: SANS, fontSize: 12.5, margin: 0, valign: "middle" });
    y += 1.42;
  });
  s.addText("We wanted a standalone, deterministic, auditable ONNX-to-generic-C flow — and no existing tool matched it at the time.", {
    x: 0.6, y: 6.5, w: 12.1, h: 0.5, fontFace: SANS, fontSize: 13, italic: true, color: MUTED, margin: 0 });
  footer(s);
  s.addNotes("Apart from onnx2c there were few obvious solutions matching standalone, deterministic, auditable C output. TVM and IREE are relevant compiler stacks, but in our evaluation they did not provide the straightforward ONNX-to-generic-C flow we needed. MLIR is the right conceptual comparison for staged lowering, but it keeps the key IR inside the compiler rather than exposing auditable C.");
}

/* SLIDE — The VLA trigger */
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  kicker(s, "The trigger");
  title(s, "The dynamic-dimension idea: C99 VLA parameters");
  card(s, 0.6, 1.95, 5.5, 2.4, "The pain point → an idea", [
    "onnx2c's weak spot was dynamic dimensions",
    "Idea: represent some dynamic dims as C99 VLA parameters",
    "Rank stays static; extents become runtime arguments",
  ], { gap: 7 });
  card(s, 0.6, 4.5, 5.5, 2.2, "Why it appealed", [
    "C can express runtime tensor extents in the signature",
    "The generated ABI stays explicit",
    "Loop bounds can use explicit extent parameters",
  ], { fill: GREEN, line: GREENLN, headColor: GREENTX, gap: 7 });
  codeBox(s,
`void model(int N, int C,
           const float x[restrict N][C],
           float       y[restrict N][C]);`,
    { x: 6.4, y: 1.95, w: 6.3, h: 1.6, fontSize: 14, caption: "VLA parameters — known rank, runtime extents" });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 6.4, y: 3.75, w: 6.3, h: 2.95, rectRadius: 0.07, fill: { color: SAND }, line: { color: SANDLN, width: 1 } });
  s.addText("Representation is not semantics — what VLAs do NOT solve", { x: 6.65, y: 3.9, w: 5.8, h: 0.6, fontFace: SANS, fontSize: 13.5, bold: true, color: SANDTX, margin: 0 });
  s.addText([
    { text: "Static memory bounds for temporaries and buffers", options: { bullet: { code: "2022" }, breakLine: true, paraSpaceAfter: 6 } },
    { text: "Dynamic rank", options: { bullet: { code: "2022" }, breakLine: true, paraSpaceAfter: 6 } },
    { text: "Unbounded sequence length and string size", options: { bullet: { code: "2022" }, breakLine: true, paraSpaceAfter: 6 } },
    { text: "Targets that lack VLA support", options: { bullet: { code: "2022" } } },
  ], { x: 6.65, y: 4.5, w: 5.85, h: 2.05, fontFace: SANS, fontSize: 12.5, color: SLATE, margin: 0, valign: "top" });
  footer(s);
  s.addNotes("VLA parameters are a representation technique for known rank with runtime extents. They keep the ABI explicit. But representation is not semantics: they do not bound memory, do not solve dynamic rank, and do not solve unbounded sequences or strings. This experiment justified building a new implementation.");
}

/* SLIDE — Flat vs N-D arrays */
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  kicker(s, "Deep dive");
  title(s, "Why N-dimensional C types matter");
  codeBox(s,
`void add_flat(int N, int C,
   const float *restrict a,
   const float *restrict b,
   float *restrict out) {
  for (int n=0; n<N; ++n)
    for (int c=0; c<C; ++c) {
      int idx = n*C + c;
      out[idx] = a[idx] + b[idx];
    }
}`,
    { x: 0.6, y: 2.0, w: 5.9, h: 3.5, fontSize: 12.5, caption: "Flat 1-D buffers — structure is only convention" });
  codeBox(s,
`void add_vla(int N, int C,
   const float a[restrict N][C],
   const float b[restrict N][C],
   float out[restrict N][C]) {
  for (int n=0; n<N; ++n)
    for (int c=0; c<C; ++c) {
      out[n][c] = a[n][c] + b[n][c];
    }
}`,
    { x: 6.8, y: 2.0, w: 5.9, h: 3.5, fontSize: 12.5, caption: "N-D VLA types — rank & extent live in the type" });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 5.65, w: 12.1, h: 1.15, rectRadius: 0.07, fill: { color: INK } });
  s.addText([
    { text: "The subtle point:  ", options: { bold: true, color: AMBER } },
    { text: "accessing beyond a declared array object is undefined behaviour — even when the next tensor row is contiguous in memory. Keeping tensor structure in the type improves readability, reviewability and downstream analysis.", options: { color: "DCE8F0" } },
  ], { x: 0.95, y: 5.65, w: 11.4, h: 1.15, fontFace: SANS, fontSize: 13.5, margin: 0, valign: "middle" });
  footer(s);
  s.addNotes("Both flat and N-D arrays address the same contiguous memory, but in C the declared array type matters. With float x[N][C] the rank and inner extent are visible in the type; with float *x the structure exists only as an offset-calculation convention. Accessing beyond a declared array object is UB even if the next logical row is contiguous. Typed arrays help readability, review and downstream analysis like vectorization.");
}

/* SLIDE — From experiment to coverage */
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  kicker(s, "The turning point");
  title(s, "From experiment to coverage project");
  const steps = [
    ["1", "Proof of concept", "The VLA-based C representation worked on early models"],
    ["2", "Goal expanded", "From prototype to broad ONNX operator & model coverage"],
    ["3", "Became a compiler", "Pass-based AOT pipeline with verification & coverage reports"],
  ];
  const bw = 3.95, gap = 0.28, x0 = 0.6, y0 = 2.2, bh = 2.95;
  steps.forEach((st, i) => {
    const x = x0 + i * (bw + gap);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: y0, w: bw, h: bh, rectRadius: 0.08, fill: { color: CARD }, line: { color: CARDLN, width: 1 }, shadow: sh({ opacity: 0.10 }) });
    badge(s, x + 0.3, y0 + 0.3, st[0]);
    s.addText(st[1], { x: x + 0.3, y: y0 + 1.1, w: bw - 0.6, h: 0.5, fontFace: SANS, fontSize: 17, bold: true, color: INK, margin: 0 });
    s.addText(st[2], { x: x + 0.3, y: y0 + 1.62, w: bw - 0.6, h: 1.1, fontFace: SANS, fontSize: 13, color: SLATE, margin: 0, valign: "top" });
    if (i < steps.length - 1) s.addText("→", { x: x + bw - 0.02, y: y0, w: gap + 0.04, h: bh, fontFace: SANS, fontSize: 22, color: TEAL, align: "center", valign: "middle", margin: 0 });
  });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 5.6, w: 12.1, h: 1.1, rectRadius: 0.07, fill: { color: INK } });
  s.addText("A high-coverage ONNX compiler cannot be a pile of special cases — it needs registries, structured lowering, stable IR objects and a verification strategy.", {
    x: 0.95, y: 5.6, w: 11.4, h: 1.1, fontFace: SANS, fontSize: 14.5, italic: true, color: "DCE8F0", margin: 0, valign: "middle" });
  footer(s);
  s.addNotes("The turning point. Once broad coverage became the goal, the project needed real architecture and testing discipline. Import, normalization, lowering and codegen needed clear boundaries; unsupported cases needed explicit, testable diagnostics; coverage reports became part of the daily workflow.");
}

/* =================================================================== */
/* SECTION 02 — WHAT IT DOES                                           */
/* =================================================================== */
sectionDivider(2, "What emx-onnx-cgen does", "The compiler and its output", [
  "Ahead-of-time ONNX → portable, deterministic C",
  "A deliberately small, static public API",
  "C is the handoff IR into the embedded-AI toolchain",
]);

/* SLIDE — Overview */
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  kicker(s, "The project");
  title(s, "What emx-onnx-cgen does");
  card(s, 0.6, 1.95, 6.1, 4.55, "Ahead-of-time ONNX → C", [
    "Compiles ONNX models to portable, deterministic C",
    "Targets deeply embedded & resource-constrained systems",
    "No dynamic allocation, no OS dependency, no runtime",
    "No hidden dispatch or callbacks — explicit loops & layout",
    "Auditable, readable, suitable for reproducible builds",
  ]);
  s.addText("Intentionally small public API", { x: 7.0, y: 2.0, w: 5.7, h: 0.4, fontFace: SANS, fontSize: 15, bold: true, color: TEAL2, margin: 0 });
  codeBox(s,
`_Bool model_load(const char *path);

void  model(/* ...inputs... */,
            /* ...outputs... */);

/* tensor IO -> C array parameters
   dynamic dims -> C99 VLAs
   weights      -> optional model.bin */`,
    { x: 7.0, y: 2.45, w: 5.7, h: 2.55, fontSize: 13 });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.0, y: 5.25, w: 5.7, h: 1.25, rectRadius: 0.07, fill: { color: CARD }, line: { color: CARDLN, width: 1 } });
  s.addText("Frontend of the emmtrix embedded-AI flow", { x: 7.2, y: 5.35, w: 5.3, h: 0.35, fontFace: SANS, fontSize: 12.5, bold: true, color: INK, margin: 0 });
  s.addText("ONNX → clean C → Vectorizer → target architecture", { x: 7.2, y: 5.75, w: 5.3, h: 0.6, fontFace: MONO, fontSize: 12.5, color: TEAL2, margin: 0, valign: "top" });
  footer(s);
  s.addNotes("Just enough context before the lessons. AOT ONNX-to-C, portable and deterministic, for embedded targets, avoiding dynamic memory and external runtimes. The generated API is deliberately tiny: a load function and a model function.");
}

/* SLIDE — Output format */
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  kicker(s, "Output format");
  title(s, "What the compiler emits");
  const files = [
    ["<out>.c", "The generated model implementation", true],
    ["<out>_data.c", "Optional — embedded constant/weight data", false],
    ["testbench .c", "Optional — drives verification with recorded IO", false],
    ["<model>.bin", "Optional — external weights loaded at runtime", false],
  ];
  let y = 2.05;
  files.forEach((f) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y, w: 6.6, h: 1.02, rectRadius: 0.07, fill: { color: f[2] ? GREEN : CARD }, line: { color: f[2] ? GREENLN : CARDLN, width: 1 }, shadow: sh({ opacity: 0.08 }) });
    s.addText(f[0], { x: 0.82, y: y + 0.12, w: 6.1, h: 0.4, fontFace: MONO, fontSize: 15, bold: true, color: INK, margin: 0 });
    s.addText(f[1], { x: 0.82, y: y + 0.52, w: 6.1, h: 0.4, fontFace: SANS, fontSize: 12, color: SLATE, margin: 0 });
    y += 1.14;
  });
  card(s, 7.45, 2.05, 5.25, 4.5, "Code-generation principles", [
    "Simple canonical loop forms",
    "Linear array accesses, explicit dims & strides",
    "No hidden pointer aliasing",
    "No dynamic dispatch, no recursion",
    "No dynamic memory allocation",
    "Stable symbol names, deterministic ordering",
  ], { fill: INK, headColor: AMBER, textColor: "DCE8F0", bulletCode: "2713", gap: 9, fontSize: 13 });
  footer(s);
  s.addNotes("Generated artifacts can include the model .c, an optional _data.c for constants, an optional testbench, and an optional .bin for external weights. The code-generation principles come from the emmtrix design goals: canonical loops, linear accesses, no aliasing, no dispatch, no recursion, no heap, explicit dims/strides — all of which make the C friendly to auto-vectorization and safety review.");
}

/* SLIDE — Tensor IO mapping */
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  kicker(s, "ABI");
  title(s, "How tensors and containers map to C parameters");
  codeBox(s,
`/* plain tensor  -> array parameter */
void model(const float x[N][C], float y[N][C]);

/* dynamic dim   -> VLA extent argument */
void model(int N, int C, const float x[N][C], ...);`,
    { x: 0.6, y: 2.0, w: 6.0, h: 1.95, fontSize: 12.5, caption: "Tensors" });
  codeBox(s,
`/* sequence input  */
const T name[EMX_SEQUENCE_MAX_LEN][elem_shape...];
idx_t  name__count;

/* sequence output */
T      name[EMX_SEQUENCE_MAX_LEN][elem_shape...];
idx_t *name__count;`,
    { x: 6.8, y: 2.0, w: 5.9, h: 1.95, fontSize: 12, caption: "Sequences — fixed capacity + count" });
  card(s, 0.6, 4.15, 6.0, 2.55, "Bounds the compiler must invent", [
    "EMX_SEQUENCE_MAX_LEN — default 32, #ifndef-overridable",
    "EMX_STRING_MAX_LEN — default 256, #ifndef-overridable",
    "Ragged inputs: --sequence-element-shape declares maxima",
    "Per-item dims: idx_t name__dim_<axis>[EMX_SEQUENCE_MAX_LEN]",
  ], { fontSize: 12, gap: 7 });
  card(s, 6.8, 4.15, 5.9, 2.55, "Reduced-precision dtypes", [
    "float16 → _Float16 (where supported)",
    "bfloat16 → __bf16 (compiler/target-specific)",
    "int2 / int4 → _BitInt(N), unsigned _BitInt(N) (C23)",
    "FP8 / FP4 → uint8 storage + conversion helpers",
  ], { fill: GREEN, line: GREENLN, headColor: GREENTX, fontSize: 12, gap: 7 });
  footer(s);
  s.addNotes("Tensor IO becomes C array parameters; dynamic dims become VLA extent arguments. Sequences become fixed-capacity arrays plus count metadata (a pointer for outputs). Strings become fixed char slots. Ragged sequences need explicit element-shape hints and per-item dimension arrays. Reduced-precision dtypes map unevenly — covered in detail in the dtype lesson.");
}

/* SLIDE — C as handoff IR */
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  kicker(s, "Positioning");
  title(s, "C is the handoff IR, not just the output");
  // flow strip
  const flow = ["ONNX model", "emx-onnx-cgen", "clean C", "emmtrix Vectorizer", "embedded / safety target"];
  let cx = 0.7; const fy = 2.15;
  flow.forEach((t, i) => {
    const wdt = 0.4 + t.length * 0.118;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: cx, y: fy, w: wdt, h: 0.62, rectRadius: 0.1, fill: { color: i === 1 || i === 2 ? TEAL2 : INK } });
    s.addText(t, { x: cx, y: fy, w: wdt, h: 0.62, fontFace: MONO, fontSize: 12.5, color: LIGHT, align: "center", valign: "middle", margin: 0 });
    cx += wdt;
    if (i < flow.length - 1) { s.addText("→", { x: cx, y: fy, w: 0.42, h: 0.62, fontFace: SANS, fontSize: 18, color: TEAL, align: "center", valign: "middle", margin: 0 }); cx += 0.42; }
  });
  card(s, 0.6, 3.2, 6.0, 3.3, "Like MLIR in spirit — different in handoff", [
    "Staged lowering of a high-level model, as in MLIR/LLVM",
    "But the exposed artifact is standard C, not an internal IR",
    "C is standardized and broadly understood",
    "Reviewable by engineers and static-analysis tools",
    "Compiles with heterogeneous & safety-critical toolchains",
  ], { gap: 8 });
  card(s, 6.8, 3.2, 5.9, 3.3, "So code quality is a contract", [
    "Poor C is not just ugly — it weakens analysis & review",
    "It undermines vectorization and safety arguments",
    "Clean, disciplined C is part of the compiler contract",
    "Used in ISO 26262 / DO-178C contexts with qualification",
  ], { fill: INK, headColor: AMBER, textColor: "DCE8F0", gap: 9 });
  footer(s);
  s.addNotes("emx-onnx-cgen is the AI frontend of the emmtrix flow. We use a disciplined subset of C plus conventions as an intermediate representation — close in spirit to MLIR staged lowering, but the visible artifact is C source, the bridge into embedded source-to-source optimization and safety-critical toolchains. That reframes generated-code quality from cosmetics to a contract.");
}

/* =================================================================== */
/* SECTION 03 — ARCHITECTURE & VERIFICATION                           */
/* =================================================================== */
sectionDivider(3, "Architecture & verification", "How 100% coverage shaped the system", [
  "Coverage as an architectural constraint, not a metric",
  "A deterministic generate → compile → compare loop",
  "ORT-derived artifacts make coverage realistic",
]);

/* SLIDE — Architecture forced */
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  kicker(s, "Architecture");
  title(s, "100% coverage shaped the architecture");
  s.addText("Goal: 100% test coverage across the compiler pipeline — not just a metric, an architectural constraint.", {
    x: 0.6, y: 1.82, w: 12.1, h: 0.45, fontFace: SANS, fontSize: 14.5, color: SLATE, margin: 0 });
  const forced = ["Pass-based architecture", "Explicit IR boundaries", "Deterministic codegen", "Precise, testable diagnostics", "Reference-based verification", "Explicit context, no global state"];
  const discouraged = ["Hidden global state", "Oversized “god” modules", "Implicit shape assumptions", "Broad mutation in passes", "Codegen driven by verify-only inputs"];
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 2.45, w: 6.0, h: 4.15, rectRadius: 0.08, fill: { color: GREEN }, line: { color: GREENLN, width: 1 }, shadow: sh({ opacity: 0.10 }) });
  s.addText("It forced", { x: 0.9, y: 2.65, w: 5.4, h: 0.4, fontFace: SANS, fontSize: 16, bold: true, color: GREENTX, margin: 0 });
  s.addText(forced.map((t) => ({ text: t, options: { bullet: { code: "2713" }, color: SLATE, breakLine: true, paraSpaceAfter: 8 } })), { x: 0.9, y: 3.1, w: 5.4, h: 3.4, fontFace: SANS, fontSize: 13.5, margin: 0, valign: "top" });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 6.85, y: 2.45, w: 5.85, h: 4.15, rectRadius: 0.08, fill: { color: RUST }, line: { color: RUSTLN, width: 1 }, shadow: sh({ opacity: 0.10 }) });
  s.addText("It discouraged", { x: 7.15, y: 2.65, w: 5.3, h: 0.4, fontFace: SANS, fontSize: 16, bold: true, color: RUSTTX, margin: 0 });
  s.addText(discouraged.map((t) => ({ text: t, options: { bullet: { code: "2715" }, color: SLATE, breakLine: true, paraSpaceAfter: 8 } })), { x: 7.15, y: 3.1, w: 5.3, h: 3.4, fontFace: SANS, fontSize: 13.5, margin: 0, valign: "top" });
  footer(s);
  s.addNotes("Coverage is not just a number — it forced the system to be modular and deterministic. Small functions, clear pass contracts, explicit context objects, deterministic output for stable golden tests. The point: the coverage target produced better compiler architecture.");
}

/* SLIDE — Verification loop */
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
  const bw = 2.32, gap = 0.13, x0 = 0.6, y0 = 2.25, bh = 2.4;
  steps.forEach((st, i) => {
    const x = x0 + i * (bw + gap);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: y0, w: bw, h: bh, rectRadius: 0.08, fill: { color: i % 2 ? CARD : GREEN }, line: { color: CARDLN, width: 1 }, shadow: sh({ opacity: 0.09 }) });
    s.addText(String(i + 1), { x: x + 0.18, y: y0 + 0.16, w: 0.5, h: 0.5, fontFace: SERIF, fontSize: 22, bold: true, color: TEAL, margin: 0 });
    s.addText(st[0], { x: x + 0.2, y: y0 + 0.74, w: bw - 0.4, h: 0.45, fontFace: SANS, fontSize: 14.5, bold: true, color: INK, margin: 0 });
    s.addText(st[1], { x: x + 0.2, y: y0 + 1.2, w: bw - 0.4, h: 1.1, fontFace: SANS, fontSize: 11.5, color: SLATE, margin: 0, valign: "top" });
    if (i < steps.length - 1) s.addText("›", { x: x + bw - 0.05, y: y0, w: gap + 0.1, h: bh, fontFace: SANS, fontSize: 20, bold: true, color: TEAL, align: "center", valign: "middle", margin: 0 });
  });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 5.15, w: 12.1, h: 1.55, rectRadius: 0.07, fill: { color: INK } });
  s.addText("Invariant", { x: 0.95, y: 5.3, w: 3, h: 0.4, fontFace: SANS, fontSize: 14, bold: true, color: AMBER, margin: 0 });
  s.addText("Verification-only inputs must never implicitly change generated code. If verify uses extra shape information, it is testing a different compiler path — compile/verify parity keeps the contract honest.", {
    x: 0.95, y: 5.7, w: 11.5, h: 0.85, fontFace: SANS, fontSize: 14, color: "DCE8F0", margin: 0, valign: "top" });
  footer(s);
  s.addNotes("Floating outputs compared with ULP thresholds after an absolute epsilon; non-floating exactly. Compile/verify parity is the key invariant.");
}

/* SLIDE — Parity invariant deep */
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  kicker(s, "Invariant");
  title(s, "Compile/verify parity in detail");
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 1.95, w: 6.0, h: 3.0, rectRadius: 0.08, fill: { color: RUST }, line: { color: RUSTLN, width: 1 }, shadow: sh({ opacity: 0.10 }) });
  s.addText("✗  Bad pattern", { x: 0.85, y: 2.1, w: 5.5, h: 0.4, fontFace: SANS, fontSize: 15, bold: true, color: RUSTTX, margin: 0 });
  s.addText([
    { text: "verify reads representative input_*.pb files", options: { bullet: { code: "2022" }, breakLine: true, paraSpaceAfter: 7 } },
    { text: "those values or shapes make codegen succeed", options: { bullet: { code: "2022" }, breakLine: true, paraSpaceAfter: 7 } },
    { text: "plain compile would fail or emit different code", options: { bullet: { code: "2022" }, breakLine: true, paraSpaceAfter: 7 } },
    { text: "→ test results no longer reflect real compilation", options: { color: RUSTTX, bold: true } },
  ], { x: 0.85, y: 2.55, w: 5.5, h: 2.3, fontFace: SANS, fontSize: 13, color: SLATE, margin: 0, valign: "top" });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 6.8, y: 1.95, w: 5.9, h: 3.0, rectRadius: 0.08, fill: { color: GREEN }, line: { color: GREENLN, width: 1 }, shadow: sh({ opacity: 0.10 }) });
  s.addText("✓  Project principle", { x: 7.05, y: 2.1, w: 5.4, h: 0.4, fontFace: SANS, fontSize: 15, bold: true, color: GREENTX, margin: 0 });
  s.addText([
    { text: "Models needing representative inputs to resolve dynamic shapes must fail clearly…", options: { bullet: { code: "2022" }, breakLine: true, paraSpaceAfter: 7 } },
    { text: "…or require an explicit compiler option that compile also accepts", options: { bullet: { code: "2022" }, breakLine: true, paraSpaceAfter: 7 } },
    { text: "Export with static shapes, or supply explicit options", options: { bullet: { code: "2022" } } },
  ], { x: 7.05, y: 2.55, w: 5.4, h: 2.3, fontFace: SANS, fontSize: 13, color: SLATE, margin: 0, valign: "top" });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 5.15, w: 12.1, h: 1.55, rectRadius: 0.07, fill: { color: INK } });
  s.addText("Why it matters", { x: 0.95, y: 5.3, w: 4, h: 0.4, fontFace: SANS, fontSize: 14, bold: true, color: AMBER, margin: 0 });
  s.addText("Reproducibility · deterministic code generation · an honest compiler contract · meaningful test results.", {
    x: 0.95, y: 5.72, w: 11.5, h: 0.8, fontFace: SANS, fontSize: 14, color: "DCE8F0", margin: 0, valign: "top" });
  footer(s);
  s.addNotes("Verification must not give the compiler information that normal compilation does not receive. If verify quietly fills codegen gaps from representative inputs, the test passes but the compiler contract is broken.");
}

/* SLIDE — ORT artifacts */
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  kicker(s, "Realistic coverage");
  title(s, "ORT-derived artifacts: emx-ort-test-artifacts");
  codeBox(s,
`ORT test case
   -> exported ONNX model + test_data_set_*
   -> backend-runnable test artifact`,
    { x: 0.6, y: 2.0, w: 6.0, h: 1.4, fontSize: 13, caption: "The export principle" });
  card(s, 0.6, 3.6, 6.0, 3.1, "Why it matters", [
    "Official ONNX backend tests are the standards baseline",
    "ORT artifacts apply the same model+test-data pattern broadly",
    "Includes contrib/operator scenarios & runtime-style edge cases",
    "Forces real model/test directory handling, not micrographs",
    "Preserves ORT's intended numerical tolerances",
    "Failures stay reproducible via recorded command lines",
  ], { fontSize: 12, gap: 6 });
  card(s, 6.8, 2.0, 5.9, 2.55, "validation.json fields", [
    "expects_failure / expected_failure_substring",
    "per-output relative_error",
    "per-output absolute_error",
    "per-output sort_output",
  ], { fill: GREEN, line: GREENLN, headColor: GREENTX, bulletCode: "2022", fontSize: 12.5, gap: 7 });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 6.8, y: 4.75, w: 5.9, h: 1.95, rectRadius: 0.08, fill: { color: INK } });
  s.addText("Proposed: convert to ONNX backend data.json", { x: 7.05, y: 4.9, w: 5.4, h: 0.4, fontFace: SANS, fontSize: 13.5, bold: true, color: AMBER, margin: 0 });
  s.addText("Each test becomes self-contained — model, protobuf test data, and comparison tolerances in the expected ONNX backend-test convention. Other backends could reuse ORT-derived tests outside the ORT C++ harness.", {
    x: 7.05, y: 5.32, w: 5.4, h: 1.3, fontFace: SANS, fontSize: 12, color: "CFE0EC", margin: 0, valign: "top" });
  footer(s);
  s.addNotes("Don't make exact percentages the point — the value is the engineering loop: broad corpus, deterministic verification, reproducible failures, visible progress. The corpus lives as emx-ort-test-artifacts-org/, tracked separately in ONNX_SUPPORT.md. Converting validation.json to ONNX backend data.json would make each artifact a native-looking backend test. The idea was raised in ORT forums with near-zero response — present as an underused community opportunity, not a complaint.");
}

/* =================================================================== */
/* SECTION 04 — ONNX LESSONS                                           */
/* =================================================================== */
sectionDivider(4, "ONNX lessons learned", "Where a flexible interchange format meets a static compiler", [
  "Lesson 1 — ONNX is effectively unbounded",
  "Lesson 2 — type & shape inference is not enough",
  "Lesson 3 — numerical accuracy is underspecified",
  "Plus: dtype mapping & generated-code quality",
]);

/* SLIDE — Lesson 1 overview */
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  lessonHead(s, 1, "ONNX is effectively unbounded");
  const items = [
    ["Dynamic dimensions", "Symbolic/unknown axes give a runtime extent, but no maximum"],
    ["Sequence length", "Containers have no natural capacity — one level above a dynamic dim"],
    ["String size", "String tensors carry no maximum character length"],
  ];
  const bw = 3.95, gap = 0.28, x0 = 0.6, y0 = 1.95, bh = 2.5;
  items.forEach((it, i) => {
    const x = x0 + i * (bw + gap);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: y0, w: bw, h: bh, rectRadius: 0.08, fill: { color: CARD }, line: { color: CARDLN, width: 1 }, shadow: sh({ opacity: 0.10 }) });
    s.addText(it[0], { x: x + 0.25, y: y0 + 0.25, w: bw - 0.5, h: 0.85, fontFace: SANS, fontSize: 16.5, bold: true, color: TEAL2, margin: 0, valign: "top" });
    s.addText(it[1], { x: x + 0.25, y: y0 + 1.08, w: bw - 0.5, h: 1.3, fontFace: SANS, fontSize: 13, color: SLATE, margin: 0, valign: "top" });
  });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 4.7, w: 12.1, h: 2.0, rectRadius: 0.07, fill: { color: INK } });
  s.addText("The core AOT lesson", { x: 0.9, y: 4.85, w: 11.5, h: 0.4, fontFace: SANS, fontSize: 14.5, bold: true, color: AMBER, margin: 0 });
  s.addText("ONNX is intentionally flexible for interchange and runtime execution. General static memory planning is impossible without additional assumptions. A tensor dimension asks “how large is this axis?”; a sequence asks “how many tensor objects exist, and what shape is each?” — comfortable for runtimes, hard for static C. Embedded C needs concrete bounds, or a clear failure.", {
    x: 0.9, y: 5.28, w: 11.5, h: 1.3, fontFace: SANS, fontSize: 14, color: "DCE8F0", margin: 0, valign: "top" });
  footer(s);
  s.addNotes("The central AOT lesson. ONNX's flexibility is valuable for interchange but becomes a problem for deterministic AOT unless constrained. The next three slides show how emx-onnx-cgen bounds dynamic dims, sequences and strings.");
}

/* SLIDE — Lesson 1a dynamic dims */
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  kicker(s, "Lesson 1 · dynamic dimensions");
  title(s, "Runtime extent ≠ maximum extent");
  card(s, 0.6, 1.95, 6.0, 2.6, "The gap", [
    "ONNX can mark dimensions symbolic or unknown",
    "That gives the compiler a runtime extent…",
    "…but never a maximum extent",
    "VLAs express runtime dims in the ABI — not a memory bound",
  ], { gap: 7 });
  card(s, 0.6, 4.75, 6.0, 1.95, "A backend without heap must…", [
    "require maxima, or",
    "require caller-provided storage with capacities, or",
    "accept unbounded stack usage, or reject the model",
  ], { fill: SAND, line: SANDLN, headColor: SANDTX, gap: 6 });
  codeBox(s,
`/* extent N, C known only at runtime */
void model(int N, int C,
           const float x[restrict N][C],
           float       y[restrict N][C]);

/* temporaries & caller buffers can still
   grow without a compile-time bound      */`,
    { x: 6.8, y: 1.95, w: 5.9, h: 2.6, fontSize: 12.5, caption: "VLAs carry the extent, not the bound" });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 6.8, y: 4.75, w: 5.9, h: 1.95, rectRadius: 0.08, fill: { color: GREEN }, line: { color: GREENLN, width: 1 } });
  s.addText("emx-onnx-cgen policy", { x: 7.05, y: 4.88, w: 5.4, h: 0.4, fontFace: SANS, fontSize: 14, bold: true, color: GREENTX, margin: 0 });
  s.addText("Require static shapes where necessary; support VLAs where the shape is representable and acceptable; fail clearly when a required bound is missing. Models that need representative inputs to resolve shapes should be exported static or given explicit options.", {
    x: 7.05, y: 5.3, w: 5.4, h: 1.3, fontFace: SANS, fontSize: 12, color: SLATE, margin: 0, valign: "top" });
  footer(s);
  s.addNotes("A dynamic dimension gives a runtime extent but no maximum, so static memory planning needs an extra assumption. VLAs represent the runtime extent in the ABI but do not bound temporaries or caller buffers.");
}

/* SLIDE — Lesson 1b sequences */
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  kicker(s, "Lesson 1 · sequences");
  title(s, "Sequences: a dynamic length one level higher");
  codeBox(s,
`/* input  */
const T name[EMX_SEQUENCE_MAX_LEN][elem_shape...];
idx_t  name__count;

/* output */
T      name[EMX_SEQUENCE_MAX_LEN][elem_shape...];
idx_t *name__count;

#ifndef EMX_SEQUENCE_MAX_LEN
#define EMX_SEQUENCE_MAX_LEN 32   /* overridable */
#endif`,
    { x: 0.6, y: 2.0, w: 6.1, h: 3.6, fontSize: 12, caption: "Fixed-capacity array + count metadata" });
  card(s, 6.9, 2.0, 5.8, 1.95, "Why containers are hard", [
    "A sequence is a container value, not a tensor with one more axis",
    "Needs max length, element dtype, rank and element shape",
    "Without a max, no fixed storage can be reserved",
  ], { fontSize: 12, gap: 6 });
  card(s, 6.9, 4.1, 5.8, 2.6, "Ragged sequences", [
    "Not accepted implicitly",
    "--sequence-element-shape declares rank + per-axis maxima",
    "e.g. sequence=[<=8] or boxes=[<=100,4]",
    "Per-item dims: idx_t name__dim_<axis>[…MAX_LEN]",
    "Only first name__count entries are meaningful",
  ], { fill: SAND, line: SANDLN, headColor: SANDTX, fontSize: 11.5, gap: 5 });
  footer(s);
  s.addNotes("A sequence asks how many tensor objects exist and what shape each has. Runtimes treat it as a growable object; static C needs an explicit capacity contract. emx represents sequence IO as fixed-capacity arrays plus count metadata, default capacity 32, overridable. Ragged inputs need explicit element-shape hints and carry per-item dimension arrays; testbench JSON records item_shapes so verify compares true per-item shapes, not padded storage.");
}

/* SLIDE — Lesson 1c strings */
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  kicker(s, "Lesson 1 · strings");
  title(s, "Strings: the same unbounded-length problem");
  card(s, 0.6, 1.95, 6.0, 2.55, "The gap", [
    "ONNX string tensors declare no maximum length",
    "A heap backend stores variable-length strings at runtime",
    "A no-heap backend must reserve storage at compile time",
    "Options: require a bound, pick a policy, truncate, or reject",
  ], { gap: 7 });
  card(s, 0.6, 4.7, 6.0, 2.0, "Directly analogous to sequences", [
    "Strings: missing bound = max characters per element",
    "Sequences: missing bound = max number of elements",
    "Both: unbounded length → bounded static storage",
  ], { fill: GREEN, line: GREENLN, headColor: GREENTX, gap: 7 });
  codeBox(s,
`char name[...][EMX_STRING_MAX_LEN];  /* per element */

#ifndef EMX_STRING_MAX_LEN
#define EMX_STRING_MAX_LEN 256   /* overridable */
#endif

/* testbench: strings over the slot are truncated
   to EMX_STRING_MAX_LEN - 1 bytes and null-padded */`,
    { x: 6.8, y: 1.95, w: 5.9, h: 3.1, fontSize: 12, caption: "Fixed-size '\\0'-terminated C string slots" });
  s.addText("Trade-off: arbitrary ONNX string length is not represented without an explicit bound — but memory stays static and deterministic.", {
    x: 6.8, y: 5.25, w: 5.9, h: 1.4, fontFace: SANS, fontSize: 12.5, italic: true, color: MUTED, margin: 0, valign: "top" });
  footer(s);
  s.addNotes("Strings are the string-length twin of the sequence-length problem. emx uses fixed-size null-terminated char slots, default 256, overridable; testbench serialization truncates over-long strings so the result stays null-terminated. The trade-off: arbitrary string length is not represented without a bound, but layout stays static and deterministic.");
}

/* SLIDE — Lesson 2 inference */
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  lessonHead(s, 2, "Type & shape inference is not enough");
  card(s, 0.6, 1.95, 6.0, 2.55, "Where it falls short", [
    "Inference is incomplete for some operators",
    "Output shape can depend on values, attributes or subgraphs",
    "Container element shapes may simply be missing",
    "Dynamic-output operators need capacity decisions",
  ], { gap: 7 });
  card(s, 0.6, 4.7, 6.0, 2.0, "Compiler response", [
    "Normalize attributes & types on import",
    "Add compiler-side inference/validation where ONNX stops",
    "Make missing information an explicit error",
  ], { fill: GREEN, line: GREENLN, headColor: GREENTX, gap: 7 });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 6.8, y: 1.95, w: 5.9, h: 4.75, rectRadius: 0.08, fill: { color: RUST }, line: { color: RUSTLN, width: 1 }, shadow: sh({ opacity: 0.10 }) });
  s.addText("The duplication problem", { x: 7.05, y: 2.1, w: 5.4, h: 0.4, fontFace: SANS, fontSize: 15.5, bold: true, color: RUSTTX, margin: 0 });
  s.addText([
    { text: "No reliable standalone library infers shapes/types for dynamic models well enough for static code generators.", options: { breakLine: true, paraSpaceAfter: 9 } },
    { text: "So importers, MLIR frontends, code generators, verifiers and optimizers each reimplement parts of it…", options: { breakLine: true, paraSpaceAfter: 9 } },
    { text: "…and may produce different answers for the same model.", options: { color: RUSTTX, bold: true, breakLine: true, paraSpaceAfter: 9 } },
    { text: "Shape/type inference should be separable both from code generation and from a fixed operator universe.", options: { italic: true } },
  ], { x: 7.05, y: 2.55, w: 5.45, h: 4.0, fontFace: SANS, fontSize: 13, color: SLATE, margin: 0, valign: "top" });
  footer(s);
  s.addNotes("Frame as mismatch, not only a flaw: a flexible model format versus a static compiler input. The duplication problem is the community-facing gap — every serious backend reimplements inference and they can disagree. The next slide is the vision for fixing it.");
}

/* SLIDE — Lesson 2 vision */
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  kicker(s, "Lesson 2 · vision  (research direction)");
  title(s, "A shared, extensible inference layer");
  const cards = [
    ["Reusable", "A shared library that understands symbolic & dynamic dims, preserves constraints, and reports unresolved facts explicitly"],
    ["Extensible", "A registry/plugin for external operator domains — e.g. Microsoft/ORT contrib ops, not just ONNX core"],
    ["Persistable", "Write inferred results back into the model (enriched value_info / metadata) so backends consume, not recompute"],
  ];
  const bw = 3.95, gap = 0.28, x0 = 0.6, y0 = 2.0, bh = 2.45;
  cards.forEach((c, i) => {
    const x = x0 + i * (bw + gap);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: y0, w: bw, h: bh, rectRadius: 0.08, fill: { color: CARD }, line: { color: CARDLN, width: 1 }, shadow: sh({ opacity: 0.10 }) });
    s.addText(c[0], { x: x + 0.25, y: y0 + 0.22, w: bw - 0.5, h: 0.45, fontFace: SANS, fontSize: 16, bold: true, color: TEAL2, margin: 0 });
    s.addText(c[1], { x: x + 0.25, y: y0 + 0.72, w: bw - 0.5, h: 1.6, fontFace: SANS, fontSize: 12.5, color: SLATE, margin: 0, valign: "top" });
  });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 4.7, w: 12.1, h: 2.0, rectRadius: 0.07, fill: { color: INK } });
  s.addText([{ text: "ONE DSL  ", options: { bold: true, color: AMBER } }, { text: "→ shape/type rules written once", options: { bold: true, color: LIGHT } }], { x: 0.9, y: 4.85, w: 11.5, h: 0.4, fontFace: SANS, fontSize: 15, margin: 0 });
  s.addText([
    { text: "Drives both C++ and Python inference", options: { bullet: { code: "2022" }, color: "CFE0EC", breakLine: true, paraSpaceAfter: 6 } },
    { text: "Numeric evaluation for concrete dims; symbolic evaluation (e.g. SymPy) for symbolic dims", options: { bullet: { code: "2022" }, color: "CFE0EC", breakLine: true, paraSpaceAfter: 6 } },
    { text: "Custom operators ship their own rule instead of every backend hard-coding it — could even live inside ONNX models", options: { bullet: { code: "2022" }, color: "CFE0EC" } },
  ], { x: 0.9, y: 5.3, w: 11.5, h: 1.35, fontFace: SANS, fontSize: 12.5, margin: 0, valign: "top" });
  footer(s);
  s.addNotes("Present as vision and research direction, not finished work. A rule written once, runnable from C++ and Python, numeric for concrete dims and symbolic (SymPy) for symbolic dims, would make custom operators tractable and could be stored inside ONNX models. Same spirit as the ORT-artifacts idea: make compiler knowledge portable as model artifacts rather than trapping it in one backend.");
}

/* SLIDE — Lesson 3 accuracy */
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  lessonHead(s, 3, "Numerical accuracy is underspecified");
  card(s, 0.6, 1.95, 6.0, 2.85, "Sources of ambiguity", [
    "Official tests give examples, not full accuracy contracts",
    "Accumulation precision, rounding, approximations",
    "Math-library & implementation-defined edge cases",
    "ONNX Runtime vs ONNX Reference behaviour differ",
    "node tests carry no per-operator tolerance (data.json = real models)",
  ], { fontSize: 12.5, gap: 6 });
  card(s, 0.6, 4.95, 6.0, 1.75, "ONNX backend default check", [
    "rtol = 1e-3, atol = 1e-7, overridable per real-model test",
    "abs(actual − expected) ≤ atol + rtol·|expected|",
  ], { fill: CARD, fontSize: 12.5, gap: 6 });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 6.8, y: 1.95, w: 5.9, h: 4.75, rectRadius: 0.08, fill: { color: INK }, shadow: sh({ opacity: 0.12 }) });
  s.addText("Why this matters to backend authors", { x: 7.05, y: 2.1, w: 5.4, h: 0.4, fontFace: SANS, fontSize: 14.5, bold: true, color: AMBER, margin: 0 });
  s.addText([
    { text: "Validation is an engineering policy unless the standard defines the contract.", options: { color: "DCE8F0", breakLine: true, paraSpaceAfter: 10 } },
    { text: "A single fixed rtol/atol is not equally meaningful across dtypes — float16, float32 and float64 have very different spacing between representable values.", options: { color: "DCE8F0", breakLine: true, paraSpaceAfter: 10 } },
    { text: "ULP-based measurement asks the real question: how many representable floating-point values apart are these results?", options: { color: LIGHT, bold: true } },
  ], { x: 7.05, y: 2.55, w: 5.45, h: 4.0, fontFace: SANS, fontSize: 13.5, margin: 0, valign: "top" });
  footer(s);
  s.addNotes("Highly relevant to backend authors. Official tests rarely express a complete tolerance policy. A fixed rtol/atol is not equally meaningful across dtypes. The next slide compares how different frameworks handle this; then emx's ULP approach.");
}

/* SLIDE — Lesson 3 tolerance table */
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  kicker(s, "Lesson 3 · landscape");
  title(s, "Everyone picks a different tolerance policy");
  const header = [
    { text: "Framework / tool", options: { bold: true, color: LIGHT, fill: { color: INK } } },
    { text: "Mechanism", options: { bold: true, color: LIGHT, fill: { color: INK } } },
    { text: "Notable point", options: { bold: true, color: LIGHT, fill: { color: INK } } },
  ];
  const rows = [
    ["ONNX backend", "assert_allclose", "rtol 1e-3, atol 1e-7; data.json only for real models"],
    ["ONNX Runtime", "per-sample tol", "1e-3 defaults; override via config.txt"],
    ["PyTorch", "assert_close", "dtype-dependent defaults (fp16/bf16/fp32/fp64)"],
    ["NumPy / JAX", "allclose", "atol + rtol·|expected|; differing defaults"],
    ["TensorFlow", "assertAllClose…", "dtype-aware: AllCloseAccordingToType"],
    ["MXNet", "assert_almost_equal", "equal if relative OR absolute check passes"],
    ["NVIDIA Polygraphy", "comparator CLI", "per-output rtol/atol; FP32 too strict for FP16/INT8"],
  ];
  const body = rows.map((r, i) => r.map((c, j) => ({
    text: c,
    options: { fill: { color: i % 2 ? "FFFFFF" : CARD }, color: j === 0 ? INK : SLATE, bold: j === 0, fontFace: j === 1 ? MONO : SANS, fontSize: j === 1 ? 11 : 12 },
  })));
  s.addTable([header, ...body], { x: 0.6, y: 2.0, w: 12.1, colW: [2.9, 2.8, 6.4], border: { pt: 0.5, color: CARDLN }, align: "left", valign: "middle", rowH: 0.55, margin: [3, 6, 3, 6] });
  s.addText("Across the ecosystem the comparison shape is similar (absolute + relative), but defaults and dtype-awareness vary widely — there is no single shared contract.", {
    x: 0.6, y: 6.55, w: 12.1, h: 0.5, fontFace: SANS, fontSize: 12.5, italic: true, color: MUTED, margin: 0 });
  footer(s);
  s.addNotes("Walk the table briefly. The point is not the exact numbers but that every framework chooses its own defaults and dtype handling — backend authors would benefit from clearer numerical expectations in operator specs and official tests.");
}

/* SLIDE — Lesson 3 emx ULP */
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  kicker(s, "Lesson 3 · the emx approach");
  title(s, "ULP-based, dtype-aware verification");
  codeBox(s,
`/* 1. ignore tiny absolute differences   */
diff <= atol_eps * eps(dtype)   ->  ok

/* 2. else measure ULP distance          */
ulp = ulp_distance(actual, expected)
report max ulp;  ulp <= max_ulp  ->  ok

/* non-floating outputs must match exactly */

CLI:  --atol-eps  --max-ulp
      --fp32-accumulation-strategy fp64`,
    { x: 0.6, y: 2.0, w: 6.2, h: 3.7, fontSize: 12.5, caption: "How emx-onnx-cgen compares outputs" });
  card(s, 7.1, 2.0, 5.6, 1.95, "Why ULP, not fixed rtol/atol", [
    "fp16 / fp32 / fp64 spacing differs enormously",
    "ULP = how many representable values apart",
    "Closer to the question that actually matters",
  ], { fill: GREEN, line: GREENLN, headColor: GREENTX, fontSize: 12.5, gap: 6 });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.1, y: 4.1, w: 5.6, h: 1.6, rectRadius: 0.08, fill: { color: INK } });
  s.addText("Consequence in practice", { x: 7.35, y: 4.22, w: 5.1, h: 0.4, fontFace: SANS, fontSize: 13.5, bold: true, color: AMBER, margin: 0 });
  s.addText("Some ONNX reference outputs appear computed at very high precision. Under strict ULP checks, this required 64-bit accumulation for selected official tests — exposed as --fp32-accumulation-strategy fp64.", {
    x: 7.35, y: 4.62, w: 5.1, h: 1.05, fontFace: SANS, fontSize: 12, color: "CFE0EC", margin: 0, valign: "top" });
  s.addText("emx ignores absolute diffs up to atol_eps·eps(dtype), then measures ULP distance — exact match required for non-float outputs.", {
    x: 0.6, y: 5.95, w: 6.2, h: 0.7, fontFace: SANS, fontSize: 12, italic: true, color: MUTED, margin: 0, valign: "top" });
  footer(s);
  s.addNotes("emx does not use a classic fixed rtol/atol. It gates on an absolute epsilon scaled by eps(dtype), then measures ULP distance, reporting the max; non-float must match exactly. Because some reference outputs are effectively high precision, strict ULP forced fp64 accumulation on selected tests, exposed via --fp32-accumulation-strategy fp64 and used in several official-test expectation files.");
}

/* SLIDE — dtype mapping */
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  kicker(s, "Lesson · dtype mapping");
  title(s, "ONNX dtype mapping is uneven in C");
  const header = [
    { text: "ONNX type", options: { bold: true, color: LIGHT, fill: { color: INK } } },
    { text: "C representation", options: { bold: true, color: LIGHT, fill: { color: INK } } },
    { text: "Note", options: { bold: true, color: LIGHT, fill: { color: INK } } },
  ];
  const rows = [
    ["int / uint, bool", "fixed-width ints, _Bool", "boring in a good way", GREEN],
    ["float, double", "float, double", "natural mapping", GREEN],
    ["float16", "_Float16", "where compiler/target supports it", CARD],
    ["bfloat16", "__bf16", "compiler/target-specific extension", CARD],
    ["int2 / int4 / uint4", "_BitInt(N) (C23)", "type range only — arrays may waste memory", SAND],
    ["FP8 / FP4", "uint8 + converters", "not standard C arithmetic types", SAND],
  ];
  const body = rows.map((r, i) => [
    { text: r[0], options: { fill: { color: r[3] }, color: INK, bold: true, fontFace: MONO, fontSize: 12 } },
    { text: r[1], options: { fill: { color: r[3] }, color: SLATE, fontFace: MONO, fontSize: 12 } },
    { text: r[2], options: { fill: { color: r[3] }, color: SLATE, fontSize: 12 } },
  ]);
  s.addTable([header, ...body], { x: 0.6, y: 2.0, w: 12.1, colW: [3.0, 3.2, 5.9], border: { pt: 0.5, color: CARDLN }, align: "left", valign: "middle", rowH: 0.6, margin: [3, 6, 3, 6] });
  s.addText([
    { text: "Takeaway:  ", options: { bold: true, color: AMBER } },
    { text: "common numeric types are easy; reduced-precision and sub-byte types expose the gap between ONNX's type system and portable C. FP8 today is more useful for type coverage than for efficient portable execution.", options: { color: SLATE } },
  ], { x: 0.6, y: 6.4, w: 12.1, h: 0.7, fontFace: SANS, fontSize: 12.5, margin: 0, valign: "top" });
  footer(s);
  s.addNotes("Common numeric types map cleanly. float16 uses _Float16 where supported; bfloat16 relies on __bf16 extensions; 2/4-bit ints use C23 _BitInt(N) but arrays can waste memory versus packed storage; FP8/FP4 aren't standard C arithmetic types and are emulated with integer storage plus conversion helpers. FP8 is currently more about completeness than efficiency.");
}

/* SLIDE — generated-code quality */
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  kicker(s, "Lesson · code quality");
  title(s, "Generated-code quality is a backend feature");
  card(s, 0.6, 1.95, 6.0, 4.55, "Desired properties of generated C", [
    "Readable and reviewable",
    "Deterministic and stable across runs",
    "Auditable, explicit memory layout",
    "No dynamic allocation",
    "Minimal runtime dependencies",
    "Simple, small public API",
  ], { fill: GREEN, line: GREENLN, headColor: GREENTX, bulletCode: "2713", gap: 9 });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 6.8, y: 1.95, w: 5.9, h: 2.2, rectRadius: 0.08, fill: { color: CARD }, line: { color: CARDLN, width: 1 }, shadow: sh({ opacity: 0.10 }) });
  s.addText("External validation", { x: 7.05, y: 2.1, w: 5.4, h: 0.4, fontFace: SANS, fontSize: 15, bold: true, color: TEAL2, margin: 0 });
  s.addText("The public ONNX Backend Scoreboard lists emx-onnx-cgen directly after ONNX Reference in the stable-build table — a useful compatibility signal, since it is based on ONNX backend unit tests.", {
    x: 7.05, y: 2.52, w: 5.4, h: 1.55, fontFace: SANS, fontSize: 13, color: SLATE, margin: 0, valign: "top" });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 6.8, y: 4.35, w: 5.9, h: 2.15, rectRadius: 0.08, fill: { color: INK } });
  s.addText("But unit-test execution ≠ backend quality", { x: 7.05, y: 4.48, w: 5.4, h: 0.4, fontFace: SANS, fontSize: 13.5, bold: true, color: AMBER, margin: 0 });
  s.addText("The scoreboard does not capture deterministic generated C, static memory planning, readable source, auditability, or safety-oriented build integration. For embedded and safety-adjacent targets, those are the real backend features.", {
    x: 7.05, y: 4.9, w: 5.4, h: 1.5, fontFace: SANS, fontSize: 12.5, color: "CFE0EC", margin: 0, valign: "top" });
  footer(s);
  s.addNotes("For embedded systems generated C has different requirements than a hidden runtime. Use the scoreboard placement as credibility, then pivot: 'can execute the unit tests' is only one dimension of backend quality. Determinism, readability, static planning and auditability matter for safety-adjacent targets.");
}

/* =================================================================== */
/* SECTION 05 — STATUS & COMMUNITY                                     */
/* =================================================================== */
sectionDivider(5, "Status & community", "Where it stands and what to take away", [
  "Opset 26, broad ORT operator support, visible coverage",
  "The scoreboard as credibility — with caveats",
  "What an AOT-friendly ONNX could specify",
]);

/* SLIDE — current status */
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
  const bw = 3.95, gap = 0.28, x0 = 0.6, y0 = 2.05, bh = 1.85;
  stats.forEach((st, i) => {
    const x = x0 + i * (bw + gap);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: y0, w: bw, h: bh, rectRadius: 0.08, fill: { color: INK }, shadow: sh({ opacity: 0.14 }) });
    s.addText(st[0], { x: x + 0.25, y: y0 + 0.2, w: bw - 0.5, h: 0.85, fontFace: SERIF, fontSize: 38, bold: true, color: AMBER, margin: 0, valign: "middle" });
    s.addText(st[1], { x: x + 0.25, y: y0 + 1.0, w: bw - 0.5, h: 0.75, fontFace: SANS, fontSize: 13, color: "CFE0EC", margin: 0, valign: "top" });
  });
  card(s, 0.6, 4.2, 6.0, 2.5, "Coverage is reported, reproducible, visible", [
    "SUPPORT_OPS.md — operator support status",
    "ONNX_SUPPORT.md — official backend model coverage",
    "ONNX_SUPPORT.md — ORT artifact corpus coverage",
    "Unsupported cases documented as expected errors",
  ], { headColor: TEAL2, gap: 6 });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 6.8, y: 4.2, w: 5.9, h: 2.5, rectRadius: 0.08, fill: { color: CARD }, line: { color: CARDLN, width: 1 }, shadow: sh({ opacity: 0.10 }) });
  s.addText("emx-ort-test-artifacts", { x: 7.05, y: 4.35, w: 5.4, h: 0.4, fontFace: SANS, fontSize: 15, bold: true, color: INK, margin: 0 });
  s.addText("Exports ORT tests into backend-test-like artifacts (model.onnx + test_data_set_*), reusable outside the ORT C++ harness — a broader, runtime-oriented reality check beyond the compact official node tests.", {
    x: 7.05, y: 4.78, w: 5.4, h: 1.1, fontFace: SANS, fontSize: 12, color: SLATE, margin: 0, valign: "top" });
  s.addText("ORT test case → ONNX model + test_data_set_* → backend-runnable", {
    x: 7.05, y: 6.05, w: 5.4, h: 0.55, fontFace: MONO, fontSize: 10.5, color: TEAL2, margin: 0, valign: "top" });
  footer(s);
  s.addNotes("Phrase coverage as report-based and corpus-based; don't make exact percentages the point. The real message: coverage became systematic, reproducible and visible.");
}

/* SLIDE — scoreboard framing */
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  kicker(s, "Credibility — with caveats");
  title(s, "Reading the ONNX Backend Scoreboard");
  card(s, 0.6, 1.95, 6.0, 4.55, "What it does tell us", [
    "emx-onnx-cgen is a visible ONNX backend, not a local experiment",
    "Listed directly after ONNX Reference (current stable build)",
    "Score is based on the ONNX backend unit tests",
    "A genuine, external compatibility signal",
  ], { fill: GREEN, line: GREENLN, headColor: GREENTX, bulletCode: "2713", gap: 9 });
  card(s, 6.8, 1.95, 5.9, 4.55, "What it does NOT measure", [
    "Deterministic generated C",
    "Static memory planning",
    "Readable, auditable source",
    "Safety-oriented build integration",
    "Generated-code quality overall",
  ], { fill: RUST, line: RUSTLN, headColor: RUSTTX, bulletCode: "2715", gap: 9 });
  s.addText("Use the placement as credibility — then make clear that executing the unit tests is one dimension of backend quality, not the whole story.", {
    x: 0.6, y: 6.65, w: 12.1, h: 0.5, fontFace: SANS, fontSize: 13, italic: true, color: MUTED, margin: 0 });
  footer(s);
  s.addNotes("Source: onnx.ai/backend-scoreboard. Mention the placement briefly as credibility, but don't make it the main claim — it measures backend unit-test execution, not auditability or deterministic AOT constraints.");
}

/* SLIDE — takeaways */
{
  const s = pres.addSlide();
  s.background = { color: INK };
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 0.52, w: 0.22, h: 0.22, rectRadius: 0.05, fill: { color: TEAL } });
  s.addText("FOR THE ONNX COMMUNITY", { x: 0.92, y: 0.47, w: 9, h: 0.33, fontFace: SANS, fontSize: 12, bold: true, color: TEAL, charSpacing: 2, margin: 0, valign: "middle" });
  s.addText("Takeaways", { x: 0.6, y: 0.9, w: 12, h: 0.85, fontFace: SERIF, fontSize: 32, bold: true, color: LIGHT, margin: 0, valign: "middle" });
  const cards = [
    ["Boundedness", "An AOT-friendly ONNX profile should state what must be bounded or static for deterministic codegen."],
    ["Shared inference", "Reliable, reusable shape/type inference for dynamic models — extensible to external domains like ORT contrib."],
    ["Persisted facts", "Inferred shapes/types should be storable in the model; a small DSL could drive C++ & Python, numeric + symbolic."],
    ["Accuracy contracts", "Numerical accuracy requirements should be explicit in specs and official tests."],
    ["Code quality", "Determinism, readability and auditability are backend features — not cosmetics."],
    ["Portable test artifacts", "ORT-exported, backend-test-like artifacts let any backend share a broader corpus."],
  ];
  const bw = 3.95, gap = 0.27, bh = 1.55, x0 = 0.6, y0 = 1.95;
  cards.forEach((c, i) => {
    const col = i % 3, row = Math.floor(i / 3);
    const x = x0 + col * (bw + gap), y = y0 + row * (bh + 0.3);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: bw, h: bh, rectRadius: 0.08, fill: { color: "13355A" }, line: { color: "1E4A78", width: 1 } });
    s.addText(c[0], { x: x + 0.25, y: y + 0.18, w: bw - 0.5, h: 0.4, fontFace: SANS, fontSize: 15, bold: true, color: AMBER, margin: 0 });
    s.addText(c[1], { x: x + 0.25, y: y + 0.6, w: bw - 0.5, h: 0.85, fontFace: SANS, fontSize: 11.5, color: "CFE0EC", margin: 0, valign: "top" });
  });
  s.addText("ONNX is excellent interchange — deterministic AOT compilation just needs a few constraints made explicit.", {
    x: 0.6, y: 6.5, w: 12.1, h: 0.5, fontFace: SANS, fontSize: 14, italic: true, color: "9FC0D6", margin: 0 });
  s.addNotes("End constructively. AOT-friendly ONNX usage needs explicit bounds; reusable, extensible, persistable inference; explicit accuracy requirements; and recognition that generated-code quality is a backend feature.");
}

/* =================================================================== */
/* SECTION 06 — DISCUSSION                                             */
/* =================================================================== */
{
  const s = pres.addSlide();
  s.background = { color: DEEP };
  s.addText("Discussion", { x: 0.85, y: 0.7, w: 11.6, h: 1.0, fontFace: SERIF, fontSize: 44, bold: true, color: LIGHT, margin: 0 });
  s.addText("What would an AOT-friendly ONNX profile need to specify?", {
    x: 0.87, y: 1.7, w: 11.6, h: 0.6, fontFace: SANS, fontSize: 18, color: AMBER, margin: 0 });
  const qs = [
    "Should ONNX have a stronger shared shape/type inference artifact for dynamic models?",
    "How should external operator domains register inference rules?",
    "Could custom operators carry portable shape/type rules inside ONNX?",
    "Which operators most need clearer numerical contracts?",
    "How should scoreboards represent deterministic codegen & generated-code quality?",
  ];
  s.addText(qs.map((q) => ({ text: q, options: { bullet: { code: "2192", indent: 22 }, color: "DCE8F0", breakLine: true, paraSpaceAfter: 13 } })), {
    x: 0.9, y: 2.6, w: 11.5, h: 3.4, fontFace: SANS, fontSize: 16.5, margin: 0, valign: "top" });
  s.addShape(pres.shapes.LINE, { x: 0.9, y: 6.25, w: 3.5, h: 0, line: { color: TEAL, width: 2 } });
  s.addText("Thank you  ·  emx-onnx-cgen is open source", { x: 0.9, y: 6.45, w: 11, h: 0.5, fontFace: SANS, fontSize: 15, bold: true, color: LIGHT, margin: 0 });
  s.addNotes("Open the floor. Lead with the profile question, then let the room pick threads: shared inference artifact, domain registration, portable custom-op rules, numerical contracts, and how scoreboards could reflect deterministic, auditable code generation rather than only unit-test execution.");
}

pres.writeFile({ fileName: "onnx-community-day-aot-onnx-to-c-v3.pptx" }).then((f) => console.log("WROTE", f, "slides:", pageNo + "(+dividers/title)"));
