const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5
pres.author = "emmtrix Technologies";
pres.title = "Lessons learned from building an AOT ONNX-to-C compiler (master deck)";

const W = 13.333, H = 7.5;

// ---- Palette (emmtrix corporate template) ----
const INK   = "0E3F66"; // dark brand blue (panels, dividers)
const DEEP  = "0A2E4A"; // darker brand blue
const TEAL  = "126AAA"; // primary brand blue (accent)
const TEAL2 = "0E537F"; // darker blue for headers
const AMBER = "F18926"; // brand orange (highlight)
const SLATE = "454545"; // body text
const MUTED = "8A8A8A";
const LIGHT = "FFFFFF";
const CARD  = "F1F5F9";
const CARDLN= "D9E3EA";
const CODEBG = process.env.CODEBG || "FBF3E8"; // warm cream code panel (v3a, overridable)
const CODEFG = process.env.CODEFG || "13344B"; // dark code text
const CODELN = process.env.CODELN || "F0E2CC"; // code panel border
const CODEAC = process.env.CODEAC || AMBER;    // code accent bar (brand orange)
const CODECAP= process.env.CODECAP || "C06A18"; // code caption text (orange caution)
const GREEN = "EEF6E8", GREENLN = "D7E9C9", GREENTX = "4E8A2E"; // brand green
const RUST  = "FBECEC", RUSTLN  = "EBD2D2", RUSTTX  = "B23A3A"; // problem red
const SAND  = "FDF1E4", SANDLN  = "F6DCC0", SANDTX  = "C06A18"; // orange caution

const SERIF = "Rajdhani Semibold"; // headings (brand title font)
const SANS  = "Arial";             // body (brand body font)
const MONO  = "Consolas";

const LOGO    = "assets/emmtrix-logo.png";        // 468x125
const LOGO_W  = "assets/emmtrix-logo-white.png";  // 753x201
const TITLEBG = "assets/title-bg.jpg";

let pageNo = 0;
const sh = (o = {}) => Object.assign({ type: "outer", color: "0A2E4A", blur: 9, offset: 3, angle: 90, opacity: 0.14 }, o);

// corporate footer: thin brand-blue separator, logo bottom-left, page number bottom-right
function footer(slide, dark) {
  pageNo += 1;
  slide.addShape(pres.shapes.LINE, { x: 0.5, y: 6.94, w: 12.33, h: 0, line: { color: dark ? "2C6FA0" : TEAL, width: 1.25 } });
  slide.addImage({ path: dark ? LOGO_W : LOGO, x: 0.5, y: 7.04, w: 1.2, h: 0.32 });
  slide.addText("ONNX Community Day  ·  emx-onnx-cgen", {
    x: 1.95, y: 7.05, w: 8, h: 0.3, fontFace: SANS, fontSize: 10.5, color: dark ? "AFC6D8" : MUTED, align: "left", valign: "middle", margin: 0 });
  slide.addText(String(pageNo), {
    x: W - 1.1, y: 7.05, w: 0.6, h: 0.3, fontFace: SANS, fontSize: 12, bold: true, color: dark ? LIGHT : TEAL2, align: "right", valign: "middle", margin: 0 });
}

// Kicker (small uppercase label above the title) intentionally disabled:
// slides use a single heading only. Kept as a no-op so existing calls still work.
function kicker(slide, text) { void slide; void text; }

function title(slide, text, y = 0.58) {
  slide.addText(text, { x: 0.6, y, w: 12.1, h: 0.9, fontFace: SERIF, fontSize: 30, bold: true, color: TEAL, margin: 0, valign: "middle" });
}

// Single-element code panel: rounded rectangle with caption + monospace code in ONE text frame
function codeBox(slide, code, opt = {}) {
  const o = Object.assign({ x: 0.6, y: 2.2, w: 6.0, h: 2.4, fontSize: 15 }, opt);
  const cf = Math.min(o.fontSize, 13); // cap monospace so long lines never wrap
  const runs = [];
  if (o.caption) runs.push({ text: o.caption, options: { fontFace: SANS, fontSize: 12, bold: true, color: o.capColor || CODECAP, breakLine: true, paraSpaceAfter: 7 } });
  runs.push({ text: code, options: { fontFace: MONO, fontSize: cf, color: o.fg || CODEFG } });
  slide.addText(runs, {
    shape: pres.shapes.ROUNDED_RECTANGLE,
    x: o.x, y: o.y, w: o.w, h: o.h, rectRadius: 0.06,
    fill: { color: o.bg || CODEBG }, line: { color: o.ln || CODELN, width: 1 }, shadow: sh({ opacity: 0.08 }),
    align: "left", valign: "top", margin: [10, 10, 10, 14], lineSpacingMultiple: 1.05,
  });
}

// Single-element card: rounded rectangle with header + bullets in ONE text frame
function card(slide, x, y, w, h, header, bullets, opt = {}) {
  const runs = [];
  if (header) {
    runs.push({ text: header, options: { fontFace: SANS, fontSize: opt.headSize || 15, bold: true, color: opt.headColor || TEAL2, breakLine: true, paraSpaceAfter: 9 } });
  }
  if (bullets && bullets.length) {
    const code = opt.bulletCode || "2022";
    bullets.forEach((b) => runs.push({ text: b, options: { bullet: { code, indent: 14 }, fontFace: SANS, fontSize: opt.fontSize || 12.5, color: opt.textColor || SLATE, breakLine: true, paraSpaceAfter: opt.gap != null ? opt.gap : 6 } }));
  }
  slide.addText(runs, {
    shape: pres.shapes.ROUNDED_RECTANGLE,
    x, y, w, h, rectRadius: 0.07,
    fill: { color: opt.fill || CARD }, line: { color: opt.line || CARDLN, width: 1 }, shadow: sh({ opacity: 0.10 }),
    align: "left", valign: "top", margin: [14, 12, 10, 18],
  });
}

// Single-element info panel: rounded rectangle with a header line + body in ONE text frame.
// `body` may be a string or an array of pptxgenjs text-run objects.
function panel(slide, x, y, w, h, header, body, opt = {}) {
  const runs = [];
  if (header) runs.push({ text: header, options: { fontFace: SANS, fontSize: opt.headSize || 16.5, bold: true, color: opt.headColor || AMBER, breakLine: true, paraSpaceAfter: opt.headGap != null ? opt.headGap : 8 } });
  if (typeof body === "string") {
    runs.push({ text: body, options: { fontFace: SANS, fontSize: opt.fontSize || 16, color: opt.textColor || "DCE8F0" } });
  } else if (Array.isArray(body)) {
    body.forEach((r) => runs.push(r));
  }
  slide.addText(runs, {
    shape: pres.shapes.ROUNDED_RECTANGLE,
    x, y, w, h, rectRadius: opt.rectRadius || 0.07,
    fill: { color: opt.fill || INK },
    line: opt.line ? { color: opt.line, width: 1 } : { type: "none" },
    shadow: opt.shadow === false ? undefined : sh({ opacity: opt.shadowOp != null ? opt.shadowOp : 0.10 }),
    align: "left", valign: opt.valign || "top", margin: opt.margin || [14, 14, 12, 18],
  });
}

function badge(slide, x, y, label, d = 0.62) {
  slide.addText(label, { shape: pres.shapes.OVAL, x, y, w: d, h: d, fill: { color: TEAL }, shadow: sh({ opacity: 0.18 }), fontFace: SERIF, fontSize: d > 0.55 ? 20 : 15, bold: true, color: LIGHT, align: "center", valign: "middle", margin: 0 });
}

function lessonHead(slide, num, text) {
  badge(slide, 0.6, 0.56, String(num));
  slide.addText(text, { x: 1.42, y: 0.56, w: 11.3, h: 0.66, fontFace: SERIF, fontSize: 29, bold: true, color: TEAL, margin: 0, valign: "middle" });
}

function sectionDivider(num, secTitle, subtitle, items) {
  const s = pres.addSlide();
  s.background = { color: INK };
  s.addText(String(num).padStart(2, "0"), { x: 0.85, y: 0.95, w: 3.2, h: 1.6, fontFace: SERIF, fontSize: 100, bold: true, color: AMBER, margin: 0 });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.9, y: 2.95, w: 0.7, h: 0.12, rectRadius: 0.06, fill: { color: AMBER } });
  s.addText(secTitle, { x: 0.85, y: 3.15, w: 11.5, h: 1.0, fontFace: SERIF, fontSize: 42, bold: true, color: LIGHT, margin: 0 });
  if (subtitle) s.addText(subtitle, { x: 0.87, y: 4.22, w: 11, h: 0.6, fontFace: SANS, fontSize: 17, color: "AFC6D8", margin: 0 });
  if (items && items.length) {
    s.addText(items.map((t) => ({ text: t, options: { bullet: { code: "2192", indent: 26 }, color: "D6E4EF", breakLine: true, paraSpaceAfter: 8 } })), {
      x: 0.9, y: 5.0, w: 11.4, h: 1.5, fontFace: SANS, fontSize: 16.5, margin: 0, valign: "top" });
  }
  s.addImage({ path: LOGO_W, x: 0.5, y: 6.78, w: 1.5, h: 0.40 });
  return s;
}

/* =================================================================== */
/* SLIDE 1 — TITLE                                                     */
/* =================================================================== */
{
  const s = pres.addSlide();
  s.background = { color: INK };
  // full-bleed brand photo + dark overlay for legibility
  s.addImage({ path: TITLEBG, x: 0, y: 0, w: W, h: H, sizing: { type: "cover", w: W, h: H } });
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: W, h: H, fill: { color: DEEP, transparency: 32 } });
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 2.05, w: W, h: 3.05, fill: { color: DEEP, transparency: 18 } });
  s.addImage({ path: LOGO_W, x: 0.85, y: 0.85, w: 2.3, h: 0.61 });

  s.addText("LESSONS LEARNED  ·  MASTER DECK", { x: 0.9, y: 2.2, w: 11, h: 0.4, fontFace: SANS, fontSize: 16.5, bold: true, color: AMBER, charSpacing: 3, margin: 0, valign: "middle" });
  s.addText("Building an AOT\nONNX-to-C Compiler", { x: 0.85, y: 2.65, w: 11.6, h: 2.0, fontFace: SERIF, fontSize: 54, bold: true, color: LIGHT, margin: 0, lineSpacingMultiple: 0.98 });
  s.addText("emx-onnx-cgen — deterministic, portable C for embedded and resource-constrained systems", {
    x: 0.87, y: 4.75, w: 11.2, h: 0.6, fontFace: SANS, fontSize: 18, color: "E6EEF4", margin: 0 });
  s.addShape(pres.shapes.LINE, { x: 0.9, y: 6.45, w: 3.4, h: 0, line: { color: AMBER, width: 2 } });
  s.addText("ONNX Community Day  ·  emmtrix Technologies", { x: 0.9, y: 6.6, w: 11, h: 0.4, fontFace: SANS, fontSize: 15.5, bold: true, color: LIGHT, margin: 0 });
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
    s.addText([
      { text: c[0], options: { fontFace: SERIF, fontSize: 32, bold: true, color: TEAL, breakLine: true, paraSpaceAfter: 8 } },
      { text: c[1], options: { fontFace: SANS, fontSize: 17, bold: true, color: INK, breakLine: true, paraSpaceAfter: 5 } },
      { text: c[2], options: { fontFace: SANS, fontSize: 14, color: SLATE } },
    ], { shape: pres.shapes.ROUNDED_RECTANGLE, x, y, w: bw, h: bh, rectRadius: 0.08, fill: { color: CARD }, line: { color: CARDLN, width: 1 }, shadow: sh({ opacity: 0.10 }), valign: "top", margin: [14, 14, 12, 20] });
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
    x: 0.6, y: 6.65, w: 12.1, h: 0.5, fontFace: SANS, fontSize: 15.5, italic: true, color: MUTED, margin: 0 });
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
    s.addText(r[0], { x: 0.85, y: y + 0.1, w: 2.7, h: 1.05, fontFace: MONO, fontSize: 17, bold: true, color: INK, margin: 0, valign: "middle" });
    s.addText([{ text: "Strength  ", options: { bold: true, color: GREENTX } }, { text: r[1], options: { color: SLATE } }], { x: 3.6, y: y + 0.18, w: 4.4, h: 0.9, fontFace: SANS, fontSize: 15, margin: 0, valign: "middle" });
    s.addText([{ text: "Gap for us  ", options: { bold: true, color: r[3] } }, { text: r[2], options: { color: SLATE } }], { x: 8.1, y: y + 0.18, w: 4.4, h: 0.9, fontFace: SANS, fontSize: 15, margin: 0, valign: "middle" });
    y += 1.42;
  });
  s.addText("We wanted a standalone, deterministic, auditable ONNX-to-generic-C flow — and no existing tool matched it at the time.", {
    x: 0.6, y: 6.5, w: 12.1, h: 0.5, fontFace: SANS, fontSize: 15.5, italic: true, color: MUTED, margin: 0 });
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
           const float x[N][C],
           float       y[N][C]);`,
    { x: 6.4, y: 1.95, w: 6.3, h: 1.6, fontSize: 16.5, caption: "VLA parameters — known rank, runtime extents" });
  panel(s, 6.4, 3.75, 6.3, 2.95, "Representation is not semantics — what VLAs do NOT solve", [
    { text: "Static memory bounds for temporaries and buffers", options: { bullet: { code: "2022" }, fontFace: SANS, fontSize: 15, color: SLATE, breakLine: true, paraSpaceAfter: 6 } },
    { text: "Dynamic rank", options: { bullet: { code: "2022" }, fontFace: SANS, fontSize: 15, color: SLATE, breakLine: true, paraSpaceAfter: 6 } },
    { text: "Unbounded sequence length and string size", options: { bullet: { code: "2022" }, fontFace: SANS, fontSize: 15, color: SLATE, breakLine: true, paraSpaceAfter: 6 } },
    { text: "Targets that lack VLA support", options: { bullet: { code: "2022" }, fontFace: SANS, fontSize: 15, color: SLATE } },
  ], { fill: SAND, line: SANDLN, headColor: SANDTX, headSize: 16, shadow: false });
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
   const float *a,
   const float *b,
   float *out) {
  for (int n=0; n<N; ++n)
    for (int c=0; c<C; ++c) {
      int idx = n*C + c;
      out[idx] = a[idx] + b[idx];
    }
}`,
    { x: 0.6, y: 2.0, w: 5.9, h: 3.5, fontSize: 15, caption: "Flat 1-D buffers — structure is only convention" });
  codeBox(s,
`void add_vla(int N, int C,
   const float a[N][C],
   const float b[N][C],
   float out[N][C]) {
  for (int n=0; n<N; ++n)
    for (int c=0; c<C; ++c) {
      out[n][c] = a[n][c] + b[n][c];
    }
}`,
    { x: 6.8, y: 2.0, w: 5.9, h: 3.5, fontSize: 15, caption: "N-D VLA types — rank & extent live in the type" });
  s.addText([
    { text: "The subtle point:  ", options: { bold: true, color: AMBER } },
    { text: "accessing beyond a declared array object is undefined behaviour — even when the next tensor row is contiguous in memory. Keeping tensor structure in the type improves readability, reviewability and downstream analysis.", options: { color: "DCE8F0" } },
  ], { shape: pres.shapes.ROUNDED_RECTANGLE, x: 0.6, y: 5.65, w: 12.1, h: 1.15, rectRadius: 0.07, fill: { color: INK }, fontFace: SANS, fontSize: 16, margin: [8, 22, 8, 26], valign: "middle" });
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
    s.addText([
      { text: st[1], options: { fontFace: SANS, fontSize: 17, bold: true, color: INK, breakLine: true, paraSpaceAfter: 6 } },
      { text: st[2], options: { fontFace: SANS, fontSize: 15.5, color: SLATE } },
    ], { shape: pres.shapes.ROUNDED_RECTANGLE, x, y: y0, w: bw, h: bh, rectRadius: 0.08, fill: { color: CARD }, line: { color: CARDLN, width: 1 }, shadow: sh({ opacity: 0.10 }), valign: "top", margin: [72, 14, 12, 22] });
    badge(s, x + 0.3, y0 + 0.3, st[0]);
    if (i < steps.length - 1) s.addText("→", { x: x + bw - 0.02, y: y0, w: gap + 0.04, h: bh, fontFace: SANS, fontSize: 22, color: TEAL, align: "center", valign: "middle", margin: 0 });
  });
  s.addText("A high-coverage ONNX compiler cannot be a pile of special cases — it needs registries, structured lowering, stable IR objects and a verification strategy.", {
    shape: pres.shapes.ROUNDED_RECTANGLE, x: 0.6, y: 5.6, w: 12.1, h: 1.1, rectRadius: 0.07, fill: { color: INK }, fontFace: SANS, fontSize: 17, italic: true, color: "DCE8F0", margin: [8, 22, 8, 26], valign: "middle" });
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
  s.addText("Intentionally small public API", { x: 7.0, y: 2.0, w: 5.7, h: 0.4, fontFace: SANS, fontSize: 17, bold: true, color: TEAL2, margin: 0 });
  codeBox(s,
`_Bool model_load(const char *path);

void  model(/* ...inputs... */,
            /* ...outputs... */);

/* tensor IO -> C array parameters
   dynamic dims -> C99 VLAs
   weights      -> optional model.bin */`,
    { x: 7.0, y: 2.45, w: 5.7, h: 2.55, fontSize: 15.5 });
  panel(s, 7.0, 5.25, 5.7, 1.25, "Frontend of the emmtrix embedded-AI flow", [
    { text: "ONNX → clean C → Vectorizer → target architecture", options: { fontFace: MONO, fontSize: 15, color: TEAL2 } },
  ], { fill: CARD, line: CARDLN, headColor: INK, headSize: 15, shadow: false });
  footer(s);
  s.addNotes("Just enough context before the lessons. AOT ONNX-to-C, portable and deterministic, for embedded targets, avoiding dynamic memory and external runtimes. The generated API is deliberately tiny: a load function and a model function.");
}

/* SLIDE — What it is: goals + a real C output example */
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  kicker(s, "In one slide");
  title(s, "What emx-onnx-cgen is — goals & generated C");
  card(s, 0.6, 1.95, 5.65, 4.55, "Goals", [
    "Ahead-of-time ONNX → portable, deterministic C",
    "Correctness-first: outputs comparable to ONNX Runtime",
    "Static, compile-time memory — no malloc/free, no heap",
    "No OS, no external runtime, no hidden dispatch",
    "Readable, auditable C for review & certification",
    "Pass-based pipeline: import → normalize → optimize → lower → emit",
  ], { gap: 9, fontSize: 14.5 });
  s.addText("Tensors map to typed N-D C arrays; each ONNX node becomes one small, explicit loop nest.", {
    x: 0.6, y: 6.6, w: 5.65, h: 0.55, fontFace: SANS, fontSize: 12.5, italic: true, color: MUTED, margin: 0, valign: "top" });
  codeBox(s,
`/* Generated by emx-onnx-cgen — Mul -> Add -> Relu */
#include <stdint.h>
#ifndef idx_t
#define idx_t int32_t
#endif

static inline float f_mul(float a,float b){return a*b;}
static inline float f_add(float a,float b){return a+b;}
static inline float f_relu(float a){return a>0.f?a:0.f;}

void node0_mul(const float in0[2][3],
               const float in1[2][3], float out[2][3]){
  for (idx_t i=0;i<2;++i) for (idx_t j=0;j<3;++j)
    out[i][j] = f_mul(in0[i][j], in1[i][j]);
}
/* node1_add, node2_relu: same shape, one op each */

_Bool model_load(const char *path){(void)path;return 1;}

void model(const float a[2][3],
           const float b[2][3],
           const float c[2][3],
           float out[2][3]){
  float t0[2][3], t1[2][3];     /* stack temporaries */
  node0_mul(a, b, t0);
  node1_add(t0, c, t1);
  node2_relu(t1, out);          /* no malloc/runtime/OS */
}`,
    { x: 6.45, y: 1.95, w: 6.27, h: 5.2, fontSize: 12, caption: "Generated C — excerpt from a golden reference (tests/golden/mul_add_relu_model.c)" });
  footer(s);
  s.addNotes("One slide that grounds the talk: what the project is, its goals, and what the output actually looks like. The C example is a faithful excerpt of an in-repo golden reference (mul_add_relu_model.c): a Mul -> Add -> Relu graph. Note the shape: every input/output is a typed N-D C array, every ONNX node becomes one small explicit loop nest with a scalar reference op, temporaries live on the stack, and the public API is just model_load() + model(). No malloc, no runtime, no OS, no hidden dispatch — that is the entire contract.");
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
    s.addText([
      { text: f[0], options: { fontFace: MONO, fontSize: 17, bold: true, color: INK, breakLine: true, paraSpaceAfter: 4 } },
      { text: f[1], options: { fontFace: SANS, fontSize: 14, color: SLATE } },
    ], { shape: pres.shapes.ROUNDED_RECTANGLE, x: 0.6, y, w: 6.6, h: 1.02, rectRadius: 0.07, fill: { color: f[2] ? GREEN : CARD }, line: { color: f[2] ? GREENLN : CARDLN, width: 1 }, shadow: sh({ opacity: 0.08 }), valign: "middle", margin: [6, 12, 6, 16] });
    y += 1.14;
  });
  card(s, 7.45, 2.05, 5.25, 4.5, "Code-generation principles", [
    "Simple canonical loop forms",
    "Linear array accesses, explicit dims & strides",
    "No hidden pointer aliasing",
    "No dynamic dispatch, no recursion",
    "No dynamic memory allocation",
    "Stable symbol names, deterministic ordering",
  ], { fill: INK, headColor: AMBER, textColor: "DCE8F0", bulletCode: "2713", gap: 9, fontSize: 15.5 });
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
    { x: 0.6, y: 2.0, w: 6.0, h: 1.95, fontSize: 15, caption: "Tensors" });
  codeBox(s,
`/* sequence input  */
const T name[EMX_SEQUENCE_MAX_LEN][elem_shape...];
idx_t  name__count;

/* sequence output */
T      name[EMX_SEQUENCE_MAX_LEN][elem_shape...];
idx_t *name__count;`,
    { x: 6.8, y: 2.0, w: 5.9, h: 1.95, fontSize: 14, caption: "Sequences — fixed capacity + count" });
  card(s, 0.6, 4.15, 6.0, 2.55, "Bounds the compiler must invent", [
    "EMX_SEQUENCE_MAX_LEN — default 32, #ifndef-overridable",
    "EMX_STRING_MAX_LEN — default 256, #ifndef-overridable",
    "Ragged inputs: --sequence-element-shape declares maxima",
    "Per-item dims: idx_t name__dim_<axis>[EMX_SEQUENCE_MAX_LEN]",
  ], { fontSize: 14, gap: 7 });
  card(s, 6.8, 4.15, 5.9, 2.55, "Reduced-precision dtypes", [
    "float16 → _Float16 (where supported)",
    "bfloat16 → __bf16 (compiler/target-specific)",
    "int2 / int4 → _BitInt(N), unsigned _BitInt(N) (C23)",
    "FP8 / FP4 → uint8 storage + conversion helpers",
  ], { fill: GREEN, line: GREENLN, headColor: GREENTX, fontSize: 14, gap: 7 });
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
    s.addText(t, { x: cx, y: fy, w: wdt, h: 0.62, fontFace: MONO, fontSize: 15, color: LIGHT, align: "center", valign: "middle", margin: 0 });
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

/* SLIDE — How emmtrix uses emx-onnx-cgen: C as IR + optimization */
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  title(s, "How we use emx-onnx-cgen at emmtrix: C as IR");
  // top flow strip (single-element chips)
  const flow = ["ONNX model", "emx-onnx-cgen", "C = our IR", "emmtrix optimizer", "optimized C → target"];
  let cx = 0.7; const fy = 1.62;
  flow.forEach((t, i) => {
    const wdt = 0.4 + t.length * 0.118;
    const fill = i === 2 ? AMBER : (i === 1 || i === 3 ? TEAL2 : INK);
    const fg = i === 2 ? INK : LIGHT;
    s.addText(t, { shape: pres.shapes.ROUNDED_RECTANGLE, x: cx, y: fy, w: wdt, h: 0.62, rectRadius: 0.1, fill: { color: fill }, fontFace: MONO, fontSize: 14.5, bold: i === 2, color: fg, align: "center", valign: "middle", margin: 0 });
    cx += wdt;
    if (i < flow.length - 1) { s.addText("→", { x: cx, y: fy, w: 0.42, h: 0.62, fontFace: SANS, fontSize: 18, color: TEAL, align: "center", valign: "middle", margin: 0 }); cx += 0.42; }
  });
  // left: two stacked text blocks · right: a wider code example
  card(s, 0.6, 2.6, 5.0, 1.7, "C as our IR — not MLIR", [
    "Staged lowering like MLIR — but the IR is plain C",
    "Standard, reviewable, toolchain-agnostic",
    "Correctness first; performance via later passes",
  ], { gap: 8, fontSize: 13.5 });
  card(s, 0.6, 4.45, 5.0, 2.2, "Then we optimize the generated C", [
    "Node fusion",
    "Memory reduction",
    "Vectorization (SIMD)",
    "Memory layout optimization",
    "Buffer reuse",
    "Weight offloading / DMA",
  ], { fill: GREEN, line: GREENLN, headColor: GREENTX, headSize: 14, bulletCode: "2713", gap: 5, fontSize: 13.5 });
  codeBox(s,
`/* matmul -> RVV; loops over N,K · 16 = vector length */
for (i = 0; i < N; ++i) {
  float sum[16];
  __riscv_lib_vse32_v_f32m1(&sum[0],
      __riscv_vfmv_v_f_f32m1(0.0f, 16), 16);

  for (k = 0; k < K; ++k)
    __riscv_lib_vse32_v_f32m1(&sum[0],
      __riscv_vfmacc_vv_f32m1(
        __riscv_lib_vle32_v_f32m1(&sum[0], 16),
        __riscv_lib_vle32_v_f32m1(&a[k][0], 16),
        __riscv_vfmv_v_f_f32m1(b[i][k], 16), 16), 16);

  __riscv_lib_vse32_v_f32m1(&c[i][0],
      __riscv_lib_vle32_v_f32m1(&sum[0], 16), 16);
}`,
    { x: 5.9, y: 2.6, w: 6.8, h: 4.05, fontSize: 12.5, caption: "Example — RISC-V RVV matmul (emmtrix Edge AI Compiler)" });
  footer(s);
  s.addNotes("This is how emx-onnx-cgen fits the emmtrix toolchain. The compiler emits deterministic, analyzable C, and we deliberately use that C as our intermediate representation instead of MLIR: it is standardized, reviewable, and accepted by heterogeneous and safety-critical toolchains, with no proprietary IR lock-in. Correctness comes first in the generated C; performance is added by later source-to-source passes over that same C — node/kernel fusion, intermediate-memory reduction and buffer reuse, vectorization onto SIMD units, target-specific memory-layout optimization, and offloading or DMA-streaming of large weights. The example is emmtrix Edge AI Compiler output for a matmul targeting RISC-V with the RVV vector extension: an outer loop over the rows (N) and an inner reduction loop over K, with the free dimension vectorized — 16 is the RVV vector length, not the problem size, so the same code scales to realistic tensors. Element-wise ops like Relu fuse into the vector store. A public Compiler Explorer version is planned. Clean generated C is what makes those passes tractable. Source: github.com/emmtrix/emx-onnx-cgen issue #723.");
}

/* SLIDE — Supporting dynamic models with VLAs */
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  title(s, "Explicit typed arrays — even for dynamic models");
  s.addText([
    { text: "Same ReduceSum on a 5-D tensor ", options: { color: SLATE } },
    { text: "[N0…N4]", options: { fontFace: MONO, color: INK } },
    { text: " — the flat version collapses the leading dims into one count (N0·N1·N2·N3); the typed array keeps every dimension.", options: { color: SLATE } },
  ], { x: 0.6, y: 1.46, w: 12.1, h: 0.45, fontFace: SANS, fontSize: 14.5, margin: 0, valign: "middle" });
  codeBox(s,
`// Input:  [N0, N1, N2, N3, N4]
// Output: [N0, N1, N2, N3]
// ReduceSum(axis = 4)

// leading dims collapse into one count:
//   outer = N0 * N1 * N2 * N3
void reduce_sum(int outer, int N4,
                const float *input, float *output) {
  for (int o = 0; o < outer; ++o) {
    float sum = 0;
    for (int i = 0; i < N4; ++i)
      sum += input[o * N4 + i];
    output[o] = sum;
  }
}`,
    { x: 0.6, y: 2.0, w: 6.0, h: 3.78, fontSize: 11, caption: "✗  Flat pointer — shape collapsed to a count", bg: RUST, ln: RUSTLN, capColor: RUSTTX });
  codeBox(s,
`// Input:  [N0, N1, N2, N3, N4]   (dynamic)
// Output: [N0, N1, N2, N3]
// ReduceSum(axis = 4)

void reduce_sum(int N0, int N1, int N2, int N3, int N4,
                const float input[N0][N1][N2][N3][N4],
                float output[N0][N1][N2][N3]) {
  for (int i0 = 0; i0 < N0; ++i0)
  for (int i1 = 0; i1 < N1; ++i1)
  for (int i2 = 0; i2 < N2; ++i2)
  for (int i3 = 0; i3 < N3; ++i3) {
    float sum = 0;
    for (int i4 = 0; i4 < N4; ++i4)
      sum += input[i0][i1][i2][i3][i4];
    output[i0][i1][i2][i3] = sum;
  }
}`,
    { x: 6.7, y: 2.0, w: 6.0, h: 3.78, fontSize: 11, caption: "✓  Explicit typed array — every dimension kept", bg: GREEN, ln: GREENLN, capColor: GREENTX });
  panel(s, 0.6, 5.9, 12.1, 0.92, "Why explicit — even at rank 5 & dynamic shapes", [
    { text: "Flat only gets a single count ", options: { fontFace: SANS, fontSize: 13, color: "DCE8F0" } },
    { text: "outer = N0·N1·N2·N3", options: { fontFace: MONO, fontSize: 12, color: "CFE0EC" } },
    { text: " — the per-axis structure is gone. The typed array ", options: { fontFace: SANS, fontSize: 13, color: "DCE8F0" } },
    { text: "input[N0][N1][N2][N3][N4]", options: { fontFace: MONO, fontSize: 12, color: AMBER } },
    { text: " keeps every dimension named and bounded; the compiler does the address math. VLAs keep this even when N0…N4 are runtime dims.", options: { fontFace: SANS, fontSize: 13, color: "DCE8F0" } },
  ], { fill: INK, headColor: AMBER, headSize: 14, valign: "middle" });
  footer(s);
  s.addNotes("Explicit, typed arrays — shown on a 5-D ReduceSum(axis=4): [N0,N1,N2,N3,N4] -> [N0,N1,N2,N3]. The rank-5 case makes the difference obvious. Left, the flat float* version: because a bare pointer carries no shape, the leading dimensions are collapsed into a single count outer = N0*N1*N2*N3 that the caller passes in, with one outer loop and input[o*N4 + i] — the per-axis structure is gone. Right, the explicit typed array: input[N0][N1][N2][N3][N4] and output[N0][N1][N2][N3] as C99 variable-length array types — the shape lives in the type, the compiler derives the strides, and the same loop nest just reads input[i0][i1][i2][i3][i4]. No stride parameters, no hand-written offset arithmetic to get wrong. N0..N4 are runtime (dynamic) dimensions and VLAs keep the array explicit anyway; accessing beyond a declared object is undefined behaviour and therefore analyzable; the disciplined form is easy to auto-vectorize. Keeping arrays explicit for dynamic shapes (onnx2c's weak spot) is exactly why we started the project. Caveat: a VLA carries the runtime extent, not a maximum, so memory capacities remain a separate policy (Lesson 1).");
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
    x: 0.6, y: 1.82, w: 12.1, h: 0.45, fontFace: SANS, fontSize: 17, color: SLATE, margin: 0 });
  const forced = ["Pass-based architecture", "Explicit IR boundaries", "Deterministic codegen", "Precise, testable diagnostics", "Reference-based verification", "Explicit context, no global state"];
  const discouraged = ["Hidden global state", "Oversized “god” modules", "Implicit shape assumptions", "Broad mutation in passes", "Codegen driven by verify-only inputs"];
  panel(s, 0.6, 2.45, 6.0, 4.15, "It forced", forced.map((t) => ({ text: t, options: { bullet: { code: "2713" }, fontFace: SANS, fontSize: 16, color: SLATE, breakLine: true, paraSpaceAfter: 8 } })), { fill: GREEN, line: GREENLN, headColor: GREENTX, headSize: 17 });
  panel(s, 6.85, 2.45, 5.85, 4.15, "It discouraged", discouraged.map((t) => ({ text: t, options: { bullet: { code: "2715" }, fontFace: SANS, fontSize: 16, color: SLATE, breakLine: true, paraSpaceAfter: 8 } })), { fill: RUST, line: RUSTLN, headColor: RUSTTX, headSize: 17 });
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
    s.addText([
      { text: String(i + 1), options: { fontFace: SERIF, fontSize: 22, bold: true, color: TEAL, breakLine: true, paraSpaceAfter: 8 } },
      { text: st[0], options: { fontFace: SANS, fontSize: 17, bold: true, color: INK, breakLine: true, paraSpaceAfter: 5 } },
      { text: st[1], options: { fontFace: SANS, fontSize: 13.5, color: SLATE } },
    ], { shape: pres.shapes.ROUNDED_RECTANGLE, x, y: y0, w: bw, h: bh, rectRadius: 0.08, fill: { color: i % 2 ? CARD : GREEN }, line: { color: CARDLN, width: 1 }, shadow: sh({ opacity: 0.09 }), valign: "top", margin: [12, 10, 10, 14] });
    if (i < steps.length - 1) s.addText("›", { x: x + bw - 0.05, y: y0, w: gap + 0.1, h: bh, fontFace: SANS, fontSize: 20, bold: true, color: TEAL, align: "center", valign: "middle", margin: 0 });
  });
  panel(s, 0.6, 5.15, 12.1, 1.55, "Invariant", "Verification-only inputs must never implicitly change generated code. If verify uses extra shape information, it is testing a different compiler path — compile/verify parity keeps the contract honest.", { fill: INK, headColor: AMBER, headSize: 16.5, textColor: "DCE8F0", fontSize: 16.5 });
  footer(s);
  s.addNotes("Floating outputs compared with ULP thresholds after an absolute epsilon; non-floating exactly. Compile/verify parity is the key invariant.");
}

/* SLIDE — Parity invariant deep */
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  kicker(s, "Invariant");
  title(s, "Compile/verify parity in detail");
  panel(s, 0.6, 1.95, 6.0, 3.0, "✗  Bad pattern", [
    { text: "verify reads representative input_*.pb files", options: { bullet: { code: "2022" }, fontFace: SANS, fontSize: 15.5, color: SLATE, breakLine: true, paraSpaceAfter: 7 } },
    { text: "those values or shapes make codegen succeed", options: { bullet: { code: "2022" }, fontFace: SANS, fontSize: 15.5, color: SLATE, breakLine: true, paraSpaceAfter: 7 } },
    { text: "plain compile would fail or emit different code", options: { bullet: { code: "2022" }, fontFace: SANS, fontSize: 15.5, color: SLATE, breakLine: true, paraSpaceAfter: 7 } },
    { text: "→ test results no longer reflect real compilation", options: { fontFace: SANS, fontSize: 15.5, color: RUSTTX, bold: true } },
  ], { fill: RUST, line: RUSTLN, headColor: RUSTTX, headSize: 17 });
  panel(s, 6.8, 1.95, 5.9, 3.0, "✓  Project principle", [
    { text: "Models needing representative inputs to resolve dynamic shapes must fail clearly…", options: { bullet: { code: "2022" }, fontFace: SANS, fontSize: 15.5, color: SLATE, breakLine: true, paraSpaceAfter: 7 } },
    { text: "…or require an explicit compiler option that compile also accepts", options: { bullet: { code: "2022" }, fontFace: SANS, fontSize: 15.5, color: SLATE, breakLine: true, paraSpaceAfter: 7 } },
    { text: "Export with static shapes, or supply explicit options", options: { bullet: { code: "2022" }, fontFace: SANS, fontSize: 15.5, color: SLATE } },
  ], { fill: GREEN, line: GREENLN, headColor: GREENTX, headSize: 17 });
  panel(s, 0.6, 5.15, 12.1, 1.55, "Why it matters", "Reproducibility · deterministic code generation · an honest compiler contract · meaningful test results.", { fill: INK, headColor: AMBER, headSize: 16.5, textColor: "DCE8F0", fontSize: 16.5 });
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
    { x: 0.6, y: 2.0, w: 6.0, h: 1.4, fontSize: 15.5, caption: "The export principle" });
  card(s, 0.6, 3.6, 6.0, 3.1, "Why it matters", [
    "Official ONNX backend tests are the standards baseline",
    "ORT artifacts apply the same model+test-data pattern broadly",
    "Includes contrib/operator scenarios & runtime-style edge cases",
    "Forces real model/test directory handling, not micrographs",
    "Preserves ORT's intended numerical tolerances",
    "Failures stay reproducible via recorded command lines",
  ], { fontSize: 14, gap: 6 });
  card(s, 6.8, 2.0, 5.9, 2.55, "validation.json fields", [
    "expects_failure / expected_failure_substring",
    "per-output relative_error",
    "per-output absolute_error",
    "per-output sort_output",
  ], { fill: GREEN, line: GREENLN, headColor: GREENTX, bulletCode: "2022", fontSize: 15, gap: 7 });
  panel(s, 6.8, 4.75, 5.9, 1.95, "Proposed: convert to ONNX backend data.json", "Each test becomes self-contained — model, protobuf test data, and comparison tolerances in the expected ONNX backend-test convention. Other backends could reuse ORT-derived tests outside the ORT C++ harness.", { fill: INK, headColor: AMBER, headSize: 16, textColor: "CFE0EC", fontSize: 14 });
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
  "Lesson 4 — operator importance is unknown",
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
    s.addText([
      { text: it[0], options: { fontFace: SANS, fontSize: 17, bold: true, color: TEAL2, breakLine: true, paraSpaceAfter: 8 } },
      { text: it[1], options: { fontFace: SANS, fontSize: 15.5, color: SLATE } },
    ], { shape: pres.shapes.ROUNDED_RECTANGLE, x, y: y0, w: bw, h: bh, rectRadius: 0.08, fill: { color: CARD }, line: { color: CARDLN, width: 1 }, shadow: sh({ opacity: 0.10 }), valign: "top", margin: [16, 14, 12, 18] });
  });
  panel(s, 0.6, 4.7, 12.1, 2.0, "The core AOT lesson", "ONNX is intentionally flexible for interchange and runtime execution. General static memory planning is impossible without additional assumptions. A tensor dimension asks “how large is this axis?”; a sequence asks “how many tensor objects exist, and what shape is each?” — comfortable for runtimes, hard for static C. Embedded C needs concrete bounds, or a clear failure.", { fill: INK, headColor: AMBER, headSize: 17, textColor: "DCE8F0", fontSize: 16.5 });
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
           const float x[N][C],
           float       y[N][C]);

/* temporaries & caller buffers can still
   grow without a compile-time bound      */`,
    { x: 6.8, y: 1.95, w: 5.9, h: 2.6, fontSize: 15, caption: "VLAs carry the extent, not the bound" });
  panel(s, 6.8, 4.75, 5.9, 1.95, "emx-onnx-cgen policy", "Require static shapes where necessary; support VLAs where the shape is representable and acceptable; fail clearly when a required bound is missing. Models that need representative inputs to resolve shapes should be exported static or given explicit options.", { fill: GREEN, line: GREENLN, headColor: GREENTX, headSize: 16.5, textColor: SLATE, fontSize: 14, shadow: false });
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
    { x: 0.6, y: 2.0, w: 6.1, h: 3.6, fontSize: 14, caption: "Fixed-capacity array + count metadata" });
  card(s, 6.9, 2.0, 5.8, 1.95, "Why containers are hard", [
    "A sequence is a container value, not a tensor with one more axis",
    "Needs max length, element dtype, rank and element shape",
    "Without a max, no fixed storage can be reserved",
  ], { fontSize: 14, gap: 6 });
  card(s, 6.9, 4.1, 5.8, 2.6, "Ragged sequences", [
    "Not accepted implicitly",
    "--sequence-element-shape declares rank + per-axis maxima",
    "e.g. sequence=[<=8] or boxes=[<=100,4]",
    "Per-item dims: idx_t name__dim_<axis>[…MAX_LEN]",
    "Only first name__count entries are meaningful",
  ], { fill: SAND, line: SANDLN, headColor: SANDTX, fontSize: 13.5, gap: 5 });
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
    { x: 6.8, y: 1.95, w: 5.9, h: 3.1, fontSize: 14, caption: "Fixed-size '\\0'-terminated C string slots" });
  s.addText("Trade-off: arbitrary ONNX string length is not represented without an explicit bound — but memory stays static and deterministic.", {
    x: 6.8, y: 5.25, w: 5.9, h: 1.4, fontFace: SANS, fontSize: 15, italic: true, color: MUTED, margin: 0, valign: "top" });
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
  panel(s, 6.8, 1.95, 5.9, 4.75, "The duplication problem", [
    { text: "No reliable standalone library infers shapes/types for dynamic models well enough for static code generators.", options: { fontFace: SANS, fontSize: 15.5, color: SLATE, breakLine: true, paraSpaceAfter: 9 } },
    { text: "So importers, MLIR frontends, code generators, verifiers and optimizers each reimplement parts of it…", options: { fontFace: SANS, fontSize: 15.5, color: SLATE, breakLine: true, paraSpaceAfter: 9 } },
    { text: "…and may produce different answers for the same model.", options: { fontFace: SANS, fontSize: 15.5, color: RUSTTX, bold: true, breakLine: true, paraSpaceAfter: 9 } },
    { text: "Shape/type inference should be separable both from code generation and from a fixed operator universe.", options: { fontFace: SANS, fontSize: 15.5, color: SLATE, italic: true } },
  ], { fill: RUST, line: RUSTLN, headColor: RUSTTX, headSize: 17 });
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
    s.addText([
      { text: c[0], options: { fontFace: SANS, fontSize: 17, bold: true, color: TEAL2, breakLine: true, paraSpaceAfter: 7 } },
      { text: c[1], options: { fontFace: SANS, fontSize: 15, color: SLATE } },
    ], { shape: pres.shapes.ROUNDED_RECTANGLE, x, y: y0, w: bw, h: bh, rectRadius: 0.08, fill: { color: CARD }, line: { color: CARDLN, width: 1 }, shadow: sh({ opacity: 0.10 }), valign: "top", margin: [14, 14, 12, 18] });
  });
  panel(s, 0.6, 4.7, 12.1, 2.0, null, [
    { text: "ONE DSL  ", options: { fontFace: SANS, fontSize: 17, bold: true, color: AMBER } },
    { text: "→ shape/type rules written once", options: { fontFace: SANS, fontSize: 17, bold: true, color: LIGHT, breakLine: true, paraSpaceAfter: 8 } },
    { text: "Drives both C++ and Python inference", options: { bullet: { code: "2022" }, fontFace: SANS, fontSize: 15, color: "CFE0EC", breakLine: true, paraSpaceAfter: 6 } },
    { text: "Numeric evaluation for concrete dims; symbolic evaluation (e.g. SymPy) for symbolic dims", options: { bullet: { code: "2022" }, fontFace: SANS, fontSize: 15, color: "CFE0EC", breakLine: true, paraSpaceAfter: 6 } },
    { text: "Custom operators ship their own rule instead of every backend hard-coding it — could even live inside ONNX models", options: { bullet: { code: "2022" }, fontFace: SANS, fontSize: 15, color: "CFE0EC" } },
  ], { fill: INK });
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
  ], { fontSize: 15, gap: 6 });
  card(s, 0.6, 4.95, 6.0, 1.75, "ONNX backend default check", [
    "rtol = 1e-3, atol = 1e-7, overridable per real-model test",
    "abs(actual − expected) ≤ atol + rtol·|expected|",
  ], { fill: CARD, fontSize: 15, gap: 6 });
  panel(s, 6.8, 1.95, 5.9, 4.75, "Why this matters to backend authors", [
    { text: "Validation is an engineering policy unless the standard defines the contract.", options: { fontFace: SANS, fontSize: 16, color: "DCE8F0", breakLine: true, paraSpaceAfter: 10 } },
    { text: "A single fixed rtol/atol is not equally meaningful across dtypes — float16, float32 and float64 have very different spacing between representable values.", options: { fontFace: SANS, fontSize: 16, color: "DCE8F0", breakLine: true, paraSpaceAfter: 10 } },
    { text: "ULP-based measurement asks the real question: how many representable floating-point values apart are these results?", options: { fontFace: SANS, fontSize: 16, color: LIGHT, bold: true } },
  ], { fill: INK, headColor: AMBER, headSize: 17, shadowOp: 0.12 });
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
  s.addTable([header, ...body], { x: 0.6, y: 1.95, w: 12.1, colW: [2.9, 2.8, 6.4], border: { pt: 0.5, color: CARDLN }, align: "left", valign: "middle", rowH: 0.48, margin: [3, 6, 3, 6] });
  s.addText("Across the ecosystem the comparison shape is similar (absolute + relative), but defaults and dtype-awareness vary widely — there is no single shared contract.", {
    x: 0.6, y: 6.15, w: 12.1, h: 0.6, fontFace: SANS, fontSize: 14, italic: true, color: MUTED, margin: 0, valign: "top" });
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
    { x: 0.6, y: 2.0, w: 6.2, h: 3.7, fontSize: 15, caption: "How emx-onnx-cgen compares outputs" });
  card(s, 7.1, 2.0, 5.6, 1.95, "Why ULP, not fixed rtol/atol", [
    "fp16 / fp32 / fp64 spacing differs enormously",
    "ULP = how many representable values apart",
    "Closer to the question that actually matters",
  ], { fill: GREEN, line: GREENLN, headColor: GREENTX, fontSize: 15, gap: 6 });
  panel(s, 7.1, 4.1, 5.6, 1.6, "Consequence in practice", "Some ONNX reference outputs appear computed at very high precision. Under strict ULP checks, this required 64-bit accumulation for selected official tests — exposed as --fp32-accumulation-strategy fp64.", { fill: INK, headColor: AMBER, headSize: 16, textColor: "CFE0EC", fontSize: 14 });
  s.addText("emx ignores absolute diffs up to atol_eps·eps(dtype), then measures ULP distance — exact match required for non-float outputs.", {
    x: 0.6, y: 5.95, w: 6.2, h: 0.7, fontFace: SANS, fontSize: 14, italic: true, color: MUTED, margin: 0, valign: "top" });
  footer(s);
  s.addNotes("emx does not use a classic fixed rtol/atol. It gates on an absolute epsilon scaled by eps(dtype), then measures ULP distance, reporting the max; non-float must match exactly. Because some reference outputs are effectively high precision, strict ULP forced fp64 accumulation on selected tests, exposed via --fp32-accumulation-strategy fp64 and used in several official-test expectation files.");
}

/* SLIDE — accuracy: fixed tolerance vs computation order */
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  kicker(s, "Lesson · accuracy");
  title(s, "Evaluation order & precision are unspecified");
  s.addText([
    { text: "Example — ReduceSum (fp32).  ", options: { bold: true, color: AMBER } },
    { text: "ONNX defines the math, not how it is evaluated — summation ", options: { bold: true, color: TEAL2 } },
    { text: "order", options: { bold: true, color: RUSTTX } },
    { text: " and accumulation ", options: { bold: true, color: TEAL2 } },
    { text: "precision", options: { bold: true, color: GREENTX } },
    { text: " are both free, so equally-correct implementations differ.", options: { color: SLATE } },
  ], { x: 0.6, y: 1.44, w: 12.1, h: 0.55, fontFace: SANS, fontSize: 13.5, margin: 0, valign: "middle" });
  // --- inline diagram helpers ---
  const edge = (x1, y1, x2, y2, color, wdt) => {
    const x = Math.min(x1, x2), y = Math.min(y1, y2), ww = Math.abs(x2 - x1), hh = Math.abs(y2 - y1);
    const leftIsTop = (x1 === x2) ? true : (x1 < x2 ? y1 <= y2 : y2 <= y1);
    s.addShape(pres.shapes.LINE, { x, y, w: ww || 0.001, h: hh || 0.001, line: { color: color || "8AA0B4", width: wdt || 1.7 }, flipV: !leftIsTop });
  };
  const node = (cx, cy, d, fill, label) => {
    const fs = d >= 0.42 ? 16 : (d >= 0.3 ? 13 : 9);
    s.addText(label || "", { shape: pres.shapes.OVAL, x: cx - d / 2, y: cy - d / 2, w: d, h: d, fill: { color: fill }, line: { color: LIGHT, width: 1 }, fontFace: SANS, fontSize: fs, bold: true, color: LIGHT, align: "center", valign: "middle", margin: 0 });
  };
  const leaf = (cx, cy) => s.addText("", { shape: pres.shapes.ROUNDED_RECTANGLE, x: cx - 0.09, y: cy - 0.09, w: 0.18, h: 0.18, rectRadius: 0.03, fill: { color: "8AA0B4" }, line: { color: LIGHT, width: 0.75 }, margin: 0 });
  const EC = "8AA0B4";
  const cols = [
    { x: 0.6, head: "Linear / sequential", tag: "ORDER", err: "error ~ O(N)·eps", accent: RUSTTX },
    { x: 4.83, head: "Pairwise / tree", tag: "ORDER", err: "error ~ O(log N)·eps", accent: TEAL2 },
    { x: 9.06, head: "fp64 accumulation", tag: "PRECISION (not order)", err: "error ~ eps", accent: GREENTX },
  ];
  const cw = 3.95;
  cols.forEach((c) => {
    s.addText("", { shape: pres.shapes.ROUNDED_RECTANGLE, x: c.x, y: 2.05, w: cw, h: 3.5, rectRadius: 0.07, fill: { color: "FBFDFE" }, line: { color: CARDLN, width: 1 }, shadow: sh({ opacity: 0.07 }) });
    s.addText(c.head, { x: c.x, y: 2.14, w: cw, h: 0.32, fontFace: SANS, fontSize: 14, bold: true, color: c.accent, align: "center", margin: 0 });
    s.addText(c.tag, { x: c.x, y: 2.46, w: cw, h: 0.24, fontFace: SANS, fontSize: 9.5, bold: true, color: MUTED, charSpacing: 1.5, align: "center", margin: 0 });
    s.addText(c.err, { x: c.x, y: 5.16, w: cw, h: 0.32, fontFace: MONO, fontSize: 12.5, bold: true, color: c.accent, align: "center", margin: 0 });
  });
  // Col 0 — linear caterpillar ((((x1+x2)+x3)+x4)
  {
    const cx = 0.6 + cw / 2, yL = 3.15;
    const L = [cx - 0.95, cx - 0.32, cx + 0.32, cx + 0.95];
    const n1 = (L[0] + L[1]) / 2, n2 = (n1 + L[2]) / 2, n3 = (n2 + L[3]) / 2;
    edge(L[0], yL, n1, 3.72, EC); edge(L[1], yL, n1, 3.72, EC);
    edge(n1, 3.72, n2, 4.28, EC); edge(L[2], yL, n2, 4.28, EC);
    edge(n2, 4.28, n3, 4.84, EC); edge(L[3], yL, n3, 4.84, EC);
    L.forEach((x) => leaf(x, yL));
    node(n1, 3.72, 0.34, RUSTTX, "+"); node(n2, 4.28, 0.34, RUSTTX, "+"); node(n3, 4.84, 0.38, INK, "Σ");
  }
  // Col 1 — balanced tree (x1+x2)+(x3+x4)
  {
    const cx = 4.83 + cw / 2, yL = 3.15;
    const L = [cx - 0.95, cx - 0.32, cx + 0.32, cx + 0.95];
    const n1 = (L[0] + L[1]) / 2, n2 = (L[2] + L[3]) / 2, rt = (n1 + n2) / 2;
    edge(L[0], yL, n1, 3.95, EC); edge(L[1], yL, n1, 3.95, EC);
    edge(L[2], yL, n2, 3.95, EC); edge(L[3], yL, n2, 3.95, EC);
    edge(n1, 3.95, rt, 4.7, EC); edge(n2, 3.95, rt, 4.7, EC);
    L.forEach((x) => leaf(x, yL));
    node(n1, 3.95, 0.34, TEAL2, "+"); node(n2, 3.95, 0.34, TEAL2, "+"); node(rt, 4.7, 0.38, INK, "Σ");
  }
  // Col 2 — fp64: same order, wider (64-bit) accumulator → bigger nodes
  {
    const cx = 9.06 + cw / 2, yL = 3.15;
    const L = [cx - 0.95, cx - 0.32, cx + 0.32, cx + 0.95];
    const n1 = (L[0] + L[1]) / 2, n2 = (n1 + L[2]) / 2, n3 = (n2 + L[3]) / 2;
    edge(L[0], yL, n1, 3.72, EC); edge(L[1], yL, n1, 3.72, EC);
    edge(n1, 3.72, n2, 4.28, EC); edge(L[2], yL, n2, 4.28, EC);
    edge(n2, 4.28, n3, 4.84, EC); edge(L[3], yL, n3, 4.84, EC);
    L.forEach((x) => leaf(x, yL));
    node(n1, 3.72, 0.44, GREENTX, "+"); node(n2, 4.28, 0.44, GREENTX, "+"); node(n3, 4.84, 0.46, INK, "Σ");
  }
  panel(s, 0.6, 5.78, 6.0, 1.02, "Consequence", "Correct C can still mismatch the reference under a fixed tolerance — and surface as failures on the ONNX Backend Scoreboard.", { fill: INK, headColor: AMBER, headSize: 14, textColor: "DCE8F0", fontSize: 13, valign: "middle" });
  panel(s, 6.7, 5.78, 6.0, 1.02, "What we do (local tests)", [
    { text: "Account for it with 64-bit accumulation on sensitive operators:  ", options: { fontFace: SANS, fontSize: 13, color: SLATE } },
    { text: "--fp32-accumulation-strategy fp64", options: { fontFace: MONO, fontSize: 12, color: INK } },
  ], { fill: GREEN, line: GREENLN, headColor: GREENTX, headSize: 14, valign: "middle" });
  footer(s);
  s.addNotes("Lesson: ONNX leaves the order of computation unspecified (and the accumulation precision too). Reductions are the clearest case. The ORDER axis: summing N values linearly accumulates rounding error of order O(N)·eps; a pairwise / tree reduction reduces that to about O(log N)·eps — same inputs, different summation order, both standard-conformant, slightly different results. The PRECISION axis (a separate point — fp64 is not an order): accumulating in 64-bit brings the error down to about eps regardless of order. Make the distinction explicit on stage, since fp64 is precision, not order. The spread grows with N and shows up in mean, variance, softmax, L2-norm, dot products, conv, matmul, etc. Because neither the order nor the precision is specified, a single fixed rtol/atol is simultaneously too loose for small reductions and too tight for large ones, and it cannot know which algorithm the reference used. This is why emx uses ULP plus an absolute floor; per-operator guidance — ideally tolerances that scale with the reduction size — would help backend authors. In practice this is not academic: under a fixed tolerance a correct backend can show failures on the ONNX Backend Scoreboard purely because the reference was computed with a different order or precision. In our local tests we account for this by using 64-bit accumulation for sensitive operators — emx's --fp32-accumulation-strategy fp64 is exactly that precision lever for cases where strict references demand it.");
}

/* SLIDE — Lesson 4 · operator importance is unknown */
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  lessonHead(s, 4, "How important is an operator? Unknown");
  s.addText("Implementing a backend, we had no signal which operators to prioritize — the spec lists them all flat.", {
    x: 1.42, y: 1.5, w: 11.3, h: 0.4, fontFace: SANS, fontSize: 14, color: SLATE, margin: 0, valign: "middle" });
  card(s, 0.6, 2.05, 5.6, 4.6, "The problem", [
    "ONNX has hundreds of operators — all listed flat",
    "No signal which are common, critical, or niche",
    "Implementation priority was largely guesswork",
    "e.g. ImageDecode is pure preprocessing, not core inference — yet looks no different from essential ops",
    "Hard to plan coverage when every operator looks equally important",
  ], { gap: 11, fontSize: 13.5 });
  card(s, 6.35, 2.05, 6.35, 2.15, "Idea 1 · operator categories", [
    "Tag operators by role: core math · NN layers · preprocessing / IO · control flow · quantization · classic ML · contrib",
    "Backend authors prioritize and scope by category",
  ], { fill: GREEN, line: GREENLN, headColor: GREENTX, gap: 8, fontSize: 13.5 });
  panel(s, 6.35, 4.4, 6.35, 2.25, "Idea 2 · an operator atlas (usage in the wild)", [
    { text: "How often does each operator actually appear in real models?", options: { fontFace: SANS, fontSize: 13.5, color: "DCE8F0", breakLine: true, paraSpaceAfter: 8 } },
    { text: "Index the ", options: { fontFace: SANS, fontSize: 13.5, color: "DCE8F0" } },
    { text: "~40k ONNX models on Hugging Face", options: { fontFace: SANS, fontSize: 13.5, bold: true, color: AMBER } },
    { text: " → a popularity + coverage atlas to guide backend authors and spec maintainers.", options: { fontFace: SANS, fontSize: 13.5, color: "DCE8F0" } },
  ], { fill: INK, headColor: AMBER, headSize: 15, valign: "top" });
  footer(s);
  s.addNotes("A process/community lesson, not a code lesson. When implementing the operator set, we had no guidance on which operators matter most — ONNX lists hundreds of operators as a flat set, with no indication of how common, critical or niche each is. So implementation priority was guesswork. Concrete example: ImageDecode is purely a preprocessing/IO operator, not part of core inference, yet in the spec it is indistinguishable in importance from essential math or NN operators. Two constructive ideas for the community: (1) introduce operator categories or roles — core math, NN layers, preprocessing/IO, control flow, quantization, classic ML, contrib — so backend authors can prioritize and scope; (2) build an operator 'atlas' of real-world usage frequency by indexing the roughly 40,000 ONNX models hosted on Hugging Face, and publish a popularity plus coverage map. That would let any backend prioritize the operators that actually appear in practice instead of guessing.");
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
    { text: r[0], options: { fill: { color: r[3] }, color: INK, bold: true, fontFace: MONO, fontSize: 14 } },
    { text: r[1], options: { fill: { color: r[3] }, color: SLATE, fontFace: MONO, fontSize: 14 } },
    { text: r[2], options: { fill: { color: r[3] }, color: SLATE, fontSize: 14 } },
  ]);
  s.addTable([header, ...body], { x: 0.6, y: 2.0, w: 12.1, colW: [3.0, 3.2, 5.9], border: { pt: 0.5, color: CARDLN }, align: "left", valign: "middle", rowH: 0.6, margin: [3, 6, 3, 6] });
  s.addText([
    { text: "Takeaway:  ", options: { bold: true, color: AMBER } },
    { text: "common numeric types are easy; reduced-precision and sub-byte types expose the gap between ONNX's type system and portable C. FP8 today is more useful for type coverage than for efficient portable execution.", options: { color: SLATE } },
  ], { x: 0.6, y: 6.4, w: 12.1, h: 0.7, fontFace: SANS, fontSize: 15, margin: 0, valign: "top" });
  footer(s);
  s.addNotes("Common numeric types map cleanly. float16 uses _Float16 where supported; bfloat16 relies on __bf16 extensions; 2/4-bit ints use C23 _BitInt(N) but arrays can waste memory versus packed storage; FP8/FP4 aren't standard C arithmetic types and are emulated with integer storage plus conversion helpers. FP8 is currently more about completeness than efficiency.");
}

/* SLIDE — complete data-type overview */
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  title(s, "Data types: complete coverage");
  s.addText([
    { text: "For ~100% coverage you need them all.  ", options: { bold: true, color: AMBER } },
    { text: "Operators and tests use every value type — so each maps to an explicit C representation; the awkward ones (sub-byte, FP8/FP4, strings, containers) cannot be skipped.", options: { color: SLATE } },
  ], { x: 0.6, y: 1.4, w: 12.1, h: 0.6, fontFace: SANS, fontSize: 14, margin: 0, valign: "middle" });
  const header = [
    { text: "Category", options: { bold: true, color: LIGHT, fill: { color: INK } } },
    { text: "ONNX value type(s)", options: { bold: true, color: LIGHT, fill: { color: INK } } },
    { text: "C representation", options: { bold: true, color: LIGHT, fill: { color: INK } } },
  ];
  const rows = [
    ["Integers & bool", "int8/16/32/64, uint8/16/32/64, bool", "int8_t … int64_t, uint8_t … uint64_t,\n_Bool", GREEN, GREENTX],
    ["Floating-point", "float, double, float16, bfloat16", "float, double,\nfloat16 → _Float16, bfloat16 → __bf16", GREEN, GREENTX],
    ["Sub-byte integers", "int2, uint2, int4, uint4", "_BitInt(4) / unsigned _BitInt(4),\n_BitInt(2) / unsigned _BitInt(2)  (C23)", GREEN, GREENTX],
    ["Low-precision floats", "float8 e4m3fn/e4m3fnuz/e5m2/e5m2fnuz/e8m0, float4 e2m1", "uint8_t storage\n+ float conversion helpers", SAND, SANDTX],
    ["Strings", "tensor(string)", "char x[3][EMX_STRING_MAX_LEN];\n'\\0'-terminated, fixed size", CARD, TEAL2],
    ["Sequences", "sequence(T)", "float x[EMX_SEQUENCE_MAX_LEN][H][W];\nidx_t x__count;", CARD, TEAL2],
    ["Optional", "optional(T)", "float x[3];\n_Bool x_present;", CARD, TEAL2],
    ["Not supported", "complex64/128, map, sparse_tensor, opaque", "—", RUST, RUSTTX],
  ];
  const body = rows.map((r) => [
    { text: r[0], options: { fill: { color: r[3] }, color: r[4], bold: true, fontFace: SANS, fontSize: 12 } },
    { text: r[1], options: { fill: { color: r[3] }, color: SLATE, fontFace: MONO, fontSize: 11 } },
    { text: r[2], options: { fill: { color: r[3] }, color: SLATE, fontFace: MONO, fontSize: 11 } },
  ]);
  s.addTable([header, ...body], { x: 0.6, y: 2.1, w: 12.1, colW: [2.5, 5.2, 4.4], border: { pt: 0.5, color: CARDLN }, align: "left", valign: "middle", rowH: 0.52, margin: [3, 8, 3, 8] });
  footer(s);
  s.addNotes("Focus: complete data-type support is a prerequisite for high coverage. The ONNX backend tests and real operators exercise every value type, so to reach ~100% coverage you cannot cherry-pick the convenient dtypes — the awkward ones (sub-byte integers, FP8/FP4, strings, sequences, optional) have to be handled too, each with an explicit C representation. The complete data-type picture, beyond the scalar numeric mapping on the previous slide. Green: types that map directly to native C — standard integers and bool, float/double, plus float16 (_Float16) and bfloat16 (__bf16) where the compiler/target supports them. Sand: types that need a workaround — 2/4-bit integers use C23 _BitInt(N) (correct range, but arrays may waste memory versus packed storage), and the float8/float4 family is stored as uint8 with explicit float conversion helpers since they are not native C arithmetic types. Neutral: aggregate/container types — strings become fixed-size '\\0'-terminated char arrays bounded by EMX_STRING_MAX_LEN; sequence(T) becomes a fixed-capacity array plus a __count, and optional(T) becomes the value plus a _Bool _present flag. Not supported: complex64/complex128 and the ONNX map, sparse_tensor and opaque value types. The unifying point: every supported value type has an explicit, statically-laid-out C representation — no dynamic typing, no heap.");
}

/* SLIDE — Proposal: declare size bounds in the model */
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  title(s, "Maximum container sizes: a compile-time parameter");
  s.addText([
    { text: "emx-onnx-cgen exposes the maximum size as a parameter — ", options: { color: SLATE } },
    { text: "today implemented as a #define macro.", options: { bold: true, color: TEAL2 } },
  ], { x: 0.6, y: 1.46, w: 12.1, h: 0.45, fontFace: SANS, fontSize: 14.5, margin: 0, valign: "middle" });
  codeBox(s,
`#ifndef EMX_SEQUENCE_MAX_LEN
#define EMX_SEQUENCE_MAX_LEN 32   /* override per build */
#endif

#ifndef EMX_STRING_MAX_LEN
#define EMX_STRING_MAX_LEN 256
#endif`,
    { x: 0.6, y: 2.05, w: 6.0, h: 3.4, fontSize: 13, caption: "Today — a build-time parameter (#define)" });
  card(s, 6.75, 2.05, 5.95, 3.4, "The problems", [
    "One global value for ALL sequences / ALL strings",
    "Too large → wasted memory; too small → truncation",
    "The bound lives in the backend, not in the model",
    "Not portable — each backend picks its own limit",
  ], { fill: SAND, line: SANDLN, headColor: SANDTX, gap: 12, fontSize: 14 });
  panel(s, 0.6, 5.7, 12.1, 1.05, "Proposal", "These limits should not be a backend parameter at all — they belong in the ONNX type system itself (a general extension, not emx-specific). Developed on the sequence-subtypes slide.", { fill: INK, headColor: AMBER, headSize: 15, textColor: "DCE8F0", fontSize: 14, valign: "middle" });
  footer(s);
  s.addNotes("This slide is only about the current mechanism and its problems — the type-system proposal is developed on the sequence-subtypes (ragged) slide. emx-onnx-cgen lets the user specify the maximum container size as a parameter; today that parameter is a compile-time #define macro (EMX_SEQUENCE_MAX_LEN default 32, EMX_STRING_MAX_LEN default 256), #ifndef-guarded so a build can override it. The problems: it is a single global value, so one number applies to every sequence or every string — too large wastes memory, too small truncates; and the bound lives in the backend rather than the model, so it is not portable and different backends can choose different limits. Close with only a pointer: the real fix is to anchor the bound in the ONNX type system itself, which is a general extension and not emx-specific.");
}

/* SLIDE — Lesson 5 · sequence element type underspecified → type-system proposal */
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  lessonHead(s, 5, "Sequences: the element type is underspecified");
  s.addText("A sequence carries a value type, but the element tensor's concrete shape — sometimes even its rank — is often missing. Static C needs it up front.", {
    x: 0.6, y: 1.46, w: 12.1, h: 0.45, fontFace: SANS, fontSize: 14.5, color: SLATE, margin: 0, valign: "middle" });
  // left: the gap
  card(s, 0.6, 2.05, 5.5, 4.55, "The gap", [
    "sequence(T) gives the element dtype, but its shape — sometimes its rank — can be unspecified",
    "Element shapes may also vary per item (ragged sequences)",
    "A runtime resolves this at execution; static C cannot",
    "No element shape → no fixed storage to reserve",
  ], { gap: 14, fontSize: 14 });
  // right: today (CLI example) → arrow → better (type system)
  panel(s, 6.25, 2.05, 6.45, 1.95, "emx today — an extra spec on the CLI", [
    { text: "--sequence-element-shape boxes=[<=100, 4]", options: { fontFace: MONO, fontSize: 13, color: INK, breakLine: true, paraSpaceAfter: 7 } },
    { text: "Declares element rank + per-axis maxima; capacity from EMX_SEQUENCE_MAX_LEN.", options: { fontFace: SANS, fontSize: 12.5, color: SLATE, breakLine: true, paraSpaceAfter: 4 } },
    { text: "But it is out-of-band — every tool needs its own flag.", options: { fontFace: SANS, fontSize: 12, color: SANDTX } },
  ], { fill: SAND, line: SANDLN, headColor: SANDTX, headSize: 14, valign: "top" });
  s.addText("↓", { x: 9.2, y: 4.0, w: 0.5, h: 0.45, fontFace: SANS, fontSize: 24, bold: true, color: TEAL, align: "center", valign: "middle", margin: 0 });
  panel(s, 6.25, 4.5, 6.45, 2.1, "Better — bound it in the ONNX type system", [
    { text: "sequence(tensor(float)[<=100, 4])[max_len=20]", options: { fontFace: MONO, fontSize: 14, bold: true, color: AMBER, breakLine: true, paraSpaceAfter: 5 } },
    { text: "element 2-D: axis 0 ≤ 100, axis 1 = 4;  ≤ 20 items.", options: { fontFace: SANS, fontSize: 12.5, color: "DCE8F0", breakLine: true, paraSpaceAfter: 9 } },
    { text: "A structured ", options: { fontFace: SANS, fontSize: 12.5, color: "DCE8F0" } },
    { text: "TypeProto / TensorShapeProto", options: { fontFace: SANS, fontSize: 12.5, bold: true, color: LIGHT } },
    { text: " field — next to dim_value / dim_param, not a string and not a per-tool flag. Read by every backend.", options: { fontFace: SANS, fontSize: 12.5, color: "DCE8F0" } },
  ], { fill: INK, headColor: AMBER, headSize: 14, valign: "top" });
  footer(s);
  s.addNotes("Two things: (1) the sequence element/base type is itself underspecified, and emx's current handling; (2) the deeper, general proposal to put such specs in the ONNX type system. On (1): not only the length is unbounded — ONNX gives the element dtype, but the concrete element shape, sometimes its rank, may be missing, and for ragged sequences it varies per item; a runtime discovers it at execution, static C cannot. emx requires an explicit extra spec on the command line, --sequence-element-shape boxes=[<=100,4], declaring element rank and per-axis maxima (capacity from EMX_SEQUENCE_MAX_LEN, variable axes get a per-item dim array); a missing spec fails clearly. But that is out-of-band: every tool needs its own flag. On (2): the bound belongs in the model. ONNX types are not strings — they are protobuf messages (TypeProto, with TypeProto.Tensor.elem_type a DataType enum and shape a TensorShapeProto of dim_value / dim_param). So a per-dimension maximum should be a structured field in TypeProto / TensorShapeProto, sitting next to dim_value / dim_param — not a string in metadata_props, and not a per-tool CLI flag. Then every backend, exporter and tool reads the same spec. The text sequence(tensor(float)[<=100, 4])[max_len=20] is only a human-readable sketch: element is 2-D with axis 0 <= 100 and axis 1 = 4, sequence holds <= 20 items (numbers deliberately different so it is clear which bound is which). This is the same idea as the size-bounds proposal, shown here on the most concrete case.");
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
  panel(s, 6.8, 1.95, 5.9, 2.2, "External validation", "The public ONNX Backend Scoreboard lists emx-onnx-cgen directly after ONNX Reference in the stable-build table — a useful compatibility signal, since it is based on ONNX backend unit tests.", { fill: CARD, line: CARDLN, headColor: TEAL2, headSize: 17, textColor: SLATE, fontSize: 15.5 });
  panel(s, 6.8, 4.35, 5.9, 2.15, "But unit-test execution ≠ backend quality", "The scoreboard does not capture deterministic generated C, static memory planning, readable source, auditability, or safety-oriented build integration. For embedded and safety-adjacent targets, those are the real backend features.", { fill: INK, headColor: AMBER, headSize: 16, textColor: "CFE0EC", fontSize: 15 });
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
    s.addText([
      { text: st[0], options: { fontFace: SERIF, fontSize: 38, bold: true, color: AMBER, breakLine: true, paraSpaceAfter: 6 } },
      { text: st[1], options: { fontFace: SANS, fontSize: 15.5, color: "CFE0EC" } },
    ], { shape: pres.shapes.ROUNDED_RECTANGLE, x, y: y0, w: bw, h: bh, rectRadius: 0.08, fill: { color: INK }, shadow: sh({ opacity: 0.14 }), valign: "top", margin: [12, 14, 10, 18] });
  });
  card(s, 0.6, 4.2, 6.0, 2.5, "Coverage is reported, reproducible, visible", [
    "SUPPORT_OPS.md — operator support status",
    "ONNX_SUPPORT.md — official backend model coverage",
    "ONNX_SUPPORT.md — ORT artifact corpus coverage",
    "Unsupported cases documented as expected errors",
  ], { headColor: TEAL2, gap: 6 });
  panel(s, 6.8, 4.2, 5.9, 2.5, "emx-ort-test-artifacts", [
    { text: "Exports ORT tests into backend-test-like artifacts (model.onnx + test_data_set_*), reusable outside the ORT C++ harness — a broader, runtime-oriented reality check beyond the compact official node tests.", options: { fontFace: SANS, fontSize: 14, color: SLATE, breakLine: true, paraSpaceAfter: 12 } },
    { text: "ORT test case → ONNX model + test_data_set_* → backend-runnable", options: { fontFace: MONO, fontSize: 12.5, color: TEAL2 } },
  ], { fill: CARD, line: CARDLN, headColor: INK, headSize: 17, textColor: SLATE });
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
    x: 0.6, y: 6.65, w: 12.1, h: 0.5, fontFace: SANS, fontSize: 15.5, italic: true, color: MUTED, margin: 0 });
  footer(s);
  s.addNotes("Source: onnx.ai/backend-scoreboard. Mention the placement briefly as credibility, but don't make it the main claim — it measures backend unit-test execution, not auditability or deterministic AOT constraints.");
}

/* SLIDE — takeaways */
{
  const s = pres.addSlide();
  s.background = { color: INK };
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 0.52, w: 0.22, h: 0.22, rectRadius: 0.05, fill: { color: TEAL } });
  s.addText("FOR THE ONNX COMMUNITY", { x: 0.92, y: 0.47, w: 9, h: 0.33, fontFace: SANS, fontSize: 14, bold: true, color: TEAL, charSpacing: 2, margin: 0, valign: "middle" });
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
    s.addText([
      { text: c[0], options: { fontFace: SANS, fontSize: 17, bold: true, color: AMBER, breakLine: true, paraSpaceAfter: 6 } },
      { text: c[1], options: { fontFace: SANS, fontSize: 13.5, color: "CFE0EC" } },
    ], { shape: pres.shapes.ROUNDED_RECTANGLE, x, y, w: bw, h: bh, rectRadius: 0.08, fill: { color: "13355A" }, line: { color: "1E4A78", width: 1 }, valign: "top", margin: [12, 14, 10, 18] });
  });
  s.addText("ONNX is excellent interchange — deterministic AOT compilation just needs a few constraints made explicit.", {
    x: 0.6, y: 6.5, w: 9.2, h: 0.5, fontFace: SANS, fontSize: 16.5, italic: true, color: "AFC6D8", margin: 0 });
  s.addImage({ path: LOGO_W, x: W - 2.0, y: 6.55, w: 1.5, h: 0.40 });
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
    x: 0.9, y: 2.6, w: 11.5, h: 3.4, fontFace: SANS, fontSize: 17, margin: 0, valign: "top" });
  s.addShape(pres.shapes.LINE, { x: 0.9, y: 6.25, w: 3.5, h: 0, line: { color: TEAL, width: 2 } });
  s.addText("Thank you  ·  emx-onnx-cgen is open source", { x: 0.9, y: 6.45, w: 9, h: 0.5, fontFace: SANS, fontSize: 17, bold: true, color: LIGHT, margin: 0 });
  s.addImage({ path: LOGO_W, x: W - 2.0, y: 6.5, w: 1.5, h: 0.40 });
  s.addNotes("Open the floor. Lead with the profile question, then let the room pick threads: shared inference artifact, domain registration, portable custom-op rules, numerical contracts, and how scoreboards could reflect deterministic, auditable code generation rather than only unit-test execution.");
}

/* =================================================================== */
/* SECTION 07 — CONDENSED SINGLE-TOPIC ALTERNATIVES                    */
/* =================================================================== */
sectionDivider(7, "Condensed alternatives", "Each lesson on a single slide (from the 20-minute deck)", [
  "Compact one-slide versions of the expanded lessons",
  "Swap in when time is short — pick these or the detailed ones",
]);

/* SLIDE — Lesson 1 condensed */
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  lessonHead(s, 1, "ONNX is effectively unbounded");
  s.addText("Each case: what ONNX leaves open, how emx-onnx-cgen bounds it, and the limitation today.", {
    x: 0.6, y: 1.38, w: 12.1, h: 0.36, fontFace: SANS, fontSize: 13.5, color: SLATE, margin: 0, valign: "middle" });
  const D = [
    {
      cat: "Dynamic dimensions",
      problem: "Symbolic / unknown axes give a runtime extent — but no maximum.",
      emx: [
        { text: "C99 VLA  x[N][C]", options: { fontFace: MONO, fontSize: 11.5, color: INK, breakLine: true, paraSpaceAfter: 5 } },
        { text: "extents become runtime arguments", options: { fontFace: SANS, fontSize: 11.5, color: SLATE, breakLine: true, paraSpaceAfter: 10 } },
      ],
      down: "unbounded temporaries still need a memory policy",
    },
    {
      cat: "Sequence length",
      problem: "Containers have no natural capacity — one level above a dynamic dim.",
      emx: [
        { text: "T x[EMX_SEQUENCE_MAX_LEN][…];", options: { fontFace: MONO, fontSize: 11, color: INK, breakLine: true, paraSpaceAfter: 2 } },
        { text: "idx_t x__count;", options: { fontFace: MONO, fontSize: 11, color: INK, breakLine: true, paraSpaceAfter: 5 } },
        { text: "#define EMX_SEQUENCE_MAX_LEN 32", options: { fontFace: MONO, fontSize: 10.5, color: TEAL2, breakLine: true, paraSpaceAfter: 10 } },
      ],
      down: "one global cap for all sequences → wasted memory",
    },
    {
      cat: "String size",
      problem: "String tensors carry no maximum character length.",
      emx: [
        { text: "char x[…][EMX_STRING_MAX_LEN];", options: { fontFace: MONO, fontSize: 11, color: INK, breakLine: true, paraSpaceAfter: 5 } },
        { text: "#define EMX_STRING_MAX_LEN 256", options: { fontFace: MONO, fontSize: 10.5, color: TEAL2, breakLine: true, paraSpaceAfter: 10 } },
      ],
      down: "one global cap for all strings → waste / truncation",
    },
  ];
  const bw = 3.84, gap = 0.28, x0 = 0.6;
  D.forEach((d, i) => {
    const x = x0 + i * (bw + gap);
    s.addText(d.cat, { x, y: 1.82, w: bw, h: 0.38, fontFace: SANS, fontSize: 14, bold: true, color: TEAL2, align: "center", valign: "middle", margin: 0 });
    s.addText(d.problem, { shape: pres.shapes.ROUNDED_RECTANGLE, x, y: 2.22, w: bw, h: 1.2, rectRadius: 0.08, fill: { color: SAND }, line: { color: SANDLN, width: 1 }, shadow: sh({ opacity: 0.09 }), fontFace: SANS, fontSize: 12.5, color: SLATE, valign: "middle", margin: [10, 12, 10, 14] });
    const runs = d.emx.concat([
      { text: "Downside: ", options: { fontFace: SANS, fontSize: 10.5, bold: true, color: SANDTX } },
      { text: d.down, options: { fontFace: SANS, fontSize: 10.5, color: SANDTX } },
    ]);
    s.addText(runs, { shape: pres.shapes.ROUNDED_RECTANGLE, x, y: 3.5, w: bw, h: 2.05, rectRadius: 0.08, fill: { color: GREEN }, line: { color: GREENLN, width: 1 }, shadow: sh({ opacity: 0.09 }), valign: "top", margin: [11, 12, 10, 14] });
  });
  panel(s, 0.6, 5.72, 12.1, 1.05, "Proposal — make size bounds part of the ONNX type system", "A general ONNX extension (not emx-specific): a standard, optional maximum size per type and per tensor — so every backend sizes each buffer exactly, instead of each inventing its own global cap that wastes memory or truncates.", { fill: INK, headColor: AMBER, headSize: 14, textColor: "DCE8F0", fontSize: 13, valign: "middle" });
  footer(s);
  s.addNotes("Condensed Lesson 1, as a problem→handling matrix. Top row: what ONNX leaves open. Middle: how emx-onnx-cgen bounds it — VLAs for dynamic dims, fixed-capacity sequence arrays plus count, fixed string slots, with overridable #define macros. The honest limitation today: the bound is a single global macro (EMX_SEQUENCE_MAX_LEN, EMX_STRING_MAX_LEN), so one value applies to every sequence/string and can waste memory or truncate. Proposal for the community: let ONNX declare maximum sizes per data type or per tensor inside the model, so backends size each buffer exactly instead of relying on one global cap.");
}

/* SLIDE — Lesson 2 condensed */
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  lessonHead(s, 2, "Type & shape inference is not enough");
  card(s, 0.6, 1.7, 6.0, 2.5, "Where it falls short", [
    "Inference is incomplete for some operators",
    "Output shape can depend on values, attributes or subgraphs",
    "Container element shapes may simply be missing",
    "Sequence types complicate static analysis & ABI design",
  ], { gap: 7, fontSize: 13.5 });
  card(s, 0.6, 4.35, 6.0, 2.3, "Compiler response", [
    "Normalize attributes & types on import",
    "Add compiler-side inference where ONNX stops",
    "Missing information becomes an explicit error",
  ], { fill: GREEN, line: GREENLN, headColor: GREENTX, gap: 7, fontSize: 13.5 });
  panel(s, 6.75, 1.7, 5.95, 2.0, "The duplication problem", "No reliable standalone library infers shapes/types for dynamic models in a way static code generators can reuse — so importers, MLIR frontends, codegens, verifiers and runtimes each reimplement it, and may disagree.", { fill: RUST, line: RUSTLN, headColor: RUSTTX, headSize: 15, textColor: SLATE, fontSize: 13 });
  panel(s, 6.75, 3.9, 5.95, 2.75, null, [
    { text: "VISION  ", options: { fontFace: SANS, fontSize: 14, bold: true, color: AMBER } },
    { text: "a shared, extensible inference layer", options: { fontFace: SANS, fontSize: 14, bold: true, color: LIGHT, breakLine: true, paraSpaceAfter: 9 } },
    { text: "Preserve symbolic dims, constraints & unresolved facts", options: { bullet: { code: "2022" }, fontFace: SANS, fontSize: 12.5, color: "CFE0EC", breakLine: true, paraSpaceAfter: 6 } },
    { text: "Registry for external domains (ORT contrib ops)", options: { bullet: { code: "2022" }, fontFace: SANS, fontSize: 12.5, color: "CFE0EC", breakLine: true, paraSpaceAfter: 6 } },
    { text: "Persist results back into ONNX (value_info / metadata)", options: { bullet: { code: "2022" }, fontFace: SANS, fontSize: 12.5, color: "CFE0EC", breakLine: true, paraSpaceAfter: 6 } },
    { text: "One DSL → C++ & Python, numeric + symbolic (SymPy)", options: { bullet: { code: "2022" }, fontFace: SANS, fontSize: 12.5, color: "CFE0EC" } },
  ], { fill: INK });
  footer(s);
  s.addNotes("Condensed Lesson 2. A mismatch, not only a flaw: flexible model format versus static compiler input. Inference should be separable from code generation and from a fixed operator universe. A shared, extensible inference library that preserves symbolic/dynamic dims and constraints — and writes results back into the model — would stop every backend reimplementing it. We are exploring a DSL where a rule is written once and runs from C++ and Python, numerically and symbolically. Present as vision/research.");
}

/* SLIDE — Lesson 3 condensed */
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  lessonHead(s, 3, "Numerical accuracy is underspecified");
  card(s, 0.6, 1.7, 6.0, 2.7, "The ambiguity", [
    "Official tests give examples, not full accuracy contracts",
    "Reference impls differ on edge cases & FP behaviour",
    "node tests carry no per-operator tolerance (data.json = real models)",
    "Accumulation, rounding, approximations left to the backend",
  ], { gap: 7, fontSize: 13.5 });
  const tHdr = [
    { text: "Framework / tool", options: { bold: true, color: LIGHT, fill: { color: INK } } },
    { text: "Tolerance policy", options: { bold: true, color: LIGHT, fill: { color: INK } } },
  ];
  const tRows = [
    ["ONNX backend", "rtol 1e-3, atol 1e-7"],
    ["ONNX Runtime", "per-sample 1e-3, config.txt"],
    ["PyTorch", "dtype-dependent defaults"],
    ["NumPy / JAX", "atol + rtol·|expected|"],
    ["Polygraphy", "per-output; FP32 too strict for FP16"],
  ];
  const tBody = tRows.map((r, i) => [
    { text: r[0], options: { fill: { color: i % 2 ? LIGHT : CARD }, color: INK, bold: true, fontFace: SANS, fontSize: 11.5 } },
    { text: r[1], options: { fill: { color: i % 2 ? LIGHT : CARD }, color: SLATE, fontFace: MONO, fontSize: 11 } },
  ]);
  s.addTable([tHdr, ...tBody], { x: 6.75, y: 1.7, w: 5.95, colW: [2.5, 3.45], border: { pt: 0.5, color: CARDLN }, align: "left", valign: "middle", rowH: 0.44, margin: [3, 7, 3, 7] });
  panel(s, 0.6, 4.55, 6.0, 2.1, "emx: ULP-based, dtype-aware", [
    { text: "Ignore abs diffs up to  ", options: { fontFace: SANS, fontSize: 12.5, color: SLATE } },
    { text: "atol_eps · eps(dtype)", options: { fontFace: MONO, fontSize: 12.5, color: INK, breakLine: true, paraSpaceAfter: 6 } },
    { text: "Then measure  ", options: { fontFace: SANS, fontSize: 12.5, color: SLATE } },
    { text: "ULP distance", options: { fontFace: MONO, fontSize: 12.5, color: INK } },
    { text: " ; non-float must match exactly", options: { fontFace: SANS, fontSize: 12.5, color: SLATE, breakLine: true, paraSpaceAfter: 6 } },
    { text: "--atol-eps  --max-ulp  --fp32-accumulation-strategy fp64", options: { fontFace: MONO, fontSize: 12, color: INK } },
  ], { fill: GREEN, line: GREENLN, headColor: GREENTX, headSize: 14, valign: "top" });
  panel(s, 6.75, 4.55, 5.95, 2.1, "Why ULP, not fixed rtol/atol", "float16, float32, float64 have very different spacing between representable values. ULP asks the real question: how many representable values apart are the results? Some reference outputs are high-precision, so a few official tests need fp64 accumulation.", { fill: INK, headColor: AMBER, headSize: 14, textColor: "CFE0EC", fontSize: 12.5 });
  footer(s);
  s.addNotes("Condensed Lesson 3. Validation is an engineering policy unless the standard defines the contract. A single fixed rtol/atol is not equally meaningful across dtypes — float16/32/64 differ enormously in spacing. emx uses an absolute-epsilon gate then ULP distance, exact for non-float. Some ONNX reference outputs are effectively high precision, which under strict ULP forced 64-bit accumulation on selected tests, exposed as --fp32-accumulation-strategy fp64.");
}

pres.writeFile({ fileName: process.env.OUT || "onnx-community-day-aot-onnx-to-c-v3.pptx" }).then((f) => console.log("WROTE", f, "slides:", pageNo + "(+dividers/title)"));
