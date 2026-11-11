// Standalone deck: alternative designs for the "explicit arrays / dynamic models" slide.
// Focus: we always emit explicit, typed N-D C arrays — even for dynamic models (VLA is just the mechanism).
const pptxgen = require("pptxgenjs");
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.title = "Explicit-arrays slide — alternatives";
const W = 13.333, H = 7.5;

const INK = "0E3F66", DEEP = "0A2E4A", TEAL = "126AAA", TEAL2 = "0E537F", AMBER = "F18926";
const SLATE = "454545", MUTED = "8A8A8A", LIGHT = "FFFFFF", CARD = "F1F5F9", CARDLN = "D9E3EA";
const CODEBG = "FBF3E8", CODEFG = "13344B", CODELN = "F0E2CC", CODECAP = "C06A18";
const GREEN = "EEF6E8", GREENLN = "D7E9C9", GREENTX = "4E8A2E";
const RUST = "FBECEC", RUSTLN = "EBD2D2", RUSTTX = "B23A3A";
const SAND = "FDF1E4", SANDLN = "F6DCC0", SANDTX = "C06A18";
const SERIF = "Rajdhani Semibold", SANS = "Arial", MONO = "Consolas";

const sh = (o = {}) => Object.assign({ type: "outer", color: "0A2E4A", blur: 9, offset: 3, angle: 90, opacity: 0.14 }, o);
let pageNo = 0;
function footer(slide) {
  pageNo += 1;
  slide.addShape(pres.shapes.LINE, { x: 0.5, y: 6.94, w: 12.33, h: 0, line: { color: TEAL, width: 1.25 } });
  slide.addText("Explicit-arrays slide — alternative " + pageNo, { x: 0.6, y: 7.05, w: 8, h: 0.3, fontFace: SANS, fontSize: 10.5, color: MUTED, margin: 0, valign: "middle" });
  slide.addText("ALT " + pageNo, { x: W - 1.3, y: 7.05, w: 0.8, h: 0.3, fontFace: SANS, fontSize: 12, bold: true, color: TEAL2, align: "right", valign: "middle", margin: 0 });
}
function title(slide, text, y = 0.58) {
  slide.addText(text, { x: 0.6, y, w: 12.1, h: 0.9, fontFace: SERIF, fontSize: 30, bold: true, color: TEAL, margin: 0, valign: "middle" });
}
function lead(slide, text, y = 1.46) {
  slide.addText(text, { x: 0.6, y, w: 12.1, h: 0.45, fontFace: SANS, fontSize: 14.5, color: SLATE, margin: 0, valign: "middle" });
}
function card(slide, x, y, w, h, header, bullets, opt = {}) {
  const runs = [];
  if (header) runs.push({ text: header, options: { fontFace: SANS, fontSize: opt.headSize || 15, bold: true, color: opt.headColor || TEAL2, breakLine: true, paraSpaceAfter: 9 } });
  if (bullets && bullets.length) {
    const code = opt.bulletCode || "2022";
    bullets.forEach((b) => runs.push({ text: b, options: { bullet: { code, indent: 14 }, fontFace: SANS, fontSize: opt.fontSize || 12.5, color: opt.textColor || SLATE, breakLine: true, paraSpaceAfter: opt.gap != null ? opt.gap : 6 } }));
  }
  slide.addText(runs, { shape: pres.shapes.ROUNDED_RECTANGLE, x, y, w, h, rectRadius: 0.07, fill: { color: opt.fill || CARD }, line: { color: opt.line || CARDLN, width: 1 }, shadow: sh({ opacity: 0.10 }), align: "left", valign: "top", margin: [14, 12, 10, 18] });
}
function codeBox(slide, code, opt = {}) {
  const o = Object.assign({ x: 0.6, y: 2.2, w: 6.0, h: 2.4, fontSize: 15 }, opt);
  const cf = Math.min(o.fontSize, 13);
  const runs = [];
  if (o.caption) runs.push({ text: o.caption, options: { fontFace: SANS, fontSize: 12, bold: true, color: o.capColor || CODECAP, breakLine: true, paraSpaceAfter: 6 } });
  runs.push({ text: code, options: { fontFace: MONO, fontSize: cf, color: o.fg || CODEFG } });
  slide.addText(runs, { shape: pres.shapes.ROUNDED_RECTANGLE, x: o.x, y: o.y, w: o.w, h: o.h, rectRadius: 0.06, fill: { color: o.bg || CODEBG }, line: { color: o.ln || CODELN, width: 1 }, shadow: sh({ opacity: 0.08 }), align: "left", valign: "top", margin: [9, 10, 9, 14], lineSpacingMultiple: 1.04 });
}
function panel(slide, x, y, w, h, header, body, opt = {}) {
  const runs = [];
  if (header) runs.push({ text: header, options: { fontFace: SANS, fontSize: opt.headSize || 16.5, bold: true, color: opt.headColor || AMBER, breakLine: true, paraSpaceAfter: opt.headGap != null ? opt.headGap : 8 } });
  if (typeof body === "string") runs.push({ text: body, options: { fontFace: SANS, fontSize: opt.fontSize || 16, color: opt.textColor || "DCE8F0" } });
  else if (Array.isArray(body)) body.forEach((r) => runs.push(r));
  slide.addText(runs, { shape: pres.shapes.ROUNDED_RECTANGLE, x, y, w, h, rectRadius: opt.rectRadius || 0.07, fill: { color: opt.fill || INK }, line: opt.line ? { color: opt.line, width: 1 } : { type: "none" }, shadow: opt.shadow === false ? undefined : sh({ opacity: 0.10 }), align: "left", valign: opt.valign || "top", margin: opt.margin || [14, 14, 12, 18] });
}
function originLine(slide, y = 6.74) {
  slide.addText("Keeping arrays explicit even for dynamic shapes (onnx2c's weak spot) is why we started emx-onnx-cgen.", { x: 0.6, y, w: 12.1, h: 0.2, fontFace: SANS, fontSize: 12.5, italic: true, color: MUTED, margin: 0 });
}

/* =================== ALT 1 — three cases, color-coded code ================= */
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  title(s, "Explicit typed arrays — even for dynamic models");
  lead(s, "Every tensor → a typed N-D C array; rank & extents live in the type, not in hand-written index math.");
  codeBox(s,
`/* FLAT pointer — shape is only a convention */
void f(int N, int C, const float *x) {
  /* ... x[n*C + c] ...  (you compute offsets) */
}`,
    { x: 0.6, y: 2.05, w: 7.3, h: 1.35, fontSize: 12.5, caption: "✗  flat — structure lost", bg: RUST, ln: RUSTLN, capColor: RUSTTX });
  codeBox(s,
`/* EXPLICIT, static extents */
void g(const float x[2][3], float y[2][3]);`,
    { x: 0.6, y: 3.55, w: 7.3, h: 0.95, fontSize: 12.5, caption: "✓  explicit & static" });
  codeBox(s,
`/* EXPLICIT, dynamic extents — C99 VLA */
void model(int N, int C,
           const float x[N][C],
           float       y[N][C]) {
  float tmp[N][C];          /* local VLA */
  for (int n = 0; n < N; ++n)
    for (int c = 0; c < C; ++c)
      tmp[n][c] = relu(x[n][c]);
}`,
    { x: 0.6, y: 4.65, w: 7.3, h: 2.0, fontSize: 12.5, caption: "✓  explicit & dynamic (params + temp)", bg: GREEN, ln: GREENLN, capColor: GREENTX });
  card(s, 8.1, 2.05, 4.6, 4.6, "Why explicit arrays", [
    "Rank & per-axis extent live in the C type",
    "x[n][c] mirrors the tensor — no n*C + c",
    "Out-of-bounds is UB on the object → analyzable",
    "Readable, reviewable, easy to vectorize",
    "Same idea holds for dynamic models — VLAs just make the extents runtime values",
  ], { fill: INK, headColor: AMBER, textColor: "DCE8F0", gap: 11, fontSize: 13.5 });
  originLine(s);
  footer(s);
}

/* =================== ALT 2 — narrative: pointer → typed → dynamic ========== */
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  title(s, "We keep tensors as explicit arrays");
  lead(s, "From a bare pointer to a typed array — and we keep it typed even when the shape is dynamic.");
  const cols = [
    ["Flat pointer", "float *x → x[n*C + c]\nshape lives only in your head; out-of-bounds is invisible", SAND, SANDLN, SANDTX, SLATE],
    ["Explicit typed array", "float x[2][3]\nrank & extents in the type; x[n][c] mirrors the tensor", GREEN, GREENLN, GREENTX, SLATE],
    ["Even when dynamic", "C99 VLA: float x[N][C]\nextents at runtime — for parameters and local temporaries", INK, INK, AMBER, "DCE8F0"],
  ];
  const bw = 3.95, gap = 0.28, x0 = 0.6, y0 = 2.0, bh = 2.55;
  cols.forEach((c, i) => {
    const x = x0 + i * (bw + gap);
    s.addText([
      { text: c[0], options: { fontFace: SANS, fontSize: 17, bold: true, color: c[4], breakLine: true, paraSpaceAfter: 9 } },
      { text: c[1], options: { fontFace: SANS, fontSize: 14, color: c[5] } },
    ], { shape: pres.shapes.ROUNDED_RECTANGLE, x, y: y0, w: bw, h: bh, rectRadius: 0.08, fill: { color: c[2] }, line: { color: c[3], width: 1 }, shadow: sh({ opacity: 0.10 }), valign: "top", margin: [16, 14, 12, 18] });
    if (i < cols.length - 1) s.addText("→", { x: x + bw - 0.02, y: y0, w: gap + 0.04, h: bh, fontFace: SANS, fontSize: 24, bold: true, color: TEAL, align: "center", valign: "middle", margin: 0 });
  });
  codeBox(s,
`void model(int N, int C,                /* dynamic */
           const float x[N][C],
           float       y[N][C]) {
  float tmp[N][C];          /* local VLA */
  for (int n = 0; n < N; ++n)
    for (int c = 0; c < C; ++c)
      tmp[n][c] = relu(x[n][c]);
}`,
    { x: 0.6, y: 4.8, w: 12.1, h: 1.85, fontSize: 13, caption: "Explicit array kept for a dynamic [N][C] model (incl. local temporary)" });
  originLine(s);
  footer(s);
}

/* =================== ALT 3 — memory-layout diagram ======================== */
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  title(s, "Explicit arrays: shape lives in the type");
  lead(s, "Same bytes in memory — the C type decides what the compiler and the reader can see.");
  // contiguous memory strip: a 2x3 tensor = 6 cells, row 0 teal, row 1 amber
  s.addText("contiguous memory (row-major) — a 2×3 tensor", { x: 3.4, y: 2.15, w: 6.6, h: 0.32, fontFace: SANS, fontSize: 12.5, italic: true, color: MUTED, margin: 0 });
  const cw = 0.92, ch = 0.8, cg = 0.08, sx = 3.4, sy = 2.55;
  for (let i = 0; i < 6; i++) {
    const x = sx + i * (cw + cg);
    s.addText("x[" + i + "]", { shape: pres.shapes.ROUNDED_RECTANGLE, x, y: sy, w: cw, h: ch, rectRadius: 0.05, fill: { color: i < 3 ? TEAL2 : AMBER }, line: { color: LIGHT, width: 1 }, fontFace: MONO, fontSize: 13, color: LIGHT, align: "center", valign: "middle", margin: 0 });
  }
  s.addText("row n=0", { x: sx, y: sy + ch + 0.08, w: cw * 3 + cg * 2, h: 0.3, fontFace: SANS, fontSize: 12, bold: true, color: TEAL2, align: "center", margin: 0 });
  s.addText("row n=1", { x: sx + 3 * (cw + cg), y: sy + ch + 0.08, w: cw * 3 + cg * 2, h: 0.3, fontFace: SANS, fontSize: 12, bold: true, color: SANDTX, align: "center", margin: 0 });
  // two interpretations of the SAME memory
  panel(s, 0.6, 4.35, 5.95, 1.95, "Flat   float *x", [
    { text: "x[n*3 + c]", options: { fontFace: MONO, fontSize: 15, color: LIGHT, breakLine: true, paraSpaceAfter: 8 } },
    { text: "You compute the offset by hand. The shape (2×3) exists only as a convention; an out-of-bounds index is invisible to the compiler.", options: { fontFace: SANS, fontSize: 13.5, color: "F2D9D9" } },
  ], { fill: RUSTTX, headColor: LIGHT, headSize: 16 });
  panel(s, 6.75, 4.35, 5.95, 1.95, "Explicit   float x[2][3]  /  x[N][C]", [
    { text: "x[n][c]", options: { fontFace: MONO, fontSize: 15, color: LIGHT, breakLine: true, paraSpaceAfter: 8 } },
    { text: "Rank & extents live in the type; indexing mirrors the tensor and bounds are part of the object. Dynamic models keep this via C99 VLAs (x[N][C]).", options: { fontFace: SANS, fontSize: 13.5, color: "DDEFD6" } },
  ], { fill: GREENTX, headColor: LIGHT, headSize: 16 });
  originLine(s);
  footer(s);
}

pres.writeFile({ fileName: process.env.OUT || "dynamic-model-alternatives-v2.pptx" }).then((f) => console.log("WROTE", f, "alts:", pageNo));
