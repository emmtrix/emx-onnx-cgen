// Alternatives for the "ONNX is effectively unbounded" slide.
// Each box pairs the ONNX problem with how emx-onnx-cgen handles it.
const pptxgen = require("pptxgenjs");
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.title = "Unbounded slide — alternatives";
const W = 13.333, H = 7.5;

const INK = "0E3F66", TEAL = "126AAA", TEAL2 = "0E537F", AMBER = "F18926";
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
  slide.addText("Unbounded slide — alternative " + pageNo, { x: 0.6, y: 7.05, w: 8, h: 0.3, fontFace: SANS, fontSize: 10.5, color: MUTED, margin: 0, valign: "middle" });
  slide.addText("ALT " + pageNo, { x: W - 1.3, y: 7.05, w: 0.8, h: 0.3, fontFace: SANS, fontSize: 12, bold: true, color: TEAL2, align: "right", valign: "middle", margin: 0 });
}
function badge(slide, x, y, label, d = 0.62) {
  slide.addText(label, { shape: pres.shapes.OVAL, x, y, w: d, h: d, fill: { color: TEAL }, shadow: sh({ opacity: 0.18 }), fontFace: SERIF, fontSize: 20, bold: true, color: LIGHT, align: "center", valign: "middle", margin: 0 });
}
function lessonHead(slide, num, text) {
  badge(slide, 0.6, 0.56, String(num));
  slide.addText(text, { x: 1.42, y: 0.56, w: 11.3, h: 0.66, fontFace: SERIF, fontSize: 29, bold: true, color: TEAL, margin: 0, valign: "middle" });
}
function lead(slide, text, y = 1.42) {
  slide.addText(text, { x: 0.6, y, w: 12.1, h: 0.4, fontFace: SANS, fontSize: 14, color: SLATE, margin: 0, valign: "middle" });
}

const DATA = [
  {
    cat: "Dynamic dimensions",
    problem: "Symbolic / unknown axes give a runtime extent — but no maximum.",
    emx: "C99 VLA x[N][C] carries the runtime extent; where memory needs a bound, require static shapes or fail clearly.",
    emxShort: "C99 VLA x[N][C]; static/bounded where memory needs it, else fail clearly.",
    code: "void model(int N, int C,\n  const float x[N][C],\n  float y[N][C]);",
  },
  {
    cat: "Sequence length",
    problem: "Containers have no natural capacity — one level above a dynamic dim.",
    emx: "Fixed-capacity array + count; ragged inputs declared via --sequence-element-shape.",
    emxShort: "fixed-capacity array + idx_t __count (32, overridable); ragged → --sequence-element-shape.",
    code: "T x[EMX_SEQUENCE_MAX_LEN][…];\nidx_t x__count;  /* 32 */",
  },
  {
    cat: "String size",
    problem: "String tensors carry no maximum character length.",
    emx: "Fixed '\\0'-terminated slots, default 256, #ifndef-overridable.",
    emxShort: "fixed char[…][EMX_STRING_MAX_LEN] slots, '\\0'-terminated (256, overridable).",
    code: "char x[…][EMX_STRING_MAX_LEN];\n/* '\\0'-terminated, 256 */",
  },
];

/* =================== ALT 1 — 3 columns, Problem + emx labelled in one box === */
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  lessonHead(s, 1, "ONNX is effectively unbounded");
  lead(s, "Each box: what ONNX leaves open, and how emx-onnx-cgen bounds it.");
  const bw = 3.95, gap = 0.28, x0 = 0.6, y0 = 1.95, bh = 4.7;
  DATA.forEach((d, i) => {
    const x = x0 + i * (bw + gap);
    s.addText([
      { text: d.cat, options: { fontFace: SANS, fontSize: 16.5, bold: true, color: TEAL2, breakLine: true, paraSpaceAfter: 11 } },
      { text: "PROBLEM", options: { fontFace: SANS, fontSize: 11, bold: true, color: SANDTX, charSpacing: 1, breakLine: true, paraSpaceAfter: 4 } },
      { text: d.problem, options: { fontFace: SANS, fontSize: 13, color: SLATE, breakLine: true, paraSpaceAfter: 14 } },
      { text: "emx-onnx-cgen", options: { fontFace: SANS, fontSize: 11, bold: true, color: GREENTX, charSpacing: 1, breakLine: true, paraSpaceAfter: 4 } },
      { text: d.emx, options: { fontFace: SANS, fontSize: 13, color: SLATE } },
    ], { shape: pres.shapes.ROUNDED_RECTANGLE, x, y: y0, w: bw, h: bh, rectRadius: 0.08, fill: { color: CARD }, line: { color: CARDLN, width: 1 }, shadow: sh({ opacity: 0.10 }), valign: "top", margin: [16, 14, 12, 18] });
  });
  footer(s);
}

/* =================== ALT 2 — full-width table: Category | Problem | emx ===== */
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  lessonHead(s, 1, "ONNX is effectively unbounded");
  lead(s, "Per case: the ONNX problem and how emx-onnx-cgen bounds it — side by side.");
  const hdr = [
    { text: "Case", options: { bold: true, color: LIGHT, fill: { color: INK } } },
    { text: "Problem (ONNX leaves it open)", options: { bold: true, color: LIGHT, fill: { color: INK } } },
    { text: "emx-onnx-cgen handles it", options: { bold: true, color: LIGHT, fill: { color: INK } } },
  ];
  const body = DATA.map((d, i) => [
    { text: d.cat, options: { fill: { color: i % 2 ? LIGHT : CARD }, color: TEAL2, bold: true, fontFace: SANS, fontSize: 13 } },
    { text: d.problem, options: { fill: { color: i % 2 ? LIGHT : CARD }, color: SLATE, fontFace: SANS, fontSize: 12.5 } },
    { text: d.emxShort, options: { fill: { color: GREEN }, color: SLATE, fontFace: SANS, fontSize: 12.5 } },
  ]);
  s.addTable([hdr, ...body], { x: 0.6, y: 2.05, w: 12.1, colW: [2.2, 4.6, 5.3], border: { pt: 0.5, color: CARDLN }, align: "left", valign: "middle", rowH: 1.25, margin: [6, 9, 6, 9] });
  s.addText("ONNX is intentionally flexible for interchange; embedded C needs concrete bounds — or a clear failure.", { x: 0.6, y: 6.62, w: 12.1, h: 0.3, fontFace: SANS, fontSize: 12.5, italic: true, color: MUTED, margin: 0 });
  footer(s);
}

/* =================== ALT 3 — 3 columns, problem text + emx code snippet ===== */
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  lessonHead(s, 1, "ONNX is effectively unbounded");
  lead(s, "Each box: the open problem, plus the concrete C that emx-onnx-cgen emits.");
  const bw = 3.95, gap = 0.28, x0 = 0.6, y0 = 1.95, bh = 4.7;
  DATA.forEach((d, i) => {
    const x = x0 + i * (bw + gap);
    // problem card (top half)
    s.addText([
      { text: d.cat, options: { fontFace: SANS, fontSize: 16, bold: true, color: TEAL2, breakLine: true, paraSpaceAfter: 8 } },
      { text: d.problem, options: { fontFace: SANS, fontSize: 13, color: SLATE } },
    ], { shape: pres.shapes.ROUNDED_RECTANGLE, x, y: y0, w: bw, h: 1.95, rectRadius: 0.08, fill: { color: SAND }, line: { color: SANDLN, width: 1 }, shadow: sh({ opacity: 0.09 }), valign: "top", margin: [14, 12, 10, 16] });
    // emx code (bottom half)
    s.addText([
      { text: "emx-onnx-cgen emits", options: { fontFace: SANS, fontSize: 11.5, bold: true, color: GREENTX, breakLine: true, paraSpaceAfter: 7 } },
      { text: d.code, options: { fontFace: MONO, fontSize: 12, color: CODEFG } },
    ], { shape: pres.shapes.ROUNDED_RECTANGLE, x, y: y0 + 2.1, w: bw, h: 2.6, rectRadius: 0.08, fill: { color: GREEN }, line: { color: GREENLN, width: 1 }, shadow: sh({ opacity: 0.08 }), valign: "top", margin: [12, 10, 10, 14], lineSpacingMultiple: 1.05 });
  });
  footer(s);
}

/* =================== ALT 4 — 6-box matrix: cols = cases, rows = problem/emx = */
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  lessonHead(s, 1, "ONNX is effectively unbounded");
  lead(s, "A 2×3 matrix: the open problem (top) and how emx-onnx-cgen bounds it (bottom).");
  const lx = 0.6, cx0 = 1.6, bw = 3.62, gap = 0.2, r1 = 2.55, r2 = 4.55, bh = 1.9;
  // row labels
  s.addText("PROBLEM", { x: lx, y: r1, w: 0.95, h: bh, fontFace: SANS, fontSize: 11.5, bold: true, color: SANDTX, align: "left", valign: "middle", margin: 0 });
  s.addText("emx-onnx-cgen", { x: lx, y: r2, w: 0.95, h: bh, fontFace: SANS, fontSize: 11.5, bold: true, color: GREENTX, align: "left", valign: "middle", margin: 0 });
  DATA.forEach((d, i) => {
    const x = cx0 + i * (bw + gap);
    s.addText(d.cat, { x, y: 1.95, w: bw, h: 0.45, fontFace: SANS, fontSize: 15, bold: true, color: TEAL2, align: "center", valign: "middle", margin: 0 });
    s.addText(d.problem, { shape: pres.shapes.ROUNDED_RECTANGLE, x, y: r1, w: bw, h: bh, rectRadius: 0.08, fill: { color: SAND }, line: { color: SANDLN, width: 1 }, shadow: sh({ opacity: 0.09 }), fontFace: SANS, fontSize: 13, color: SLATE, valign: "middle", margin: [12, 12, 12, 14] });
    s.addText([
      { text: d.emx, options: { fontFace: SANS, fontSize: 12.5, color: SLATE, breakLine: true, paraSpaceAfter: 7 } },
      { text: d.code.split("\n")[0], options: { fontFace: MONO, fontSize: 11, color: INK } },
    ], { shape: pres.shapes.ROUNDED_RECTANGLE, x, y: r2, w: bw, h: bh, rectRadius: 0.08, fill: { color: GREEN }, line: { color: GREENLN, width: 1 }, shadow: sh({ opacity: 0.09 }), valign: "top", margin: [11, 12, 10, 14] });
  });
  footer(s);
}

/* =================== ALT 5 — 6 boxes, rows = cases, Problem → emx ============ */
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  lessonHead(s, 1, "ONNX is effectively unbounded");
  lead(s, "One row per case: the ONNX problem on the left, the emx-onnx-cgen answer on the right.");
  const y0 = 1.95, rh = 1.5, rg = 0.12;
  DATA.forEach((d, i) => {
    const y = y0 + i * (rh + rg);
    s.addText([
      { text: d.cat, options: { fontFace: SANS, fontSize: 14.5, bold: true, color: TEAL2, breakLine: true, paraSpaceAfter: 6 } },
      { text: d.problem, options: { fontFace: SANS, fontSize: 13, color: SLATE } },
    ], { shape: pres.shapes.ROUNDED_RECTANGLE, x: 0.6, y, w: 5.85, h: rh, rectRadius: 0.07, fill: { color: SAND }, line: { color: SANDLN, width: 1 }, shadow: sh({ opacity: 0.08 }), valign: "middle", margin: [10, 12, 10, 16] });
    s.addText("→", { x: 6.48, y, w: 0.3, h: rh, fontFace: SANS, fontSize: 22, bold: true, color: TEAL, align: "center", valign: "middle", margin: 0 });
    s.addText([
      { text: "emx-onnx-cgen", options: { fontFace: SANS, fontSize: 11.5, bold: true, color: GREENTX, breakLine: true, paraSpaceAfter: 6 } },
      { text: d.emxShort, options: { fontFace: SANS, fontSize: 13, color: SLATE } },
    ], { shape: pres.shapes.ROUNDED_RECTANGLE, x: 6.85, y, w: 5.85, h: rh, rectRadius: 0.07, fill: { color: GREEN }, line: { color: GREENLN, width: 1 }, shadow: sh({ opacity: 0.08 }), valign: "middle", margin: [10, 12, 10, 16] });
  });
  footer(s);
}

pres.writeFile({ fileName: process.env.OUT || "unbounded-alternatives.pptx" }).then((f) => console.log("WROTE", f, "alts:", pageNo));
