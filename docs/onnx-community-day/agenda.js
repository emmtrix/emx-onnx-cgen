// Standalone: an overview / agenda slide for 260626_ONNX_Community_onnxcgen_v3.pptx.
// Matches the master-deck palette and fonts. Output → a separate file.
const pptxgen = require("pptxgenjs");
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.title = "ONNX Community Day — agenda";
const W = 13.333, H = 7.5;

const INK = "0E3F66", DEEP = "0A2E4A", TEAL = "126AAA", TEAL2 = "0E537F", AMBER = "F18926";
const SLATE = "454545", MUTED = "8A8A8A", LIGHT = "FFFFFF", CARD = "F1F5F9", CARDLN = "D9E3EA";
const GREEN = "EEF6E8", GREENLN = "D7E9C9", GREENTX = "4E8A2E";
const SERIF = "Rajdhani Semibold", SANS = "Arial", MONO = "Consolas";
const LOGO = "assets/emmtrix-logo.png";

const sh = (o = {}) => Object.assign({ type: "outer", color: "0A2E4A", blur: 9, offset: 3, angle: 90, opacity: 0.14 }, o);

function footer(slide, n) {
  slide.addShape(pres.shapes.LINE, { x: 0.5, y: 6.94, w: 12.33, h: 0, line: { color: TEAL, width: 1.25 } });
  slide.addImage({ path: LOGO, x: 0.5, y: 7.04, w: 1.2, h: 0.32 });
  slide.addText("ONNX Community Day  ·  emx-onnx-cgen", { x: 1.95, y: 7.05, w: 8, h: 0.3, fontFace: SANS, fontSize: 10.5, color: MUTED, align: "left", valign: "middle", margin: 0 });
  slide.addText(String(n), { x: W - 1.1, y: 7.05, w: 0.6, h: 0.3, fontFace: SANS, fontSize: 12, bold: true, color: TEAL2, align: "right", valign: "middle", margin: 0 });
}

const s = pres.addSlide();
s.background = { color: LIGHT };
s.addText("Agenda", { x: 0.6, y: 0.58, w: 12.1, h: 0.9, fontFace: SERIF, fontSize: 30, bold: true, color: TEAL, margin: 0, valign: "middle" });
s.addText("Lessons learned building an AOT ONNX-to-C compiler with 100% test coverage.", {
  x: 0.6, y: 1.46, w: 12.1, h: 0.4, fontFace: SANS, fontSize: 14.5, color: SLATE, margin: 0, valign: "middle" });

const cols = [
  {
    x: 0.6, num: "01", head: "The project & approach", fill: CARD, line: CARDLN, headColor: TEAL2, numColor: TEAL, textColor: SLATE,
    items: [
      "What emx-onnx-cgen is — goals & generated C",
      "C as our IR — not MLIR",
      "Explicit typed arrays — even for dynamic models",
      "Data types: complete coverage",
    ],
  },
  {
    x: 4.73, num: "02", head: "Five ONNX lessons learned", fill: "FBF3E8", line: "F0E2CC", headColor: "C06A18", numColor: AMBER, textColor: SLATE,
    items: [
      "1 · ONNX is effectively unbounded",
      "2 · Sequences: element type underspecified",
      "3 · Type & shape inference is not enough",
      "4 · Operator importance is unknown",
      "5 · Evaluation order & precision unspecified",
    ],
  },
  {
    x: 8.86, num: "03", head: "Status", fill: GREEN, line: GREENLN, headColor: GREENTX, numColor: GREENTX, textColor: SLATE,
    items: [
      "Where it stands today",
      "Opset & broad operator coverage",
      "ONNX Backend Scoreboard",
      "Reports: SUPPORT_OPS · ONNX_SUPPORT",
    ],
  },
];

const cw = 3.85, y0 = 2.05, ch = 4.55;
cols.forEach((c) => {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: c.x, y: y0, w: cw, h: ch, rectRadius: 0.08, fill: { color: c.fill }, line: { color: c.line, width: 1 }, shadow: sh({ opacity: 0.10 }) });
  s.addText(c.num, { x: c.x + 0.28, y: y0 + 0.25, w: cw - 0.5, h: 0.9, fontFace: SERIF, fontSize: 44, bold: true, color: c.numColor, margin: 0 });
  s.addText(c.head, { x: c.x + 0.3, y: y0 + 1.2, w: cw - 0.55, h: 0.7, fontFace: SANS, fontSize: 16.5, bold: true, color: c.headColor, margin: 0, valign: "top" });
  s.addText(c.items.map((t) => ({ text: t, options: { bullet: { code: "2022", indent: 13 }, color: c.textColor, breakLine: true, paraSpaceAfter: 8 } })), {
    x: c.x + 0.3, y: y0 + 2.0, w: cw - 0.55, h: ch - 2.2, fontFace: SANS, fontSize: 12.5, margin: 0, valign: "top" });
});

s.addText("Not a product pitch — the engineering lessons about ONNX as input to deterministic AOT compilation.", {
  x: 0.6, y: 6.7, w: 12.1, h: 0.35, fontFace: SANS, fontSize: 13, italic: true, color: MUTED, margin: 0 });
footer(s, 1);

pres.writeFile({ fileName: process.env.OUT || "260626_ONNX_Community_onnxcgen_agenda.pptx" }).then((f) => console.log("WROTE", f));
