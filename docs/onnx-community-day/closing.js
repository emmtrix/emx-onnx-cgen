// Standalone: a closing slide for 260626_ONNX_Community_onnxcgen_v3.pptx.
// Matches the master-deck palette and fonts. Output → a separate file.
const pptxgen = require("pptxgenjs");
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.title = "ONNX Community Day — closing";
const W = 13.333, H = 7.5;

const INK = "0E3F66", TEAL = "126AAA", TEAL2 = "0E537F", AMBER = "F18926";
const SLATE = "454545", MUTED = "8A8A8A", LIGHT = "FFFFFF", CARD = "F1F5F9", CARDLN = "D9E3EA";
const GREEN = "EEF6E8", GREENLN = "D7E9C9", GREENTX = "4E8A2E";
const SERIF = "Rajdhani Semibold", SANS = "Arial", MONO = "Consolas";
const LOGO = "assets/emmtrix-logo.png";

const sh = (o = {}) => Object.assign({ type: "outer", color: "0A2E4A", blur: 9, offset: 3, angle: 90, opacity: 0.14 }, o);

function card(slide, x, y, w, h, header, bullets, opt = {}) {
  const runs = [];
  if (header) runs.push({ text: header, options: { fontFace: SANS, fontSize: opt.headSize || 16, bold: true, color: opt.headColor || TEAL2, breakLine: true, paraSpaceAfter: 10 } });
  bullets.forEach((b) => runs.push({ text: b, options: { bullet: { code: "2022", indent: 14 }, fontFace: SANS, fontSize: opt.fontSize || 13.5, color: opt.textColor || SLATE, breakLine: true, paraSpaceAfter: opt.gap != null ? opt.gap : 8 } }));
  slide.addText(runs, { shape: pres.shapes.ROUNDED_RECTANGLE, x, y, w, h, rectRadius: 0.08, fill: { color: opt.fill || CARD }, line: { color: opt.line || CARDLN, width: 1 }, shadow: sh({ opacity: 0.10 }), align: "left", valign: "top", margin: [16, 14, 12, 18] });
}

const s = pres.addSlide();
s.background = { color: LIGHT };
s.addText("Takeaways & discussion", { x: 0.6, y: 0.58, w: 12.1, h: 0.9, fontFace: SERIF, fontSize: 30, bold: true, color: TEAL, margin: 0, valign: "middle" });
s.addText("ONNX gave us a common model language — building emx-onnx-cgen showed what an embedded AOT backend still needs to turn it into deterministic C.", {
  x: 0.6, y: 1.46, w: 12.1, h: 0.5, fontFace: SANS, fontSize: 14.5, italic: true, color: SLATE, margin: 0, valign: "middle" });

card(s, 0.6, 2.1, 6.0, 4.0, "What an AOT-friendly ONNX could add", [
  "Bounded sizes in the type system — per type / per tensor",
  "A shared, extensible shape/type inference layer",
  "Clearer numerical contracts — order & precision, not just fixed rtol/atol",
  "An operator-importance signal — categories + a usage atlas",
  "Generated-code quality as a backend feature",
], { fill: CARD, line: CARDLN, headColor: TEAL2, gap: 9 });

card(s, 6.7, 2.1, 6.0, 4.0, "Discussion", [
  "What should an AOT-friendly ONNX profile specify?",
  "How to persist inferred shapes/types and size bounds in the model?",
  "Per-operator numerical guidance (summation order, accumulation)?",
  "A real-world operator usage atlas — e.g. index ~40k models on Hugging Face?",
], { fill: GREEN, line: GREENLN, headColor: GREENTX, gap: 11 });

// closing line + link (soft amber accent, no dark block)
s.addShape(pres.shapes.LINE, { x: 0.6, y: 6.35, w: 3.2, h: 0, line: { color: AMBER, width: 2 } });
s.addText([
  { text: "Thank you", options: { fontFace: SERIF, fontSize: 22, bold: true, color: TEAL } },
  { text: "    emx-onnx-cgen is open source  ·  ", options: { fontFace: SANS, fontSize: 14, color: SLATE } },
  { text: "github.com/emmtrix/emx-onnx-cgen", options: { fontFace: MONO, fontSize: 14, bold: true, color: TEAL2 } },
], { x: 0.6, y: 6.5, w: 11.0, h: 0.5, margin: 0, valign: "middle" });
s.addImage({ path: LOGO, x: W - 1.85, y: 6.55, w: 1.35, h: 0.36 });

pres.writeFile({ fileName: process.env.OUT || "260626_ONNX_Community_onnxcgen_closing.pptx" }).then((f) => console.log("WROTE", f));
