// Standalone: a minimal "Thank you" slide for 260626_ONNX_Community_onnxcgen_v3.pptx.
const pptxgen = require("pptxgenjs");
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.title = "ONNX Community Day — thank you";
const W = 13.333, H = 7.5;

const INK = "0E3F66", TEAL = "126AAA", TEAL2 = "0E537F", AMBER = "F18926";
const SLATE = "454545", MUTED = "8A8A8A", LIGHT = "FFFFFF";
const SERIF = "Rajdhani Semibold", SANS = "Arial", MONO = "Consolas";
const LOGO = "assets/emmtrix-logo.png";

const s = pres.addSlide();
s.background = { color: LIGHT };

s.addImage({ path: LOGO, x: 0.85, y: 1.5, w: 2.3, h: 0.61 });

s.addText("Thank you", { x: 0.8, y: 2.7, w: 11.6, h: 1.6, fontFace: SERIF, fontSize: 64, bold: true, color: INK, margin: 0, valign: "middle" });

s.addText("emx-onnx-cgen — deterministic, portable C for embedded and resource-constrained systems.", {
  x: 0.87, y: 4.25, w: 11.2, h: 0.5, fontFace: SANS, fontSize: 18, color: SLATE, margin: 0 });

s.addShape(pres.shapes.LINE, { x: 0.9, y: 5.15, w: 3.4, h: 0, line: { color: AMBER, width: 2 } });

s.addText([
  { text: "Open source  ·  ", options: { fontFace: SANS, fontSize: 16, color: MUTED } },
  { text: "github.com/emmtrix/emx-onnx-cgen", options: { fontFace: MONO, fontSize: 16, bold: true, color: TEAL2 } },
], { x: 0.9, y: 5.35, w: 11, h: 0.5, margin: 0, valign: "middle" });

s.addText("ONNX Community Day  ·  emmtrix Technologies", { x: 0.9, y: 6.6, w: 11, h: 0.4, fontFace: SANS, fontSize: 13, color: MUTED, margin: 0 });

pres.writeFile({ fileName: process.env.OUT || "260626_ONNX_Community_onnxcgen_thankyou.pptx" }).then((f) => console.log("WROTE", f));
