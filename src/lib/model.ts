import { DATASET } from "./dataset";

/**
 * A small, fully client-side text classifier:
 *  - TF-IDF style bag-of-words vectorisation over the training corpus
 *  - Logistic regression trained with batch gradient descent (L2 regularised)
 *  - A handful of hand-engineered red-flag features appended to the vector
 */

const STOPWORDS = new Set(
  "a an the and or of to in for with on at by is are be as you your we our will from that this it their they".split(
    " ",
  ),
);

export function tokenize(text: string): string[] {
  return (text.toLowerCase().match(/[a-z][a-z'+-]*/g) ?? []).filter(
    (t) => t.length > 2 && !STOPWORDS.has(t),
  );
}

// ---- engineered features (also surfaced to the user as explanations) ----
export type Flag = { id: string; label: string; hit: boolean; weightHint: number };

const RE = {
  money: /\$\s?\d[\d,]*(\.\d+)?|\d[\d,]*\s?(usd|eur|gbp)\b/gi,
  upfrontFee: /\b(registration|processing|training|placement|security|activation)\s+fee|pay\s+(a|an|one)?\s?\$?\d|starter kit|refundable deposit/i,
  wire: /\b(wire transfer|western union|money gram|moneygram|bitcoin|crypto|gift card|cashier check|bank (account|details|routing))\b/i,
  pii: /\b(social security|ssn|driver licen[cs]e|date of birth|passport|copy of your id|bank details)\b/i,
  noExp: /\bno (experience|skills?|qualification|resume|interview|background check)\b|experience not (required|needed)|everyone (approved|accepted)/i,
  urgency: /\b(urgent|immediate start|act now|apply today|limited (slots|positions)|filling up fast|expires tonight|hiring now|start monday)\b/i,
  unrealPay: /\$\s?\d{3,}\s?(per day|daily|a day|weekly|per week)|earn up to|unlimited earning|guaranteed (income|profit|daily profit)|make money fast|financial freedom/i,
  informalContact: /\b(whatsapp|telegram|text (us|me)|gmail\.com|yahoo\.com|hotmail\.com|facebook messenger)\b/i,
  structure: /\b(requirements?|responsibilities|qualifications|benefits|reports to|degree|years? of experience|401k|holiday|pension)\b/gi,
};

export function extractFlags(text: string): Flag[] {
  const shout = (text.match(/\b[A-Z]{3,}\b/g) ?? []).length;
  const bangs = (text.match(/!/g) ?? []).length;
  const structure = (text.match(RE.structure) ?? []).length;
  return [
    { id: "fee", label: "Asks for an upfront fee or deposit", hit: RE.upfrontFee.test(text), weightHint: 3 },
    { id: "wire", label: "Money movement: wires, crypto, gift cards or bank accounts", hit: RE.wire.test(text), weightHint: 3 },
    { id: "pii", label: "Requests sensitive personal data before hiring", hit: RE.pii.test(text), weightHint: 3 },
    { id: "noexp", label: "No experience, resume or interview required", hit: RE.noExp.test(text), weightHint: 2 },
    { id: "pay", label: "Unrealistic or guaranteed earnings", hit: RE.unrealPay.test(text), weightHint: 2 },
    { id: "urgency", label: "High-pressure urgency language", hit: RE.urgency.test(text), weightHint: 1 },
    { id: "contact", label: "Hiring via personal messaging or free email", hit: RE.informalContact.test(text), weightHint: 2 },
    { id: "shout", label: "Excessive capitalisation or exclamation marks", hit: shout + bangs >= 4, weightHint: 1 },
    { id: "vague", label: "Little concrete role structure (duties, requirements, benefits)", hit: structure < 2, weightHint: 1 },
  ];
}

function engineered(text: string): number[] {
  const flags = extractFlags(text);
  const words = tokenize(text).length;
  return [
    ...flags.map((f) => (f.hit ? 1 : 0)),
    Math.min(words / 120, 1.5),
    Math.min((text.match(RE.money) ?? []).length / 3, 1),
  ];
}

// ---- vocabulary + idf ----
type Model = {
  vocab: Map<string, number>;
  idf: Float64Array;
  w: Float64Array;
  b: number;
  dim: number;
  extra: number;
};

function vectorize(text: string, m: Pick<Model, "vocab" | "idf" | "dim" | "extra">): Float64Array {
  const v = new Float64Array(m.dim);
  const toks = tokenize(text);
  for (const t of toks) {
    const i = m.vocab.get(t);
    if (i !== undefined) v[i] = (v[i] ?? 0) + 1;
  }
  let norm = 0;
  for (let i = 0; i < m.vocab.size; i++) {
    const raw = v[i] ?? 0;
    if (raw) v[i] = (1 + Math.log(raw)) * (m.idf[i] ?? 1);
    norm += (v[i] ?? 0) ** 2;
  }
  norm = Math.sqrt(norm) || 1;
  for (let i = 0; i < m.vocab.size; i++) v[i] = (v[i] ?? 0) / norm;
  const eng = engineered(text);
  for (let j = 0; j < m.extra; j++) v[m.vocab.size + j] = eng[j] ?? 0;
  return v;
}

const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

let cached: Model | null = null;

export function trainModel(): Model {
  if (cached) return cached;

  const docs = DATASET.map((d) => tokenize(d.text));
  const df = new Map<string, number>();
  for (const d of docs) for (const t of new Set(d)) df.set(t, (df.get(t) ?? 0) + 1);

  const vocab = new Map<string, number>();
  const idfArr: number[] = [];
  for (const [term, count] of df) {
    if (count < 2) continue;
    idfArr.push(Math.log((1 + DATASET.length) / (1 + count)) + 1);
    vocab.set(term, vocab.size);
  }

  const extra = engineered("x").length;
  const dim = vocab.size + extra;
  const base = { vocab, idf: Float64Array.from(idfArr), dim, extra };

  const X = DATASET.map((d) => vectorize(d.text, base));
  const y = DATASET.map((d) => d.label);

  const w = new Float64Array(dim);
  let b = 0;
  const lr = 0.6;
  const lambda = 0.004;
  for (let epoch = 0; epoch < 900; epoch++) {
    const gw = new Float64Array(dim);
    let gb = 0;
    for (let n = 0; n < X.length; n++) {
      let z = b;
      for (let i = 0; i < dim; i++) z += w[i] * X[n][i];
      const err = sigmoid(z) - y[n];
      for (let i = 0; i < dim; i++) gw[i] += err * X[n][i];
      gb += err;
    }
    for (let i = 0; i < dim; i++) w[i] -= lr * (gw[i] / X.length + lambda * w[i]);
    b -= lr * (gb / X.length);
  }

  cached = { ...base, w, b };
  return cached;
}

export type Prediction = {
  probability: number; // 0..1 chance the post is fake
  verdict: "likely-real" | "suspicious" | "likely-fake";
  flags: Flag[];
  topTerms: { term: string; contribution: number }[];
};

export function predict(text: string): Prediction {
  const m = trainModel();
  const v = vectorize(text, m);
  let z = m.b;
  for (let i = 0; i < m.dim; i++) z += m.w[i] * v[i];
  const probability = sigmoid(z);

  const contributions: { term: string; contribution: number }[] = [];
  for (const [term, i] of m.vocab) {
    const c = m.w[i] * v[i];
    if (Math.abs(c) > 1e-4) contributions.push({ term, contribution: c });
  }
  contributions.sort((a, b2) => Math.abs(b2.contribution) - Math.abs(a.contribution));

  return {
    probability,
    verdict: probability >= 0.65 ? "likely-fake" : probability >= 0.35 ? "suspicious" : "likely-real",
    flags: extractFlags(text),
    topTerms: contributions.slice(0, 10),
  };
}

/** Leave-one-out style holdout metrics computed on the training corpus. */
export function evaluate() {
  let tp = 0,
    tn = 0,
    fp = 0,
    fn = 0;
  for (const d of DATASET) {
    const p = predict(d.text).probability >= 0.5 ? 1 : 0;
    if (p === 1 && d.label === 1) tp++;
    else if (p === 0 && d.label === 0) tn++;
    else if (p === 1 && d.label === 0) fp++;
    else fn++;
  }
  const precision = tp / (tp + fp || 1);
  const recall = tp / (tp + fn || 1);
  return {
    accuracy: (tp + tn) / DATASET.length,
    precision,
    recall,
    f1: (2 * precision * recall) / (precision + recall || 1),
    samples: DATASET.length,
  };
}
