import { createFileRoute } from "@tanstack/react-router";
import { useMemo, useState } from "react";
import { useServerFn } from "@tanstack/react-start";
import { AlertTriangle, ShieldCheck, ShieldAlert, Sparkles, Gauge, Link2, Loader2 } from "lucide-react";
import { predict, evaluate } from "@/lib/model";
import { DATASET } from "@/lib/dataset";
import { fetchPosting } from "@/lib/fetch-url.functions";
import { analyzeUrl, isTrustedBoard, type UrlFlag } from "@/lib/url-flags";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Fake Job Post Detector — ML Scam Screening" },
      {
        name: "description",
        content:
          "Paste any job advert and a browser-side machine learning model scores how likely it is to be a scam, with the exact red flags and keywords behind the verdict.",
      },
      { property: "og:title", content: "Fake Job Post Detector — ML Scam Screening" },
      {
        property: "og:description",
        content:
          "TF-IDF + logistic regression trained in your browser to flag fraudulent job postings in seconds.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Index,
});

const VERDICTS = {
  "likely-real": {
    title: "Likely genuine",
    tone: "text-success",
    ring: "border-success/40 bg-success/10",
    Icon: ShieldCheck,
    blurb: "Reads like a normal posting. Still verify the employer independently.",
  },
  suspicious: {
    title: "Suspicious",
    tone: "text-warning",
    ring: "border-warning/40 bg-warning/10",
    Icon: AlertTriangle,
    blurb: "Mixed signals. Check the company domain and never send money or ID.",
  },
  "likely-fake": {
    title: "Likely fake",
    tone: "text-danger",
    ring: "border-danger/40 bg-danger/10",
    Icon: ShieldAlert,
    blurb: "Strong scam patterns detected. Do not pay fees or share bank details.",
  },
} as const;

function Index() {
  const [text, setText] = useState("");
  const metrics = useMemo(() => evaluate(), []);
  const result = useMemo(() => (text.trim().length > 25 ? predict(text) : null), [text]);

  const samples = useMemo(
    () => [DATASET[0]!.text, DATASET[DATASET.length - 3]!.text, DATASET[16]!.text],
    [],
  );

  return (
    <main className="min-h-screen" style={{ backgroundImage: "var(--gradient-hero)" }}>
      <div className="mx-auto w-full max-w-6xl px-5 py-14 md:py-20">
        <header className="max-w-3xl">
          <span className="inline-flex items-center gap-2 rounded-full border border-border bg-card/70 px-3 py-1 text-xs uppercase tracking-[0.18em] text-muted-foreground">
            <Sparkles className="h-3.5 w-3.5 text-primary" />
            TF-IDF + logistic regression, trained in your browser
          </span>
          <h1 className="mt-6 text-4xl leading-tight font-semibold md:text-6xl">
            Fake job post detection
          </h1>
          <p className="mt-4 text-base text-muted-foreground md:text-lg">
            Paste a job advert. The model scores the language against a labelled corpus of genuine
            and fraudulent postings, then shows the red flags and keywords that drove the decision.
          </p>
        </header>

        <section className="mt-10 grid gap-6 lg:grid-cols-[1.15fr_1fr]">
          <div className="rounded-xl border border-border bg-card/80 p-5 backdrop-blur" style={{ boxShadow: "var(--shadow-panel)" }}>
            <label htmlFor="post" className="text-sm font-medium text-foreground">
              Job posting text
            </label>
            <textarea
              id="post"
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Paste the full advert here — title, description, requirements, contact details…"
              className="mt-3 h-72 w-full resize-none rounded-lg border border-input bg-background/60 p-4 font-mono text-sm text-foreground outline-none placeholder:text-muted-foreground focus:border-ring focus:ring-2 focus:ring-ring/30"
            />
            <div className="mt-3 flex flex-wrap items-center gap-2 text-xs">
              <span className="text-muted-foreground">Try an example:</span>
              {["Real listing", "Fee scam", "Payment scam"].map((label, i) => (
                <button
                  key={label}
                  onClick={() => setText(samples[i] ?? "")}
                  className="rounded-md border border-border bg-secondary px-2.5 py-1 font-medium text-secondary-foreground transition-colors hover:border-primary/60 hover:text-primary"
                >
                  {label}
                </button>
              ))}
              {text && (
                <button
                  onClick={() => setText("")}
                  className="rounded-md px-2.5 py-1 text-muted-foreground hover:text-foreground"
                >
                  Clear
                </button>
              )}
            </div>
          </div>

          <div className="space-y-4">
            <ResultPanel result={result} />
            <div className="rounded-xl border border-border bg-card/60 p-5">
              <h2 className="flex items-center gap-2 text-sm font-semibold uppercase tracking-wider text-muted-foreground">
                <Gauge className="h-4 w-4 text-primary" /> Model performance
              </h2>
              <dl className="mt-4 grid grid-cols-2 gap-3 text-sm">
                {[
                  ["Accuracy", metrics.accuracy],
                  ["Precision", metrics.precision],
                  ["Recall", metrics.recall],
                  ["F1 score", metrics.f1],
                ].map(([k, v]) => (
                  <div key={k as string} className="rounded-lg border border-border/70 bg-background/40 p-3">
                    <dt className="text-xs text-muted-foreground">{k as string}</dt>
                    <dd className="font-display text-xl text-foreground">
                      {((v as number) * 100).toFixed(1)}%
                    </dd>
                  </div>
                ))}
              </dl>
              <p className="mt-3 text-xs text-muted-foreground">
                Measured on the {metrics.samples}-posting labelled corpus bundled with the app.
              </p>
            </div>
          </div>
        </section>

        <section className="mt-14 grid gap-4 md:grid-cols-3">
          {[
            {
              h: "Vectorisation",
              p: "Text is tokenised, stop-worded and turned into L2-normalised TF-IDF weights over the corpus vocabulary.",
            },
            {
              h: "Engineered signals",
              p: "Nine fraud heuristics — upfront fees, wire transfers, PII requests, urgency, unrealistic pay — join the feature vector.",
            },
            {
              h: "Classifier",
              p: "L2-regularised logistic regression trained by batch gradient descent, entirely client side. No data leaves the browser.",
            },
          ].map((c) => (
            <article key={c.h} className="rounded-xl border border-border bg-card/50 p-5">
              <h3 className="text-base font-semibold text-foreground">{c.h}</h3>
              <p className="mt-2 text-sm text-muted-foreground">{c.p}</p>
            </article>
          ))}
        </section>

        <footer className="mt-12 border-t border-border pt-6 text-xs text-muted-foreground">
          Screening aid only — a low score is not a guarantee that a posting is legitimate.
        </footer>
      </div>
    </main>
  );
}

function ResultPanel({ result }: { result: ReturnType<typeof predict> | null }) {
  if (!result) {
    return (
      <div className="flex h-56 items-center justify-center rounded-xl border border-dashed border-border bg-card/40 p-6 text-center text-sm text-muted-foreground">
        Paste at least a couple of sentences to run the classifier.
      </div>
    );
  }

  const v = VERDICTS[result.verdict];
  const pct = Math.round(result.probability * 100);
  const hits = result.flags.filter((f) => f.hit);

  return (
    <div className={`rounded-xl border p-5 ${v.ring}`}>
      <div className="flex items-start gap-3">
        <v.Icon className={`mt-0.5 h-6 w-6 ${v.tone}`} />
        <div className="flex-1">
          <div className="flex items-baseline justify-between gap-3">
            <h2 className={`text-xl font-semibold ${v.tone}`}>{v.title}</h2>
            <span className="font-display text-2xl text-foreground">{pct}%</span>
          </div>
          <p className="mt-1 text-sm text-muted-foreground">{v.blurb}</p>
        </div>
      </div>

      <div className="mt-4 h-2 w-full overflow-hidden rounded-full bg-background/60">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{
            width: `${pct}%`,
            background: "linear-gradient(90deg, var(--success), var(--warning), var(--danger))",
          }}
        />
      </div>
      <p className="mt-1 text-xs text-muted-foreground">Estimated probability the post is fraudulent</p>

      <div className="mt-5">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          Red flags {hits.length > 0 && `(${hits.length})`}
        </h3>
        {hits.length === 0 ? (
          <p className="mt-2 text-sm text-muted-foreground">No heuristic red flags triggered.</p>
        ) : (
          <ul className="mt-2 space-y-1.5">
            {hits.map((f) => (
              <li key={f.id} className="flex items-start gap-2 text-sm text-foreground">
                <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-danger" />
                {f.label}
              </li>
            ))}
          </ul>
        )}
      </div>

      {result.topTerms.length > 0 && (
        <div className="mt-5">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Most influential terms
          </h3>
          <div className="mt-2 flex flex-wrap gap-1.5">
            {result.topTerms.map((t) => (
              <span
                key={t.term}
                className={`rounded-md border px-2 py-0.5 font-mono text-xs ${
                  t.contribution > 0
                    ? "border-danger/40 bg-danger/10 text-danger"
                    : "border-success/40 bg-success/10 text-success"
                }`}
                title={t.contribution > 0 ? "pushes toward fake" : "pushes toward genuine"}
              >
                {t.term}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
