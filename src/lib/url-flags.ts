export type UrlFlag = { id: string; label: string; hit: boolean };

const SHORTENERS = ["bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "cutt.ly", "rb.gy", "shorturl.at"];
const FREE_HOSTS = [
  "wixsite.com",
  "weebly.com",
  "blogspot.com",
  "wordpress.com",
  "webnode",
  "godaddysites.com",
  "sites.google.com",
  "github.io",
  "netlify.app",
  "glitch.me",
];
const TRUSTED = [
  "linkedin.com",
  "indeed.com",
  "glassdoor.com",
  "greenhouse.io",
  "lever.co",
  "workday.com",
  "ashbyhq.com",
  "smartrecruiters.com",
  "workable.com",
  "monster.com",
  "ziprecruiter.com",
];
const RISKY_TLDS = [".xyz", ".top", ".click", ".buzz", ".work", ".icu", ".tk", ".ml", ".cf", ".gq", ".rest"];

/** Heuristics derived from the link itself, complementing the text classifier. */
export function analyzeUrl(raw: string): { host: string; flags: UrlFlag[] } {
  let host = "";
  try {
    host = new URL(raw).hostname.toLowerCase().replace(/^www\./, "");
  } catch {
    return { host: raw, flags: [] };
  }

  const insecure = raw.startsWith("http://");
  const flags: UrlFlag[] = [
    { id: "shortener", label: "Link is hidden behind a URL shortener", hit: SHORTENERS.some((d) => host === d || host.endsWith(`.${d}`)) },
    { id: "freehost", label: "Hosted on a free website builder rather than a company domain", hit: FREE_HOSTS.some((d) => host.includes(d)) },
    { id: "tld", label: "Uncommon top-level domain often used in scams", hit: RISKY_TLDS.some((t) => host.endsWith(t)) },
    { id: "insecure", label: "Page is served over plain HTTP, not HTTPS", hit: insecure },
    { id: "lookalike", label: "Domain imitates a known job board with extra words or hyphens", hit: TRUSTED.some((t) => host.includes(t.split(".")[0]!) && !host.endsWith(t)) },
    { id: "numeric", label: "Domain contains digits or unusually many hyphens", hit: /\d/.test(host.split(".")[0] ?? "") || (host.match(/-/g) ?? []).length >= 3 },
  ];

  return { host, flags };
}

export function isTrustedBoard(host: string): boolean {
  return TRUSTED.some((t) => host === t || host.endsWith(`.${t}`));
}
