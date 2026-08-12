import { createServerFn } from "@tanstack/react-start";
import { z } from "zod";

const schema = z.object({ url: z.string().url() });

function stripHtml(html: string): string {
  return html
    .replace(/<script[\s\S]*?<\/script>/gi, " ")
    .replace(/<style[\s\S]*?<\/style>/gi, " ")
    .replace(/<noscript[\s\S]*?<\/noscript>/gi, " ")
    .replace(/<\/(p|div|li|br|h[1-6]|tr)>/gi, "\n")
    .replace(/<[^>]+>/g, " ")
    .replace(/&nbsp;/gi, " ")
    .replace(/&amp;/gi, "&")
    .replace(/&quot;/gi, '"')
    .replace(/&#39;/gi, "'")
    .replace(/&lt;/gi, "<")
    .replace(/&gt;/gi, ">")
    .replace(/[ \t]+/g, " ")
    .replace(/\n\s*\n\s*/g, "\n")
    .trim();
}

export type FetchedPosting = {
  url: string;
  title: string;
  text: string;
};

export const fetchPosting = createServerFn({ method: "POST" })
  .inputValidator((data: unknown) => schema.parse(data))
  .handler(async ({ data }): Promise<FetchedPosting> => {
    const target = new URL(data.url);
    if (target.protocol !== "http:" && target.protocol !== "https:") {
      throw new Error("Only http and https links are supported.");
    }

    const res = await fetch(target.toString(), {
      headers: {
        "user-agent":
          "Mozilla/5.0 (compatible; JobPostScanner/1.0; +https://lovable.dev)",
        accept: "text/html,application/xhtml+xml",
      },
      redirect: "follow",
    });

    if (!res.ok) {
      throw new Error(`The page could not be fetched (HTTP ${res.status}).`);
    }

    const html = (await res.text()).slice(0, 500_000);
    const title = /<title[^>]*>([\s\S]*?)<\/title>/i.exec(html)?.[1]?.trim() ?? target.hostname;
    const text = stripHtml(html);

    if (text.length < 80) {
      throw new Error(
        "Not enough readable text on that page — it may require JavaScript or a login. Paste the advert text instead.",
      );
    }

    return { url: target.toString(), title: stripHtml(title), text: text.slice(0, 20_000) };
  });
