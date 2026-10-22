import type { BankResponse, ClassifyResponse, GenerateRequest, GenerateResponse, Health, ResultsTables } from "./types";

export const API_URL = (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, "") ?? "http://localhost:8000";

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText}${text ? `: ${text}` : ""}`);
  }
  return (await res.json()) as T;
}

export const api = {
  health: () => fetch(`${API_URL}/health`).then(handle<Health>),
  generate: (body: GenerateRequest) =>
    fetch(`${API_URL}/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ nearest: true, ...body }),
    }).then(handle<GenerateResponse>),
  classify: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return fetch(`${API_URL}/classify`, { method: "POST", body: form }).then(handle<ClassifyResponse>);
  },
  bank: (grade?: number) =>
    fetch(`${API_URL}/bank${grade === undefined ? "" : `?grade=${grade}`}`).then(handle<BankResponse>),
  bankUrl: (file: string) => `${API_URL}/static/bank/${file}`,
  // Static results exported by src/analysis into public/results (no backend needed).
  results: () => fetch("/results/tables.json").then(handle<ResultsTables>),
};

export const b64png = (b64: string) => `data:image/png;base64,${b64}`;
