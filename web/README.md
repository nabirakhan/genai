# web/ (React frontend, Vercel)

Vite + React + TypeScript skeleton: typed API client (`src/api.ts`, `src/types.ts`) and three
unstyled routes (Generate, Classify, Results) that prove the wiring against `api/`. The visual
design is done last; everything in `src/pages/` can be rewritten freely as long as it uses `api.ts`.

```bash
cd web && npm install
echo "VITE_API_URL=http://localhost:8000" > .env.local   # or the HF Space URL
npm run dev
```

Deploy: Vercel project with root directory `web/`, build `npm run build`, output `dist/`,
env `VITE_API_URL=https://<space>.hf.space`. `vercel.json` rewrites all routes to the SPA.
`public/results/` (tables.json + figures) is written by `src/analysis/aggregate.py` and `plots.py`
and committed, so the Results page needs no backend.
