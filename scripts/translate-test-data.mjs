#!/usr/bin/env node

import fs from "node:fs/promises";
import path from "node:path";

const ROOT_DIR = process.cwd();
const TEST_DATA_DIR = path.join(ROOT_DIR, "test-data");
const OUTPUT_DIR = path.join(ROOT_DIR, "translated-test-data");

const DEFAULT_MODEL = process.env.HF_MODEL || "meta-llama/Llama-3.3-70B-Instruct";
const HF_TOKEN = process.env.HF_TOKEN;
const TRANSIENT_STATUS_CODES = new Set([408, 409, 425, 429, 500, 502, 503, 504]);

const TARGET_FILES = [
  "detect_all_symptom_mention_using_llm.json",
  "detect_followup_info_in_previous_user_response.csv",
  "detect_whether_patient_is_experiencing_symptom.csv",
];

const DEFAULT_LANGUAGES = [
  { key: "spanish", label: "Spanish" },
  { key: "french", label: "French" },
  { key: "german", label: "German" },
  { key: "polish", label: "Polish" },
  { key: "portuguese_brazilian", label: "Portuguese (Brazilian)" },
  { key: "turkish", label: "Turkish" },
  { key: "arabic", label: "Arabic" },
  { key: "chinese_mandarin", label: "Chinese Mandarin" },
  { key: "bengali", label: "Bengali" },
  { key: "urdu", label: "Urdu" }
];

const LANGUAGE_ALIASES = new Map(
  DEFAULT_LANGUAGES.flatMap((language) => [
    [language.key, language],
    [language.label.toLowerCase(), language],
  ]),
);

const TRANSLATABLE_FIELDS = {
  "detect_all_symptom_mention_using_llm.json": ["response"],
  "detect_followup_info_in_previous_user_response.csv": ["llm_question", "response"],
  "detect_whether_patient_is_experiencing_symptom.csv": ["llm_question", "response"],
};

function parseArgs(argv) {
  const args = {
    languages: DEFAULT_LANGUAGES,
    model: DEFAULT_MODEL,
    outputDir: OUTPUT_DIR,
    provider: process.env.HF_PROVIDER || "",
    overwrite: false,
    resume: true,
    delayMs: Number(process.env.HF_DELAY_MS || 0),
    maxTokens: Number(process.env.HF_MAX_TOKENS || 256),
    limit: null,
    maxRetries: Number(process.env.HF_MAX_RETRIES || 5),
    retryBaseMs: Number(process.env.HF_RETRY_BASE_MS || 5000),
  };

  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];

    if (token === "--languages" || token === "-l") {
      args.languages = resolveLanguages(argv[index + 1]);
      index += 1;
      continue;
    }

    if (token === "--model") {
      args.model = argv[index + 1];
      index += 1;
      continue;
    }

    if (token === "--provider") {
      args.provider = argv[index + 1];
      index += 1;
      continue;
    }

    if (token === "--output-dir") {
      args.outputDir = path.resolve(ROOT_DIR, argv[index + 1]);
      index += 1;
      continue;
    }

    if (token === "--overwrite") {
      args.overwrite = true;
      continue;
    }

    if (token === "--no-resume") {
      args.resume = false;
      continue;
    }

    if (token === "--delay-ms") {
      args.delayMs = Number(argv[index + 1]);
      index += 1;
      continue;
    }

    if (token === "--max-tokens") {
      args.maxTokens = Number(argv[index + 1]);
      index += 1;
      continue;
    }

    if (token === "--limit") {
      args.limit = Number(argv[index + 1]);
      index += 1;
      continue;
    }

    if (token === "--max-retries") {
      args.maxRetries = Number(argv[index + 1]);
      index += 1;
      continue;
    }

    if (token === "--retry-base-ms") {
      args.retryBaseMs = Number(argv[index + 1]);
      index += 1;
      continue;
    }

    throw new Error(`Unknown argument: ${token}`);
  }

  if (!Number.isFinite(args.delayMs) || args.delayMs < 0) {
    throw new Error("--delay-ms must be a non-negative number");
  }

  if (!Number.isFinite(args.maxTokens) || args.maxTokens <= 0) {
    throw new Error("--max-tokens must be a positive number");
  }

  if (args.limit !== null && (!Number.isFinite(args.limit) || args.limit <= 0)) {
    throw new Error("--limit must be a positive number");
  }

  if (!Number.isFinite(args.maxRetries) || args.maxRetries < 0) {
    throw new Error("--max-retries must be zero or greater");
  }

  if (!Number.isFinite(args.retryBaseMs) || args.retryBaseMs <= 0) {
    throw new Error("--retry-base-ms must be a positive number");
  }

  return args;
}

function resolveLanguages(rawValue) {
  if (!rawValue || rawValue.trim().toLowerCase() === "all") {
    return DEFAULT_LANGUAGES;
  }

  return rawValue
    .split(",")
    .map((value) => value.trim().toLowerCase())
    .filter(Boolean)
    .map((value) => {
      const language = LANGUAGE_ALIASES.get(value);
      if (!language) {
        throw new Error(`Unsupported language: ${value}`);
      }
      return language;
    });
}

function sanitizePathPart(value) {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 100);
}

function buildRunName(model, provider) {
  const modelPart = sanitizePathPart(model);
  const providerPart = provider ? `__${sanitizePathPart(provider)}` : "";
  return `${modelPart}${providerPart}`;
}

async function ensureDir(dirPath) {
  await fs.mkdir(dirPath, { recursive: true });
}

async function exists(filePath) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function readJson(filePath) {
  return JSON.parse(await fs.readFile(filePath, "utf8"));
}

async function writeJson(filePath, value) {
  await fs.writeFile(filePath, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

function sleep(ms) {
  if (!ms) {
    return Promise.resolve();
  }

  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = "";
  let inQuotes = false;

  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    const next = text[index + 1];

    if (inQuotes) {
      if (char === '"' && next === '"') {
        field += '"';
        index += 1;
      } else if (char === '"') {
        inQuotes = false;
      } else {
        field += char;
      }
      continue;
    }

    if (char === '"') {
      inQuotes = true;
      continue;
    }

    if (char === ",") {
      row.push(field);
      field = "";
      continue;
    }

    if (char === "\n") {
      row.push(field);
      rows.push(row);
      row = [];
      field = "";
      continue;
    }

    if (char === "\r") {
      continue;
    }

    field += char;
  }

  if (field.length > 0 || row.length > 0) {
    row.push(field);
    rows.push(row);
  }

  const [header, ...body] = rows;
  return body.map((fields, rowIndex) => {
    if (fields.length !== header.length) {
      throw new Error(
        `CSV parse error on row ${rowIndex + 2}: expected ${header.length} fields, got ${fields.length}`,
      );
    }

    return Object.fromEntries(header.map((column, columnIndex) => [column, fields[columnIndex]]));
  });
}

function buildModelIdentifier(model, provider) {
  return provider ? `${model}:${provider}` : model;
}

function csvEscape(value) {
  const text = String(value ?? "");
  if (text.includes('"') || text.includes(",") || text.includes("\n") || text.includes("\r")) {
    return `"${text.replaceAll('"', '""')}"`;
  }

  return text;
}

function stringifyCsv(rows) {
  if (rows.length === 0) {
    return "";
  }

  const columns = Object.keys(rows[0]);
  const lines = [columns.map(csvEscape).join(",")];

  for (const row of rows) {
    lines.push(columns.map((column) => csvEscape(row[column])).join(","));
  }

  return `${lines.join("\n")}\n`;
}

function buildMessages(languageLabel, text) {
  return [
    {
      role: "user",
      content: `Translate the text below into ${languageLabel}. Return only the translation.\n\nText: """\n${text}\n"""`,
    },
  ];
}

function parseRetryAfterMs(headerValue) {
  if (!headerValue) {
    return null;
  }

  const seconds = Number(headerValue);
  if (Number.isFinite(seconds)) {
    return Math.max(0, seconds * 1000);
  }

  const dateMs = Date.parse(headerValue);
  if (Number.isFinite(dateMs)) {
    return Math.max(0, dateMs - Date.now());
  }

  return null;
}

function buildHfErrorMessage(status, errorText) {
  if (status === 401) {
    return `HF API error 401: authentication failed. Check HF_TOKEN. Raw response: ${errorText}`;
  }

  if (status === 402) {
    return `HF API error 402: billing problem or payment required. Check Hugging Face billing. Raw response: ${errorText}`;
  }

  if (status === 403) {
    return `HF API error 403: access denied. Confirm model access and token permissions. Raw response: ${errorText}`;
  }

  if (status === 404) {
    return `HF API error 404: model or route not found. Check model/provider settings. Raw response: ${errorText}`;
  }

  if (status === 429) {
    return `HF API error 429: rate limited. Try a larger delay or retry later. Raw response: ${errorText}`;
  }

  return `HF API error ${status}: ${errorText}`;
}

async function fetchTranslation({ model, provider, languageLabel, text, maxTokens, maxRetries, retryBaseMs }) {
  const url = "https://router.huggingface.co/v1/chat/completions";
  const requestBody = {
    model: buildModelIdentifier(model, provider),
    messages: buildMessages(languageLabel, text),
    temperature: 0.1,
    max_tokens: maxTokens,
  };

  for (let attempt = 0; attempt <= maxRetries; attempt += 1) {
    try {
      const response = await fetch(url, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${HF_TOKEN}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestBody),
      });

      if (response.ok) {
        const payload = await response.json();
        const content = payload?.choices?.[0]?.message?.content;
        if (typeof content !== "string" || !content.trim()) {
          throw new Error(`Unexpected HF API response: ${JSON.stringify(payload)}`);
        }
        return content.trim();
      }

      const errorText = await response.text();
      const shouldRetry = TRANSIENT_STATUS_CODES.has(response.status) && attempt < maxRetries;
      if (!shouldRetry) {
        throw new Error(buildHfErrorMessage(response.status, errorText));
      }

      const retryAfterMs = parseRetryAfterMs(response.headers.get("retry-after"));
      const delayMs = retryAfterMs ?? (retryBaseMs * (2 ** attempt));
      console.error(
        `Transient HF API error ${response.status}. Waiting ${delayMs}ms before retry ${attempt + 1} of ${maxRetries}.`,
      );
      await sleep(delayMs);
    } catch (error) {
      const isLastAttempt = attempt >= maxRetries;
      const isResponseError = typeof error?.message === "string" && error.message.startsWith("HF API error");

      if (isResponseError || isLastAttempt) {
        throw error;
      }

      const delayMs = retryBaseMs * (2 ** attempt);
      console.error(
        `Network or fetch error: ${error.message}. Waiting ${delayMs}ms before retry ${attempt + 1} of ${maxRetries}.`,
      );
      await sleep(delayMs);
    }
  }

  throw new Error("HF request failed after all retries.");
}

async function loadCache(cachePath, overwrite) {
  if (overwrite || !(await exists(cachePath))) {
    return {};
  }

  return readJson(cachePath);
}

async function translateWithCache({
  cache,
  cachePath,
  model,
  provider,
  languageLabel,
  text,
  delayMs,
  maxTokens,
  maxRetries,
  retryBaseMs,
}) {
  if (!text) {
    return text;
  }

  if (cache[text]) {
    return cache[text];
  }

  const translated = await fetchTranslation({
    model,
    provider,
    languageLabel,
    text,
    maxTokens,
    maxRetries,
    retryBaseMs,
  });

  cache[text] = translated;
  await writeJson(cachePath, cache);
  await sleep(delayMs);
  return translated;
}

async function processJsonFile({
  inputPath,
  outputPath,
  cache,
  cachePath,
  language,
  model,
  provider,
  delayMs,
  maxTokens,
  limit,
  maxRetries,
  retryBaseMs,
}) {
  const rows = await readJson(inputPath);
  const workingRows = limit ? rows.slice(0, limit) : rows;
  const fields = TRANSLATABLE_FIELDS[path.basename(inputPath)];

  for (const row of workingRows) {
    for (const field of fields) {
      row[field] = await translateWithCache({
        cache,
        cachePath,
        model,
        provider,
        languageLabel: language.label,
        text: row[field],
        delayMs,
        maxTokens,
        maxRetries,
        retryBaseMs,
      });
    }
  }

  await writeJson(outputPath, workingRows);
}

async function processCsvFile({
  inputPath,
  outputPath,
  cache,
  cachePath,
  language,
  model,
  provider,
  delayMs,
  maxTokens,
  limit,
  maxRetries,
  retryBaseMs,
}) {
  const csvText = await fs.readFile(inputPath, "utf8");
  const rows = parseCsv(csvText);
  const workingRows = limit ? rows.slice(0, limit) : rows;
  const fields = TRANSLATABLE_FIELDS[path.basename(inputPath)];

  for (const row of workingRows) {
    for (const field of fields) {
      row[field] = await translateWithCache({
        cache,
        cachePath,
        model,
        provider,
        languageLabel: language.label,
        text: row[field],
        delayMs,
        maxTokens,
        maxRetries,
        retryBaseMs,
      });
    }
  }

  await fs.writeFile(outputPath, stringifyCsv(workingRows), "utf8");
}

async function writeManifest(outputDir, languages, model, provider, limit, runName) {
  const manifest = {
    created_at: new Date().toISOString(),
    model,
    provider,
    run_name: runName,
    languages,
    files: TARGET_FILES,
    limit,
  };
  await writeJson(path.join(outputDir, "_translation_manifest.json"), manifest);
}

async function main() {
  if (!HF_TOKEN) {
    throw new Error("HF_TOKEN is not set. Create a Hugging Face token and export it before running the script.");
  }

  const args = parseArgs(process.argv.slice(2));
  const runName = buildRunName(args.model, args.provider);
  const runDir = path.join(args.outputDir, runName);

  await ensureDir(runDir);
  await writeManifest(
    runDir,
    args.languages.map((language) => ({ key: language.key, label: language.label })),
    args.model,
    args.provider,
    args.limit,
    runName,
  );

  for (const language of args.languages) {
    const languageDir = path.join(runDir, language.key);
    await ensureDir(languageDir);

    const cachePath = path.join(languageDir, "_translation_cache.json");
    const cache = await loadCache(cachePath, args.overwrite || !args.resume);

    for (const fileName of TARGET_FILES) {
      const inputPath = path.join(TEST_DATA_DIR, fileName);
      const outputPath = path.join(languageDir, fileName);

      if (!args.overwrite && args.resume && (await exists(outputPath))) {
        console.log(`Skipping existing file for ${language.label}: ${fileName}`);
        continue;
      }

      console.log(`Translating ${fileName} -> ${language.label} [${runName}]`);

      const commonOptions = {
        inputPath,
        outputPath,
        cache,
        cachePath,
        language,
        model: args.model,
        provider: args.provider,
        delayMs: args.delayMs,
        maxTokens: args.maxTokens,
        limit: args.limit,
        maxRetries: args.maxRetries,
        retryBaseMs: args.retryBaseMs,
      };

      if (fileName.endsWith(".json")) {
        await processJsonFile(commonOptions);
      } else if (fileName.endsWith(".csv")) {
        await processCsvFile(commonOptions);
      } else {
        throw new Error(`Unsupported file type: ${fileName}`);
      }
    }
  }
}

main().catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});
