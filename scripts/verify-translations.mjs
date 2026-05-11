#!/usr/bin/env node

import fs from "node:fs/promises";
import path from "node:path";

const ROOT_DIR = process.cwd();
const SOURCE_DIR = path.join(ROOT_DIR, "test-data");
const TRANSLATED_DIR = path.join(ROOT_DIR, "translated-test-data");
const REPORT_DIR = path.join(ROOT_DIR, "translation-verification");

const HF_TOKEN = process.env.HF_TOKEN;
const DEFAULT_MODEL = process.env.HF_MODEL || "meta-llama/Llama-3.3-70B-Instruct";

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
    translatedDir: TRANSLATED_DIR,
    reportDir: REPORT_DIR,
    provider: process.env.HF_PROVIDER || "",
    sampleSize: 15,
    roundtrip: false,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];

    if (token === "--languages" || token === "-l") {
      args.languages = resolveLanguages(argv[index + 1]);
      index += 1;
      continue;
    }

    if (token === "--translated-dir") {
      args.translatedDir = path.resolve(ROOT_DIR, argv[index + 1]);
      index += 1;
      continue;
    }

    if (token === "--report-dir") {
      args.reportDir = path.resolve(ROOT_DIR, argv[index + 1]);
      index += 1;
      continue;
    }

    if (token === "--provider") {
      args.provider = argv[index + 1];
      index += 1;
      continue;
    }

    if (token === "--sample-size") {
      args.sampleSize = Number(argv[index + 1]);
      index += 1;
      continue;
    }

    if (token === "--roundtrip") {
      args.roundtrip = true;
      continue;
    }

    throw new Error(`Unknown argument: ${token}`);
  }

  if (!Number.isFinite(args.sampleSize) || args.sampleSize <= 0) {
    throw new Error("--sample-size must be a positive number");
  }

  if (args.roundtrip && !HF_TOKEN) {
    throw new Error("HF_TOKEN is required when --roundtrip is enabled.");
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

async function ensureDir(dirPath) {
  await fs.mkdir(dirPath, { recursive: true });
}

async function readJson(filePath) {
  return JSON.parse(await fs.readFile(filePath, "utf8"));
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
  return {
    header,
    rows: body.map((fields, rowIndex) => {
      if (fields.length !== header.length) {
        throw new Error(
          `CSV parse error on row ${rowIndex + 2}: expected ${header.length} fields, got ${fields.length}`,
        );
      }
      return Object.fromEntries(header.map((column, columnIndex) => [column, fields[columnIndex]]));
    }),
  };
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

function buildModelIdentifier(model, provider) {
  return provider ? `${model}:${provider}` : model;
}

async function backTranslate({ provider, text, sourceLanguage }) {
  const response = await fetch("https://router.huggingface.co/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${HF_TOKEN}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: buildModelIdentifier(DEFAULT_MODEL, provider),
      messages: [
        {
          role: "system",
          content: "Translate the text back to English. Return only the English text.",
        },
        {
          role: "user",
          content: `Source language: ${sourceLanguage}\nText:\n${text}`,
        },
      ],
      temperature: 0.1,
      max_tokens: 256,
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`HF API error ${response.status}: ${errorText}`);
  }

  const payload = await response.json();
  return payload?.choices?.[0]?.message?.content?.trim() || "";
}

function sampleEvenly(items, sampleSize) {
  if (items.length <= sampleSize) {
    return items;
  }

  const result = [];
  const step = (items.length - 1) / (sampleSize - 1);
  for (let index = 0; index < sampleSize; index += 1) {
    result.push(items[Math.round(index * step)]);
  }
  return result;
}

async function verifyJsonFile({ sourcePath, translatedPath, fileName }) {
  const sourceRows = await readJson(sourcePath);
  const translatedRows = await readJson(translatedPath);
  const fields = TRANSLATABLE_FIELDS[fileName];

  const issues = [];
  if (!Array.isArray(translatedRows)) {
    issues.push("Translated JSON is not an array");
  }
  if (sourceRows.length !== translatedRows.length) {
    issues.push(`Row count mismatch: source=${sourceRows.length}, translated=${translatedRows.length}`);
  }

  const samples = [];
  for (let index = 0; index < Math.min(sourceRows.length, translatedRows.length); index += 1) {
    for (const field of fields) {
      samples.push({
        row_number: index + 1,
        file_name: fileName,
        field_name: field,
        source_text: sourceRows[index][field],
        translated_text: translatedRows[index][field],
      });

      if (!translatedRows[index][field]) {
        issues.push(`Empty translated value at row ${index + 1}, field ${field}`);
      }
    }
  }

  return { issues, samples };
}

async function verifyCsvFile({ sourcePath, translatedPath, fileName }) {
  const source = parseCsv(await fs.readFile(sourcePath, "utf8"));
  const translated = parseCsv(await fs.readFile(translatedPath, "utf8"));
  const fields = TRANSLATABLE_FIELDS[fileName];

  const issues = [];
  if (source.header.join(",") !== translated.header.join(",")) {
    issues.push("Header mismatch");
  }
  if (source.rows.length !== translated.rows.length) {
    issues.push(`Row count mismatch: source=${source.rows.length}, translated=${translated.rows.length}`);
  }

  const samples = [];
  for (let index = 0; index < Math.min(source.rows.length, translated.rows.length); index += 1) {
    for (const field of fields) {
      samples.push({
        row_number: index + 1,
        file_name: fileName,
        field_name: field,
        source_text: source.rows[index][field],
        translated_text: translated.rows[index][field],
      });

      if (!translated.rows[index][field]) {
        issues.push(`Empty translated value at row ${index + 1}, field ${field}`);
      }
    }
  }

  return { issues, samples };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  await ensureDir(args.reportDir);

  const summaryRows = [];

  for (const language of args.languages) {
    const languageDir = path.join(args.translatedDir, language.key);
    const languageReportDir = path.join(args.reportDir, language.key);
    await ensureDir(languageReportDir);

    const reviewSamples = [];
    const issueRows = [];

    for (const fileName of TARGET_FILES) {
      const sourcePath = path.join(SOURCE_DIR, fileName);
      const translatedPath = path.join(languageDir, fileName);

      let result;
      if (fileName.endsWith(".json")) {
        result = await verifyJsonFile({ sourcePath, translatedPath, fileName });
      } else {
        result = await verifyCsvFile({ sourcePath, translatedPath, fileName });
      }

      for (const issue of result.issues) {
        issueRows.push({
          language: language.label,
          file_name: fileName,
          issue,
        });
      }

      reviewSamples.push(...sampleEvenly(result.samples, args.sampleSize));
      summaryRows.push({
        language: language.label,
        file_name: fileName,
        issue_count: result.issues.length,
        status: result.issues.length === 0 ? "PASS" : "CHECK",
      });
    }

    if (args.roundtrip) {
      for (const sample of reviewSamples) {
        sample.back_translated_english = await backTranslate({
          provider: args.provider,
          sourceLanguage: language.label,
          text: sample.translated_text,
        });
      }
    }

    await fs.writeFile(
      path.join(languageReportDir, "review_samples.csv"),
      stringifyCsv(reviewSamples),
      "utf8",
    );

    const issuesOutput = issueRows.length > 0
      ? issueRows
      : [{ language: language.label, file_name: "", issue: "" }];

    await fs.writeFile(
      path.join(languageReportDir, "issues.csv"),
      stringifyCsv(issuesOutput),
      "utf8",
    );
  }

  await fs.writeFile(path.join(args.reportDir, "summary.csv"), stringifyCsv(summaryRows), "utf8");
}

main().catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});
