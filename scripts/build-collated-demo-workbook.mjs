#!/usr/bin/env node

import fs from "node:fs/promises";
import path from "node:path";
import { spawnSync } from "node:child_process";
import crypto from "node:crypto";

const ROOT_DIR = process.cwd();
const DEFAULT_SOURCE_DIR = path.join(ROOT_DIR, "test-data");
const DEFAULT_TRANSLATED_DIR = path.join(ROOT_DIR, "translated-test-data");
const DEFAULT_TEMPLATE_DIR = path.join(ROOT_DIR, "translation_demo_sheet_unzipped");
const DEFAULT_OUTPUT = path.join(ROOT_DIR, "llama_translation_collated.xlsx");

const LANGUAGE_LABELS = {
  arabic: "Arabic",
  bengali: "Bengali",
  chinese_mandarin: "Chinese Mandarin",
  french: "French",
  german: "German",
  polish: "Polish",
  portuguese_brazilian: "Portuguese (Brazilian)",
  spanish: "Spanish",
  turkish: "Turkish",
  urdu: "Urdu",
};

function parseArgs(argv) {
  const args = {
    sourceDir: DEFAULT_SOURCE_DIR,
    translatedDir: DEFAULT_TRANSLATED_DIR,
    templateDir: DEFAULT_TEMPLATE_DIR,
    outputPath: DEFAULT_OUTPUT,
    dedupeWithinGroups: false,
    mappingOutput: null,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];

    if (token === "--source-dir") {
      args.sourceDir = path.resolve(ROOT_DIR, argv[index + 1]);
      index += 1;
      continue;
    }

    if (token === "--translated-dir") {
      args.translatedDir = path.resolve(ROOT_DIR, argv[index + 1]);
      index += 1;
      continue;
    }

    if (token === "--template-dir") {
      args.templateDir = path.resolve(ROOT_DIR, argv[index + 1]);
      index += 1;
      continue;
    }

    if (token === "--output") {
      args.outputPath = path.resolve(ROOT_DIR, argv[index + 1]);
      index += 1;
      continue;
    }

    if (token === "--dedupe-within-groups") {
      args.dedupeWithinGroups = true;
      continue;
    }

    if (token === "--mapping-output") {
      args.mappingOutput = path.resolve(ROOT_DIR, argv[index + 1]);
      index += 1;
      continue;
    }

    throw new Error(`Unknown argument: ${token}`);
  }

  if (args.dedupeWithinGroups && !args.mappingOutput) {
    const ext = path.extname(args.outputPath);
    const base = ext ? args.outputPath.slice(0, -ext.length) : args.outputPath;
    args.mappingOutput = `${base}_mapping.csv`;
  }

  return args;
}

function xmlEscape(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&apos;");
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

function inlineCell(ref, styleId, value) {
  if (value === null || value === undefined || value === "") {
    return `<c r="${ref}" s="${styleId}"/>`;
  }
  return `<c r="${ref}" s="${styleId}" t="inlineStr"><is><t xml:space="preserve">${xmlEscape(value)}</t></is></c>`;
}

function cleanText(value) {
  let text = String(value ?? "").replace(/\r\n/g, "\n").trim();
  text = text.replace(/^\s*"{2,}\s*/, "").replace(/\s*"{2,}\s*$/, "").trim();
  while ((text.startsWith('"') && text.endsWith('"')) || (text.startsWith("'") && text.endsWith("'"))) {
    text = text.slice(1, -1).trim();
  }
  text = text.replace(/\n+/g, " ").replace(/\s{2,}/g, " ").trim();
  return text;
}

function cleanTopicKey(value) {
  return String(value ?? "").trim();
}

function combineTopicKeys(topicKeys) {
  return [...new Set(topicKeys.map(cleanTopicKey).filter(Boolean))].join("|");
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
  return body.map((fields) => Object.fromEntries(header.map((column, columnIndex) => [column, fields[columnIndex] ?? ""])));
}

async function readJson(filePath) {
  return JSON.parse(await fs.readFile(filePath, "utf8"));
}

async function ensureDir(dirPath) {
  await fs.mkdir(dirPath, { recursive: true });
}

function buildHash(...parts) {
  return crypto.createHash("sha1").update(parts.join("||")).digest("hex").slice(0, 12);
}

function buildGroupedRows(sourceRows, translatedRows, prefix, languageLabel, sourceFile, dedupeWithinGroups) {
  const groups = [];
  const groupMap = new Map();

  for (let index = 0; index < sourceRows.length; index += 1) {
    const sourceRow = sourceRows[index];
    const translatedRow = translatedRows[index];
    const question = cleanText(sourceRow.llm_question);
    const translatedQuestion = cleanText(translatedRow.llm_question);
    const answer = cleanText(sourceRow.response);
    const translatedAnswer = cleanText(translatedRow.response);
    const topicKey = cleanTopicKey(sourceRow.topic_key);

    let group = groupMap.get(question);
    if (!group) {
      group = {
        question,
        translatedQuestion,
        topicKeys: [],
        answers: [],
      };
      groupMap.set(question, group);
      groups.push(group);
    }

    group.topicKeys.push(topicKey);
    group.answers.push({ answer, translatedAnswer, topicKey });
  }

  const rows = [];
  const mappings = [];

  for (let index = 0; index < groups.length; index += 1) {
    const groupNumber = index + 1;
    const group = groups[index];
    const groupTopicKey = combineTopicKeys(group.topicKeys);
    const questionIdentifier = `${prefix}Q${groupNumber}`;
    const questionHash = buildHash(languageLabel, sourceFile, questionIdentifier, group.question, group.translatedQuestion, groupTopicKey);
    rows.push({
      identifier: questionIdentifier,
      original: group.question,
      translated: group.translatedQuestion,
      topicKey: groupTopicKey,
      rowType: "question",
    });
    mappings.push({
      language: languageLabel,
      source_file: sourceFile,
      deduped_identifier: questionIdentifier,
      row_type: "question",
      topic_key: groupTopicKey,
      hash: questionHash,
      source_identifiers: questionIdentifier,
      original_text: group.question,
      translated_text: group.translatedQuestion,
    });

    if (!dedupeWithinGroups) {
      for (let answerIndex = 0; answerIndex < group.answers.length; answerIndex += 1) {
        const answer = group.answers[answerIndex];
        const identifier = `${prefix}A${groupNumber}.${answerIndex + 1}`;
        const hash = buildHash(languageLabel, sourceFile, questionIdentifier, identifier, answer.answer, answer.translatedAnswer, answer.topicKey);
        rows.push({
          identifier,
          original: answer.answer,
          translated: answer.translatedAnswer,
          topicKey: answer.topicKey,
          rowType: "answer",
        });
        mappings.push({
          language: languageLabel,
          source_file: sourceFile,
          deduped_identifier: identifier,
          row_type: "answer",
          topic_key: answer.topicKey,
          hash,
          source_identifiers: identifier,
          original_text: answer.answer,
          translated_text: answer.translatedAnswer,
        });
      }
      continue;
    }

    const dedupedAnswers = [];
    const answerMap = new Map();

    for (let answerIndex = 0; answerIndex < group.answers.length; answerIndex += 1) {
      const answer = group.answers[answerIndex];
      const sourceIdentifier = `${prefix}A${groupNumber}.${answerIndex + 1}`;
      const dedupeKey = `${answer.answer}\u0001${answer.translatedAnswer}`;
      let deduped = answerMap.get(dedupeKey);
      if (!deduped) {
        deduped = {
          original: answer.answer,
          translated: answer.translatedAnswer,
          topicKeys: [],
          sourceIdentifiers: [],
        };
        answerMap.set(dedupeKey, deduped);
        dedupedAnswers.push(deduped);
      }
      deduped.topicKeys.push(answer.topicKey);
      deduped.sourceIdentifiers.push(sourceIdentifier);
    }

    for (let dedupedIndex = 0; dedupedIndex < dedupedAnswers.length; dedupedIndex += 1) {
      const answer = dedupedAnswers[dedupedIndex];
      const identifier = `${prefix}A${groupNumber}.${dedupedIndex + 1}`;
      const topicKey = combineTopicKeys(answer.topicKeys);
      const hash = buildHash(languageLabel, sourceFile, questionIdentifier, identifier, answer.original, answer.translated, topicKey);
      rows.push({
        identifier,
        original: answer.original,
        translated: answer.translated,
        topicKey,
        rowType: "answer",
      });
      mappings.push({
        language: languageLabel,
        source_file: sourceFile,
        deduped_identifier: identifier,
        row_type: "answer",
        topic_key: topicKey,
        hash,
        source_identifiers: answer.sourceIdentifiers.join("|"),
        original_text: answer.original,
        translated_text: answer.translated,
      });
    }
  }

  return { rows, mappings };
}

function buildResponseOnlyRows(sourceRows, translatedRows, prefix, languageLabel, sourceFile) {
  const rows = [];
  const mappings = [];

  for (let index = 0; index < sourceRows.length; index += 1) {
    const identifier = `${prefix}${index + 1}`;
    const original = cleanText(sourceRows[index].response);
    const translated = cleanText(translatedRows[index].response);
    const topicKey = cleanTopicKey(sourceRows[index].topic_key);
    const hash = buildHash(languageLabel, sourceFile, identifier, original, translated, topicKey);
    rows.push({
      identifier,
      original,
      translated,
      topicKey,
      rowType: "answer",
    });
    mappings.push({
      language: languageLabel,
      source_file: sourceFile,
      deduped_identifier: identifier,
      row_type: "answer",
      topic_key: topicKey,
      hash,
      source_identifiers: identifier,
      original_text: original,
      translated_text: translated,
    });
  }

  return { rows, mappings };
}

async function buildLanguageRows({ languageKey, languageLabel, sourceDir, translatedDir, dedupeWithinGroups }) {
  const languageDir = path.join(translatedDir, languageKey);

  const followupSource = parseCsv(await fs.readFile(path.join(sourceDir, "detect_followup_info_in_previous_user_response.csv"), "utf8"));
  const followupTranslated = parseCsv(await fs.readFile(path.join(languageDir, "detect_followup_info_in_previous_user_response.csv"), "utf8"));

  const symptomSource = parseCsv(await fs.readFile(path.join(sourceDir, "detect_whether_patient_is_experiencing_symptom.csv"), "utf8"));
  const symptomTranslated = parseCsv(await fs.readFile(path.join(languageDir, "detect_whether_patient_is_experiencing_symptom.csv"), "utf8"));

  const mentionSource = await readJson(path.join(sourceDir, "detect_all_symptom_mention_using_llm.json"));
  const mentionTranslated = await readJson(path.join(languageDir, "detect_all_symptom_mention_using_llm.json"));

  const followup = buildGroupedRows(
    followupSource,
    followupTranslated,
    "F",
    languageLabel,
    "detect_followup_info_in_previous_user_response.csv",
    dedupeWithinGroups,
  );

  const symptom = buildGroupedRows(
    symptomSource,
    symptomTranslated,
    "S",
    languageLabel,
    "detect_whether_patient_is_experiencing_symptom.csv",
    dedupeWithinGroups,
  );

  const mention = buildResponseOnlyRows(
    mentionSource,
    mentionTranslated,
    "R",
    languageLabel,
    "detect_all_symptom_mention_using_llm.json",
  );

  return {
    rows: [...followup.rows, ...symptom.rows, ...mention.rows],
    mappings: [...followup.mappings, ...symptom.mappings, ...mention.mappings],
  };
}

function buildSheetXml(languageLabel, rows) {
  const rowXml = [];
  rowXml.push(
    `<row r="1">${[
      inlineCell("A1", 7, "identifier"),
      inlineCell("B1", 7, "original"),
      inlineCell("C1", 7, "machine_translated_dora"),
      inlineCell("D1", 2, "Language"),
      inlineCell("E1", 8, "topic_key"),
      inlineCell("F1", 8, "edit_required_dora"),
      inlineCell("G1", 8, "post_edited_dora"),
      inlineCell("H1", 8, "post_edited_cmqm_class"),
      inlineCell("I1", 8, "annotator_notes"),
      inlineCell("J1", 3, "Phase 2 error_categories"),
    ].join("")}</row>`,
  );

  for (let index = 0; index < rows.length; index += 1) {
    const rowNumber = index + 2;
    const row = rows[index];
    const textStyle = row.rowType === "question" ? 4 : 10;
    rowXml.push(
      `<row r="${rowNumber}">${[
        inlineCell(`A${rowNumber}`, 9, row.identifier),
        inlineCell(`B${rowNumber}`, textStyle, row.original),
        inlineCell(`C${rowNumber}`, textStyle, row.translated),
        inlineCell(`D${rowNumber}`, 9, languageLabel),
        inlineCell(`E${rowNumber}`, 9, row.topicKey),
        inlineCell(`F${rowNumber}`, 5, ""),
        inlineCell(`G${rowNumber}`, 6, ""),
        inlineCell(`H${rowNumber}`, 6, ""),
        inlineCell(`I${rowNumber}`, 6, ""),
        inlineCell(`J${rowNumber}`, 6, ""),
        inlineCell(`K${rowNumber}`, 6, ""),
      ].join("")}</row>`,
    );
  }

  const validationEnd = Math.max(2, rows.length + 1);

  return `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:mx="http://schemas.microsoft.com/office/mac/excel/2008/main" xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" xmlns:mv="urn:schemas-microsoft-com:mac:vml" xmlns:x14="http://schemas.microsoft.com/office/spreadsheetml/2009/9/main" xmlns:x15="http://schemas.microsoft.com/office/spreadsheetml/2010/11/main" xmlns:x14ac="http://schemas.microsoft.com/office/spreadsheetml/2009/9/ac" xmlns:xm="http://schemas.microsoft.com/office/excel/2006/main"><sheetPr><outlinePr summaryBelow="0" summaryRight="0"/></sheetPr><sheetViews><sheetView workbookViewId="0"/></sheetViews><sheetFormatPr customHeight="1" defaultColWidth="12.63" defaultRowHeight="15.75"/><cols><col customWidth="1" min="2" max="2" width="71.63"/><col customWidth="1" min="3" max="3" width="87.5"/><col customWidth="1" min="5" max="5" width="30.0"/><col customWidth="1" min="6" max="6" width="25.25"/><col customWidth="1" min="7" max="7" width="27.13"/><col customWidth="1" min="8" max="8" width="22.25"/><col customWidth="1" min="9" max="9" width="23.88"/><col customWidth="1" min="10" max="10" width="24.88"/></cols><sheetData>${rowXml.join("")}</sheetData><mergeCells count="1"><mergeCell ref="J1:K1"/></mergeCells><dataValidations><dataValidation type="list" allowBlank="1" showErrorMessage="1" sqref="F2:F${validationEnd}"><formula1>&quot;Yes,Option 2No&quot;</formula1></dataValidation><dataValidation type="list" allowBlank="1" showErrorMessage="1" sqref="H2:H${validationEnd}"><formula1>&quot;none,minor,major&quot;</formula1></dataValidation></dataValidations></worksheet>`;
}

function buildWorkbookXml(sheetNames) {
  const sheets = sheetNames.map((name, index) => `<sheet state="visible" name="${xmlEscape(name)}" sheetId="${index + 1}" r:id="rId${index + 3}"/>`).join("");
  return `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><workbookPr/><sheets>${sheets}</sheets><definedNames/><calcPr/></workbook>`;
}

function buildWorkbookRelsXml(sheetCount) {
  const rels = [
    '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="theme/theme1.xml"/>',
    '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>',
  ];

  for (let index = 0; index < sheetCount; index += 1) {
    rels.push(`<Relationship Id="rId${index + 3}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet${index + 1}.xml"/>`);
  }

  return `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">${rels.join("")}</Relationships>`;
}

function buildContentTypesXml(sheetCount) {
  const overrides = [];
  for (let index = 0; index < sheetCount; index += 1) {
    overrides.push(`<Override ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml" PartName="/xl/worksheets/sheet${index + 1}.xml"/>`);
  }

  overrides.push('<Override ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml" PartName="/xl/styles.xml"/>');
  overrides.push('<Override ContentType="application/vnd.openxmlformats-officedocument.theme+xml" PartName="/xl/theme/theme1.xml"/>');
  overrides.push('<Override ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml" PartName="/xl/workbook.xml"/>');

  return `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default ContentType="application/xml" Extension="xml"/><Default ContentType="application/vnd.openxmlformats-package.relationships+xml" Extension="rels"/>${overrides.join("")}</Types>`;
}

async function buildWorkbook(args) {
  const tempDir = path.join(ROOT_DIR, ".collated_workbook_build");
  await fs.rm(tempDir, { recursive: true, force: true });
  await ensureDir(path.join(tempDir, "_rels"));
  await ensureDir(path.join(tempDir, "xl", "_rels"));
  await ensureDir(path.join(tempDir, "xl", "theme"));
  await ensureDir(path.join(tempDir, "xl", "worksheets"));

  const stylesPath = path.join(args.templateDir, "xl", "styles.xml");
  const themePath = path.join(args.templateDir, "xl", "theme", "theme1.xml");

  await fs.copyFile(stylesPath, path.join(tempDir, "xl", "styles.xml"));
  await fs.copyFile(themePath, path.join(tempDir, "xl", "theme", "theme1.xml"));

  await fs.writeFile(
    path.join(tempDir, "_rels", ".rels"),
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>',
    "utf8",
  );

  const languageEntries = (await fs.readdir(args.translatedDir, { withFileTypes: true }))
    .filter((entry) => entry.isDirectory())
    .map((entry) => ({
      key: entry.name,
      label: LANGUAGE_LABELS[entry.name] || entry.name,
    }))
    .sort((a, b) => a.label.localeCompare(b.label));

  const allMappings = [];

  for (let index = 0; index < languageEntries.length; index += 1) {
    const language = languageEntries[index];
    const { rows, mappings } = await buildLanguageRows({
      languageKey: language.key,
      languageLabel: language.label,
      sourceDir: args.sourceDir,
      translatedDir: args.translatedDir,
      dedupeWithinGroups: args.dedupeWithinGroups,
    });

    const sheetXml = buildSheetXml(language.label, rows);
    await fs.writeFile(path.join(tempDir, "xl", "worksheets", `sheet${index + 1}.xml`), sheetXml, "utf8");
    allMappings.push(...mappings);
  }

  await fs.writeFile(path.join(tempDir, "xl", "workbook.xml"), buildWorkbookXml(languageEntries.map((entry) => entry.label)), "utf8");
  await fs.writeFile(path.join(tempDir, "xl", "_rels", "workbook.xml.rels"), buildWorkbookRelsXml(languageEntries.length), "utf8");
  await fs.writeFile(path.join(tempDir, "[Content_Types].xml"), buildContentTypesXml(languageEntries.length), "utf8");

  await fs.rm(args.outputPath, { force: true });
  const zipPath = `${args.outputPath}.zip`;
  await fs.rm(zipPath, { force: true });

  const compress = spawnSync(
    "powershell",
    [
      "-NoLogo",
      "-NoProfile",
      "-Command",
      `Compress-Archive -Path '${tempDir}\\*' -DestinationPath '${zipPath}' -Force; Move-Item -LiteralPath '${zipPath}' -Destination '${args.outputPath}' -Force`,
    ],
    { stdio: "inherit" },
  );

  if (compress.status !== 0) {
    throw new Error("Failed to create xlsx archive.");
  }

  if (args.mappingOutput) {
    await fs.writeFile(args.mappingOutput, stringifyCsv(allMappings), "utf8");
  }
}

const args = parseArgs(process.argv.slice(2));
buildWorkbook(args).catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});
