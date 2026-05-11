# Hugging Face Translation Setup

This repo now includes:

- `scripts/translate-test-data.mjs`
- `scripts/verify-translations.mjs`

The translator processes these files:

- `test-data/detect_all_symptom_mention_using_llm.json`
- `test-data/detect_followup_info_in_previous_user_response.csv`
- `test-data/detect_whether_patient_is_experiencing_symptom.csv`

What gets translated:

- JSON: `response`
- CSV `detect_followup_info_in_previous_user_response.csv`: `llm_question` and `response`
- CSV `detect_whether_patient_is_experiencing_symptom.csv`: `llm_question` and `response`

`context` is left unchanged.

Default target languages:

- Spanish
- French
- German
- Polish
- Portuguese (Brazilian)
- Turkish
- Arabic
- Chinese Mandarin
- Bengali
- Urdu

The translator keeps a per-language cache in `translated-test-data/<model-run-name>/<language>/_translation_cache.json`, so repeated `llm_question` strings are translated once and then reused.

## Prompting choice

The translation prompt is intentionally simple. It uses a direct zero-shot instruction with the instruction placed first and the source text clearly delimited.

Current prompt shape:

```text
Translate the text below into <language>. Return only the translation.

Text: """
<source text>
"""
```

This is deliberate. Since you want humans to rate natural LLM translation quality and observe the natural error distribution, the prompt avoids extra style constraints, domain rules, chain-of-thought requests, and few-shot examples.

## 1. Subscribe to Hugging Face PRO

As of March 24, 2026, Hugging Face lists PRO at `$9/month` and says it includes `20x` the free inference credits. Their billing docs currently describe that as `$2.00` monthly routed inference credits for PRO users, versus `$0.10` for free users, and PRO users can continue with pay-as-you-go after credits run out.

Open:

- `https://huggingface.co/pricing`

## 2. Accept the model license

For gated models, open the model page and accept the license before running the script.

Example:

- `https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct`

## 3. Create an API token

Open:

- `https://huggingface.co/settings/tokens`

Create a token with permission for `Inference Providers`.

In PowerShell, set it for the current session:

```powershell
$env:HF_TOKEN="hf_your_token_here"
```

Optional overrides:

```powershell
$env:HF_MODEL="meta-llama/Llama-3.3-70B-Instruct"
$env:HF_PROVIDER=""
$env:HF_MAX_RETRIES="5"
$env:HF_RETRY_BASE_MS="5000"
```

## 4. Run the translator

Default run:

```powershell
node .\scripts\translate-test-data.mjs
```

Recommended command for unstable provider errors:

```powershell
node .\scripts\translate-test-data.mjs --delay-ms 300 --max-retries 5 --retry-base-ms 5000
```

Example with 5 languages only:

```powershell
node .\scripts\translate-test-data.mjs --languages spanish,french,german,arabic,urdu
```

Example pilot run for 2 languages with only 10 rows per file:

```powershell
node .\scripts\translate-test-data.mjs --languages spanish,french --limit 10
```

Example using a different model:

```powershell
node .\scripts\translate-test-data.mjs --model meta-llama/Meta-Llama-3.1-8B-Instruct --languages spanish,french,german,arabic,urdu
```

Outputs are now written to model-specific folders, for example:

```text
translated-test-data/
  meta_llama_llama_3_3_70b_instruct/
    spanish/
    french/
  meta_llama_meta_llama_3_1_8b_instruct/
    spanish/
    french/
```

## 5. Error handling

The translator retries transient HF/provider errors automatically.

Retried by default:

- `429`
- `500`
- `502`
- `503`
- `504`
- network/fetch failures

It also prints clearer messages for:

- `401` invalid token
- `402` billing/payment problem
- `403` model access or permission problem
- `404` bad model/provider route

## 6. Verification

Run the structural verifier after translation:

```powershell
node .\scripts\verify-translations.mjs --translated-dir .\translated-test-data\meta_llama_llama_3_3_70b_instruct
```

For a stronger audit on the sampled rows, run round-trip verification:

```powershell
node .\scripts\verify-translations.mjs --translated-dir .\translated-test-data\meta_llama_llama_3_3_70b_instruct --roundtrip
```

## 7. Good comparison models

Reasonable HF comparison set for your study:

- `meta-llama/Llama-3.3-70B-Instruct` as the main model
- `meta-llama/Meta-Llama-3.1-70B-Instruct` as the closest same-family comparison
- `Qwen/Qwen2.5-72B-Instruct` as another strong large open instruct model
- `mistralai/Mixtral-8x7B-Instruct-v0.1` as a smaller MoE-style comparison
- `meta-llama/Meta-Llama-3.1-8B-Instruct` as the smaller Llama baseline

## 8. Sources

- Hugging Face pricing: `https://huggingface.co/pricing`
- Hugging Face Inference Providers billing: `https://huggingface.co/docs/api-inference/en/pricing`
- Hugging Face chat completions API: `https://huggingface.co/docs/inference-providers/en/tasks/chat-completion`
- Llama 3.3 70B Instruct model page: `https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct`
