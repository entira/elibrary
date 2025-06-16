# GitHub Issue — Improve Metadata Extraction Pipeline: Prompt, Parsing, Model Compatibility & Error Handling

## Title
Improve Metadata Extraction Pipeline: Prompt, Parsing, Model Compatibility & Error Handling

---

## Description

The current implementation for extracting bibliographic metadata from academic/technical documents using local LLMs (via Ollama API) exhibits multiple reliability, accuracy, and robustness issues. This issue aims to document the problems and propose concrete improvements across prompt design, input preprocessing, output parsing, and retry logic—especially with newer models such as phi-4, qwen3:14b, granite3.2:8b, and others.

---

## ⚠️ Current Problems

### Prompt Design Issues
- **Example JSON is often copied verbatim**: Especially by smaller models, instead of generating fresh metadata.
- **Mix of instructions and sample output causes confusion** (e.g., "Return ONLY valid JSON…" followed by sample JSON inside the prompt).
- **No hard delimiter** results in non-JSON content prepended/appended (e.g., "Here is the JSON you requested:").
- **Stop token `\n\n` cuts off multi-line fields** (e.g., authors), breaking JSON structure.

### Input Slicing
- **Using `sample_text[:2000]` naively truncates text**, often removing critical title/author/DOI data from headers.
- **Extracting random portions** (instead of structured parts like the first page) reduces model context.

### Parsing Logic
- **JSON is extracted using `re.search(r'\{.*\}')`**, which is greedy and fails if:
  - Multiple JSON objects are returned.
  - Closing brace is missing due to truncation.
- **No fallback if model returns multiple JSONs**, wrapped content, or partial outputs.
- **Fails silently when keys are present but values are garbage** or placeholder-like.

### Retry Logic
- **Repeats the same prompt three times with no variation** → no improvement expected.
- **Fallback to filename metadata only works when `strip()` cleans correctly**—non-visible Unicode or tabs may still be present.

### Output Validation
- **Only checks for key existence and non-empty strings**. No format normalization:
  - `year: "© 2024"` should become `"2024"`.
  - `doi: "doi:10.xxxx"` should normalize to just the identifier.

---

## ✅ Recommendations

### Prompt Refactor

Split system/user roles and use schema-driven few-shot prompting:

```
System: You are a metadata extraction parser. Respond ONLY with valid JSON that conforms to the given schema.
User:
### INPUT TEXT
{first_page_text}

### JSON SCHEMA
{
  "title": "string",
  "authors": "string", 
  "publishers": "string",
  "year": "string",
  "doi": "string"
}
```

- Emphasize "Return ONLY the JSON object" outside of examples.
- Add stop marker `<<END>>` to avoid over-generation.

### Input Handling
- **Extract only the first page** (or document header).
- **Avoid arbitrary slicing with `[:2000]`**—important metadata is usually structured at the top.

### JSON Parsing
- Replace greedy regex with more reliable logic:

```python
start = resp.find('{')
end   = resp.rfind('}')
metadata = json.loads(resp[start:end+1])
```

- For models supporting JSON mode, rely on structured output (`"format": "json"` or tools/function_calling).

### Model-Specific Tuning

| Model | JSON mode | Notes |
|-------|-----------|-------|
| `phi4:latest` | ✅ | Supports function-calling, deterministic JSON |
| `granite3.2:8b` | ✅ | Best used with `format="json"` |
| `gemma3:4b` | ❌ | Needs strong few-shot and low `top_p` |
| `deepseek-r1:1.5b` | ✅ | Use `json:` prefix or schema examples |
| `qwen3:14b` | ✅ | Supports structured response via `response_format` |

### Output Normalization

Implement a normalization layer post-extraction:

```python
def normalize(metadata):
    metadata['year'] = extract_4digit_year(metadata.get('year', ''))
    metadata['doi'] = metadata['doi'].replace("doi:", "").strip()
    ...
    return metadata
```

### Retry Strategy
- **Use few-shot examples after first attempt**.
- **Introduce exponential backoff and minor prompt mutations** per retry.

### Token Allocation
- **`max_tokens=256` is too small**. Use adaptive allocation (e.g., 64–512 depending on model and input).
- **For larger models, give room for verbose authors or long titles**.

---

## 📦 Expected Benefits
- **Significantly improved accuracy** of extracted metadata.
- **Fewer parsing and JSONDecodeError failures**.
- **Better utilization of capabilities in modern local models** (e.g., JSON mode, system prompts).
- **Increased robustness of fallback logic** (e.g., partial data via filename).

---

## 🛠️ Tasks

### Core Implementation
- [ ] **Refactor prompt to separate system/user roles**
- [ ] **Normalize input slicing** (first page only)
- [ ] **Replace regex JSON parsing** with robust extraction
- [ ] **Use `format: json` where supported**
- [ ] **Improve retry logic** with few-shot fallback
- [ ] **Add normalization for year and DOI**

### Model Compatibility
- [ ] **Update model config table** for prompt compatibility
- [ ] **Implement model-specific JSON mode handling**
- [ ] **Add adaptive token allocation** per model type
- [ ] **Create model capability detection**

### Testing & Validation
- [ ] **Write unit tests** for extraction + fallback fusion
- [ ] **Add integration tests** for different model types
- [ ] **Performance benchmarking** before/after improvements
- [ ] **Validation suite** for metadata quality

### Documentation
- [ ] **Update extraction methodology** documentation
- [ ] **Create model compatibility guide**
- [ ] **Add troubleshooting guide** for common failures

---

## 🔧 Implementation Details

### Current Code Location
- **File**: `pdf_library_processor.py`
- **Method**: `extract_metadata_with_ollama()`
- **Lines**: ~611-720

### Proposed Refactor Structure
```python
class MetadataExtractor:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.capabilities = self._detect_model_capabilities()
    
    def extract(self, text: str, filename: str = "") -> Dict[str, str]:
        # Main extraction logic with model-specific handling
        pass
    
    def _build_prompt(self, text: str, attempt: int = 1) -> Dict[str, Any]:
        # Adaptive prompt building based on attempt and model
        pass
    
    def _parse_response(self, response: str) -> Dict[str, str]:
        # Robust JSON parsing with fallbacks
        pass
    
    def _normalize_metadata(self, metadata: Dict[str, str]) -> Dict[str, str]:
        # Post-processing normalization
        pass
```

### Testing Strategy
1. **Unit Tests**: Individual method testing
2. **Integration Tests**: Full extraction pipeline
3. **Model Tests**: Specific model behavior validation
4. **Performance Tests**: Speed and accuracy benchmarks

---

## 📊 Success Metrics

### Before/After Comparison
- **JSON Parse Success Rate**: Target >95% (currently ~70-85%)
- **Metadata Completeness**: Target >90% (currently ~60-80%)
- **Processing Speed**: Maintain <3s per document
- **Error Handling**: Graceful degradation in 100% of cases

### Quality Indicators
- **Title Extraction**: >95% accuracy on academic papers
- **Author Parsing**: >90% accuracy with proper name separation
- **Year Normalization**: 100% 4-digit year format
- **DOI Cleaning**: 100% proper identifier format

---

## 🎯 Priority & Timeline

**Priority**: High  
**Estimated Effort**: 3-5 days  
**Target Sprint**: Next available  

### Phase 1 (Day 1-2): Core Improvements
- Prompt refactoring
- JSON parsing robustness
- Basic normalization

### Phase 2 (Day 3-4): Model Compatibility
- Model-specific handling
- JSON mode integration
- Adaptive token allocation

### Phase 3 (Day 5): Testing & Validation
- Comprehensive test suite
- Performance benchmarking
- Documentation updates

---

**Labels**: enhancement, metadata-extraction, llm-integration, data-quality  
**Assignee**: TBD  
**Dependencies**: None  
**Related Issues**: #[pdf-deeplink-support]