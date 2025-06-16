# Model Performance Analysis Report

## Executive Summary

This report analyzes the performance of 10 different language models tested for PDF metadata extraction tasks across 234 files. The analysis focuses on success rates, processing speed, error patterns, and library-specific performance differences.

## Model Performance Overview

| Model | Success Rate | Processing Speed (files/min) | Avg Time/File (s) | Total Processing Time (s) |
|-------|-------------|------------------------------|-------------------|---------------------------|
| **Gemma3 12B-IT-QAT** | **96.15%** | 12.52 | 4.79 | 1121.54 |
| **Gemma3 12B** | **96.15%** | 11.14 | 5.38 | 1259.98 |
| **Phi4 Latest** | **95.30%** | 10.66 | 5.63 | 1316.56 |
| **Gemma3 4B-IT-QAT** | **95.73%** | 29.83 | 2.01 | 470.65 |
| **Gemma3 4B** | **95.73%** | 25.87 | 2.32 | 542.79 |
| **Granite3.2 8B** | **95.73%** | 15.60 | 3.85 | 899.87 |
| **Mistral Latest** | **85.90%** | 18.65 | 3.22 | 752.92 |
| **Llama3.2 3B** | **87.61%** | 41.47 | 1.45 | 338.55 |
| **DeepSeek-R1 1.5B** | **0.00%** | 33.47 | 1.79 | 419.46 |
| **Qwen3 14B** | **0.00%** | 4.51 | 13.31 | 3114.05 |

## Key Findings

### 1. Success Rate Analysis

**Top Performers (>95% success rate):**
- Gemma3 12B-IT-QAT: 96.15% (225/234 successful extractions)
- Gemma3 12B: 96.15% (225/234 successful extractions)
- Phi4 Latest: 95.30% (223/234 successful extractions)
- Gemma3 4B-IT-QAT: 95.73% (224/234 successful extractions)
- Gemma3 4B: 95.73% (224/234 successful extractions)
- Granite3.2 8B: 95.73% (224/234 successful extractions)

**Moderate Performers:**
- Mistral Latest: 85.90% (201/234 successful extractions)
- Llama3.2 3B: 87.61% (205/234 successful extractions)

**Failed Models:**
- DeepSeek-R1 1.5B: 0% success rate (0/234 successful extractions)
- Qwen3 14B: 0% success rate (0/234 successful extractions)

### 2. Processing Speed Analysis

**Fastest Models:**
1. **Llama3.2 3B**: 41.47 files/min (1.45s/file)
2. **DeepSeek-R1 1.5B**: 33.47 files/min (1.79s/file) - but 0% success
3. **Gemma3 4B-IT-QAT**: 29.83 files/min (2.01s/file)
4. **Gemma3 4B**: 25.87 files/min (2.32s/file)

**Balanced Performance:**
- Mistral Latest: 18.65 files/min (3.22s/file)
- Granite3.2 8B: 15.60 files/min (3.85s/file)
- Gemma3 12B-IT-QAT: 12.52 files/min (4.79s/file)

**Slower Models:**
- Gemma3 12B: 11.14 files/min (5.38s/file)
- Phi4 Latest: 10.66 files/min (5.63s/file)
- Qwen3 14B: 4.51 files/min (13.31s/file)

### 3. Error Pattern Analysis

**Error Types by Model:**

| Model | JSON Decode Errors | Missing Keys Errors | No JSON Errors | Other Errors | Fallback Used |
|-------|-------------------|--------------------|-----------------|--------------|---------------|
| DeepSeek-R1 1.5B | 0 | 0 | **675** | 0 | 225 |
| Qwen3 14B | 0 | 0 | **675** | 0 | 225 |
| Mistral Latest | **39** | 6 | 37 | 0 | 24 |
| Llama3.2 3B | 0 | **10** | 56 | 0 | 20 |
| Granite3.2 8B | 0 | **4** | 0 | 0 | 1 |
| Phi4 Latest | 0 | **6** | 0 | 0 | 2 |
| Gemma3 4B-IT-QAT | **3** | 0 | 0 | 0 | 1 |
| Gemma3 4B | **3** | 0 | 0 | 0 | 1 |
| Gemma3 12B-IT-QAT | 0 | 0 | 0 | 0 | 0 |
| Gemma3 12B | 0 | 0 | 0 | 0 | 0 |

**Error Pattern Insights:**
- **DeepSeek-R1 1.5B** and **Qwen3 14B** completely fail to generate valid JSON (675 no_json_errors each)
- **Mistral Latest** has the most JSON decode errors (39), indicating malformed JSON output
- **Llama3.2 3B** and other models show missing keys errors, suggesting incomplete metadata extraction
- **Gemma3 12B variants** show perfect error-free performance

### 4. Library-Specific Performance (Library 1 vs Library 2)

**Average Processing Time Comparison:**

| Model | Library 1 (s) | Library 2 (s) | Difference | Library 2 Penalty |
|-------|---------------|---------------|------------|-------------------|
| Gemma3 12B-IT-QAT | 4.66 | 4.97 | +0.31s | +6.6% |
| Gemma3 12B | 5.17 | 5.67 | +0.50s | +9.7% |
| Phi4 Latest | 5.45 | 5.87 | +0.42s | +7.7% |
| Gemma3 4B-IT-QAT | 1.94 | 2.10 | +0.16s | +8.2% |
| Gemma3 4B | 2.21 | 2.47 | +0.26s | +11.8% |
| Granite3.2 8B | 3.73 | 4.00 | +0.27s | +7.2% |
| Mistral Latest | 3.22 | 3.22 | +0.00s | +0.0% |
| Llama3.2 3B | 1.43 | 1.47 | +0.04s | +2.8% |
| DeepSeek-R1 1.5B | 1.71 | 1.90 | +0.19s | +11.1% |
| Qwen3 14B | 13.07 | 13.63 | +0.56s | +4.3% |

**Library Performance Insights:**
- **Library 2 consistently slower** than Library 1 across all models
- **Mistral Latest** shows no difference between libraries (unusual)
- **Llama3.2 3B** has the smallest penalty (+2.8%) for Library 2
- **Gemma3 4B** has the largest penalty (+11.8%) for Library 2

## Model Rankings

### Overall Performance Ranking (Success Rate × Speed)

1. **Gemma3 4B-IT-QAT** - Best balance of high success rate (95.73%) and fast speed (29.83 files/min)
2. **Gemma3 4B** - High success rate (95.73%) with good speed (25.87 files/min)
3. **Gemma3 12B-IT-QAT** - Highest success rate (96.15%) but moderate speed (12.52 files/min)
4. **Granite3.2 8B** - High success rate (95.73%) with reasonable speed (15.60 files/min)
5. **Gemma3 12B** - Highest success rate (96.15%) but slower (11.14 files/min)
6. **Phi4 Latest** - High success rate (95.30%) but slower (10.66 files/min)
7. **Llama3.2 3B** - Fastest processing (41.47 files/min) but lower success rate (87.61%)
8. **Mistral Latest** - Moderate success rate (85.90%) and speed (18.65 files/min)
9. **DeepSeek-R1 1.5B** - Fast but completely ineffective (0% success)
10. **Qwen3 14B** - Slow and completely ineffective (0% success)

## Recommendations

### For Production Use:
1. **Gemma3 4B-IT-QAT** - Optimal choice for most scenarios (high accuracy, fast processing)
2. **Gemma3 12B-IT-QAT** - Best for accuracy-critical applications where speed is less important
3. **Granite3.2 8B** - Good alternative with balanced performance

### For High-Volume Processing:
- **Gemma3 4B-IT-QAT** offers the best throughput while maintaining high accuracy
- Avoid DeepSeek-R1 1.5B and Qwen3 14B due to complete failure in JSON generation

### For Quality-Critical Applications:
- **Gemma3 12B variants** provide the highest success rates with minimal errors
- Consider the processing speed trade-off for your specific use case

## Technical Notes

- All models were tested on the same 234 PDF files
- Processing includes both text extraction and metadata generation
- Library 1 and Library 2 likely represent different PDF processing backends
- Success rate is based on successful JSON metadata extraction with required fields
- Models showing 0% success rate appear to have fundamental issues with JSON format generation