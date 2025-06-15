# License Analysis for eLibrary Project

## Overview

This document analyzes the licensing requirements for making the eLibrary project public, considering all dependencies and development tools used.

## Dependency License Analysis

### Core Dependencies

| Library | License | Commercial Use | Copyleft | Notes |
|---------|---------|----------------|----------|-------|
| **memvid** | MIT | ✅ Yes | ❌ No | Permissive, allows commercial use |
| **pymupdf** | AGPL v3.0 / Commercial | ⚠️ Conditional | ✅ Yes | **Critical**: AGPL requires open source |
| **requests** | Apache 2.0 | ✅ Yes | ❌ No | Permissive, patent protection |
| **tqdm** | MIT / MPL-2.0 | ✅ Yes | ⚠️ Partial | Dual license, mostly permissive |
| **tiktoken** | MIT | ✅ Yes | ❌ No | OpenAI library, permissive |
| **qrcode[pil]** | MIT | ✅ Yes | ❌ No | Includes Pillow (PIL) dependency |
| **opencv-python** | Apache 2.0 / LGPL | ✅ Yes | ⚠️ Partial | Contains LGPL components (FFmpeg) |

### Development Tools

| Tool | License | Impact |
|------|---------|---------|
| **Claude Code** | Anthropic Commercial Terms | Code generated can be used commercially under appropriate terms |

## Critical Licensing Issue: PyMuPDF (AGPL v3.0)

### The Problem
PyMuPDF is licensed under **AGPL v3.0**, which is a **strong copyleft license** that requires:

1. **Source Code Disclosure**: Any software using PyMuPDF must be distributed under AGPL v3.0
2. **Network Copyleft**: Even if software is used as a web service, source code must be made available
3. **Compatibility**: AGPL is incompatible with most permissive licenses for commercial use

### Impact on Project
- ✅ **Open Source Projects**: Perfect for open source projects
- ❌ **Commercial Products**: Cannot be used in proprietary commercial software
- ✅ **SaaS Applications**: Can be used but source code must be made available

### Solutions

#### Option 1: Keep PyMuPDF (Recommended for Open Source)
- **License Project**: AGPL v3.0 or GPL v3.0
- **Benefit**: Can use all current dependencies
- **Limitation**: Anyone using this code must also open source their projects

#### Option 2: Replace PyMuPDF with Commercial License
- **Cost**: Contact Artifex Software for commercial licensing
- **Benefit**: Can use permissive license (MIT/Apache 2.0)
- **Limitation**: Licensing fees required

#### Option 3: Replace PyMuPDF with Alternative
- **Alternative**: `pdfplumber`, `PDFMiner`, or `pdftotext`
- **Benefit**: Can use permissive license
- **Limitation**: May require code changes and potentially lower quality text extraction

## Recommended License Options

### Option A: AGPL v3.0 (Recommended)
```
GNU Affero General Public License v3.0
```

**Pros:**
- ✅ Compatible with all current dependencies
- ✅ Ensures project remains open source
- ✅ Strong copyleft protection
- ✅ Network copyleft protects against SaaS exploitation

**Cons:**
- ❌ Cannot be used in proprietary commercial software
- ❌ Derivative works must be AGPL-licensed

**Best For:** Open source projects, research, educational use

### Option B: MIT License + PyMuPDF Replacement
```
MIT License
```

**Pros:**
- ✅ Most permissive and business-friendly
- ✅ Can be used in any commercial project
- ✅ Simple and well-understood

**Cons:**
- ❌ Requires replacing PyMuPDF
- ❌ Potential text extraction quality loss

**Best For:** Maximum compatibility and commercial adoption

## Claude Code Usage Rights

### Commercial Use
- ✅ **Code Generation**: Generated code can be used commercially
- ✅ **Integration**: Can be integrated into commercial products
- ⚠️ **Attribution**: Should include attribution to development process

### Recommendations
- Include acknowledgment of Claude Code usage in development
- Generated code is owned by the user under commercial terms
- No licensing restrictions on the generated code itself

## Final Recommendation

### For Public Open Source Release: AGPL v3.0

**Rationale:**
1. **Compatibility**: Works with all current dependencies including PyMuPDF
2. **Protection**: Ensures the project remains open source
3. **Community**: Encourages open source contributions
4. **Quality**: Can keep high-quality PyMuPDF text extraction

### License Text
```
GNU AFFERO GENERAL PUBLIC LICENSE
Version 3, 19 November 2007

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.
```

### Additional Files Needed

1. **LICENSE** - Full AGPL v3.0 license text
2. **NOTICE** - Attribution for dependencies and development tools
3. **README Update** - License section explaining terms

## Alternative Path for Commercial Use

If commercial licensing flexibility is required:

1. Replace PyMuPDF with `pdfplumber` or similar
2. Use MIT license
3. Note the trade-off in text extraction quality

## Compliance Requirements

### For AGPL v3.0 License:
- ✅ Include license file
- ✅ Include copyright notices
- ✅ Provide source code access
- ✅ Include installation/modification instructions
- ✅ Network copyleft compliance for any hosted versions

### For Dependencies:
- ✅ Include attribution for all dependencies
- ✅ Respect original license terms
- ✅ Include third-party license notices

## Conclusion

The **AGPL v3.0 license is the recommended choice** for this project as it:
- Ensures legal compatibility with PyMuPDF
- Maintains the high-quality text extraction capabilities
- Protects the open source nature of the project
- Encourages community contributions

This makes the project suitable for open source use, research, education, and commercial SaaS applications where source code disclosure is acceptable.