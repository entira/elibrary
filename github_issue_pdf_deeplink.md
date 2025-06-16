# GitHub Issue — Add Deep-Link Support to PDF Viewer via PDF.js

## Title
Enable page-level (and optional text search) deep-linking for PDFs using PDF.js

---

## Background / Problem Statement

Our RAG pipeline returns citations with page numbers—and sometimes quoted snippets—that users want to inspect in the source PDF.

At the moment, clicking a citation simply downloads or opens the PDF at page 1, forcing users to scroll manually. This breaks the "instant context" promise of RAG.

---

## User Story

**As a reader who clicks a citation produced by the RAG system,**  
**I want the linked PDF to open directly on the cited page (and, when possible, highlight the quoted text),**  
**so that I immediately see the relevant evidence without manual searching.**

---

## Acceptance Criteria

### 1. Page jump
- **Given** a citation with page = N, **when** clicking the link **then** the PDF opens at page N in the embedded viewer.

### 2. Optional search highlight
- **If** the citation also provides `searchTerm`, **then** the viewer performs a case-insensitive search and highlights all occurrences on that page (first match auto-scrolled into view).

### 3. URL format
- Links follow the pattern:
  ```
  /pdfjs/web/viewer.html?file=<ENCODED_FILE_URL>#page=<N>[&search=<ENCODED_TERM>]
  ```
  where `search` is omitted when no term is supplied.

### 4. Browser support
- **Works** in Chromium-based browsers and Firefox out-of-the-box.
- **In Safari**, page jumps must still function; if search highlighting is unavailable, fail gracefully (no error shown).

### 5. Security & compliance
- The `file` parameter must be URL-encoded and pass through our existing signed-URL / CORS gatekeeper.
- No additional user-visible auth prompts.

### 6. Performance
- Initial render time for a deep-linked page should not exceed baseline viewer load by more than 200 ms (measured on a typical 10-page, 2 MB PDF).

---

## Implementation Notes

### Bundle PDF.js
- Add/update to the latest stable `pdfjs-<version>/web/viewer.html` in `public/pdfjs/`.

### Link builder
```javascript
function pdfLink(fileUrl: string, page: number, term?: string) {
  const base = '/pdfjs/web/viewer.html';
  const params = new URLSearchParams({
    file: fileUrl
  });
  const hash = [`page=${page}`];
  if (term) hash.push(`search=${encodeURIComponent(term)}`);
  return `${base}?${params.toString()}#${hash.join('&')}`;
}
```

### Search highlight
- PDF.js viewer automatically triggers its find-bar when `search=<term>` is present in the hash; no extra code required.

### Fallback for native viewers
- If PDF.js cannot be loaded (network error, ad-block, etc.), redirect to `fileUrl#page=<N>` which most browsers honour as a secondary strategy.

### Testing
- Add Cypress tests that assert the correct page number from `window.PDFViewerApplication.page`.
- Provide fixtures with and without the target term present.

---

## Out of Scope
- Generating named-destinations inside PDFs.
- Modifying or re-writing existing RAG citations.

---

## References / Resources
- [PDF.js Viewer Parameters](https://github.com/mozilla/pdf.js/wiki/Viewer-options)
- [Adobe PDF Open Parameters Overview](https://www.adobe.com/content/dam/acom/en/devnet/acrobat/pdfs/pdf_open_parameters.pdf)

---

## Technical Implementation Details

### Current State Analysis
Our memvid RAG system generates citations in format:
```
[Book Title, page X - Library Y]
```

### Proposed Citation Enhancement
```javascript
// Current citation data structure
const citation = {
  title: "Book Title",
  page: 42,
  library: "Library 1",
  text_snippet: "quoted text from the document" // optional
};

// Enhanced link generation
const pdfUrl = generateSignedPdfUrl(citation.library, citation.filename);
const deepLink = pdfLink(pdfUrl, citation.page, citation.text_snippet);
```

### Integration Points
1. **PDF Chat Interface** (`pdf_chat.py`)
   - Modify `_add_citations_to_context()` to include deep-link URLs
   - Update citation format to include clickable links

2. **Web Interface** (if applicable)
   - Add PDF.js viewer component
   - Implement link handler for citations

3. **Citation Display**
   ```html
   <a href="/pdfjs/web/viewer.html?file=library%2F1%2Fdata%2Flibrary.pdf#page=42&search=quoted%20text"
      target="_blank">
     [Book Title, page 42 - Library 1] 🔗
   </a>
   ```

### Security Considerations
- PDF files served through existing signed URL mechanism
- CORS headers properly configured for PDF.js access
- URL encoding to prevent injection attacks
- File path validation to prevent directory traversal

### Performance Optimizations
- Lazy load PDF.js bundle
- Preload common PDF files
- Cache PDF.js viewer instance
- Progressive loading for large PDFs

---

## Effort Estimation

**Story Points: 8**

### Breakdown:
- **Setup PDF.js bundle**: 1 point
- **Implement link builder function**: 1 point  
- **Integrate with citation system**: 3 points
- **Add browser compatibility layer**: 1 point
- **Security and CORS configuration**: 1 point
- **Testing (unit + integration)**: 1 point

**Timeline: 1 sprint (2 weeks)**

---

## Definition of Done
- [ ] PDF.js viewer integrated and deployed
- [ ] Deep-linking works for page navigation
- [ ] Text search highlighting functional
- [ ] Cross-browser compatibility verified (Chrome, Firefox, Safari)
- [ ] Security review completed
- [ ] Performance benchmarks meet criteria (<200ms overhead)
- [ ] Unit tests written and passing
- [ ] Integration tests cover citation deep-linking
- [ ] Documentation updated with usage examples
- [ ] Deployed to staging environment
- [ ] Product owner acceptance testing completed

---

**Priority: High**  
**Labels: enhancement, rag-pipeline, user-experience, pdf-viewer**  