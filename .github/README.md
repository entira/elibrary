# 🤖 GitHub Actions Auto-Fix System

Automatizovaný systém na riešenie issues pomocou Claude AI.

## 📁 Štruktúra

```
.github/
├── workflows/
│   └── auto-fix-issues.yml          # Hlavný workflow
├── actions/
│   └── claude-ai-fix/
│       └── action.yml               # Custom action pre Claude AI
├── ISSUE_TEMPLATE/
│   ├── auto-fix-bug.md             # Template pre bug reports
│   └── auto-fix-enhancement.md     # Template pre enhancements
├── SETUP_INSTRUCTIONS.md           # Detailné setup inštrukcie
├── test_auto_fix.py               # Test script
└── README.md                      # Tento súbor
```

## 🚀 Rýchly štart

### 1. Setup
```bash
# Pridajte Claude API key do repository secrets
gh secret set CLAUDE_API_KEY -b"sk-ant-api03-your-key-here"

# Commit GitHub Actions súbory
git add .github/
git commit -m "Add auto-fix system"
git push
```

### 2. Použitie
```bash
# Vytvorte issue s auto-fix template
# Pridajte label "auto-fix"
# AI automaticky vytvorí PR s riešením
```

### 3. Test
```bash
# Spustite test script
export GITHUB_REPOSITORY="username/repo"
export GITHUB_TOKEN="your-token"
python3 .github/test_auto_fix.py
```

## 🔧 Konfigurácia

### Required Secrets
- `CLAUDE_API_KEY` - API kľúč pre Claude AI

### Workflow Permissions
- `contents: write` - modifikácia súborov
- `pull-requests: write` - vytváranie PR
- `issues: write` - komentovanie issues

## 📋 Supported Issue Types

### 🐛 Bug Fixes
- Syntax errors
- Configuration issues  
- Parameter adjustments
- Import problems

### 🚀 Enhancements
- Adding error handling
- Logging improvements
- Performance optimizations
- Code refactoring

## 🧪 Testing

### Automated Tests
```bash
# GitHub Actions workflow includes:
- Syntax validation
- Import testing
- Basic functionality checks
```

### Manual Testing
```bash
# Create test issues:
python3 .github/test_auto_fix.py

# Check workflow status:
gh run list --workflow="auto-fix-issues.yml"

# View logs:
gh run view --log
```

## 🔍 Monitoring

### GitHub Actions Dashboard
- `Actions` tab - všetky workflow runs
- Real-time logs a status
- Success/failure metrics

### Issue Tracking
- AI komentuje progress na issues
- Automatic PR linking
- Status updates

## 🛡️ Security

### Bezpečnostné opatrenia
- ✅ Všetky changes cez PR (nie direct push)
- ✅ Required manual review pred merge
- ✅ Syntax validation pred PR creation
- ✅ Limited file access (nie .github/)
- ✅ Secrets properly protected

### Claude AI Integration
- ✅ Žiadne sensitive data v prompts
- ✅ Iba public issue content + code context
- ✅ API key stored as GitHub secret
- ✅ Rate limiting a error handling

## 📊 Analytics

### Success Metrics
```bash
# Workflow success rate
gh run list --workflow="auto-fix-issues.yml" --json conclusion

# Issue resolution time
gh issue list --label="auto-fix" --state=closed

# PR merge rate
gh pr list --label="auto-fix" --state=merged
```

## 🔧 Customization

### Pridanie nových fix patterns
```yaml
# V .github/actions/claude-ai-fix/action.yml
# Rozšíriť logic pre specific issue types
```

### Custom issue templates
```markdown
# Vytvorte nové templates v .github/ISSUE_TEMPLATE/
# Použite existing templates ako základ
```

### Workflow modifications
```yaml
# Upravte .github/workflows/auto-fix-issues.yml
# Pridajte additional testing steps
# Integrate s external tools
```

## 🚨 Troubleshooting

### Bežné problémy

#### Workflow sa nespustil
```bash
# Check labels
gh issue view ISSUE_NUMBER --json labels

# Check workflow permissions
gh api repos/:owner/:repo/actions/permissions
```

#### Claude API errors
```bash
# Verify secret
gh secret list

# Check API quota
# Visit Anthropic console
```

#### PR creation failed
```bash
# Check token permissions
gh auth status

# Verify branch protection
gh api repos/:owner/:repo/branches/main/protection
```

## 📚 Resources

### Documentation
- [GitHub Actions Docs](https://docs.github.com/en/actions)
- [Claude API Docs](https://docs.anthropic.com/claude/reference)
- [Setup Instructions](.github/SETUP_INSTRUCTIONS.md)

### Examples
- [Test Script](.github/test_auto_fix.py)
- [Issue Templates](.github/ISSUE_TEMPLATE/)
- [Workflow Examples](.github/workflows/)

---

## 🤝 Contributing

Chcete vylepšiť auto-fix systém?

1. Fork repository
2. Vytvorte feature branch  
3. Testujte changes s test script
4. Vytvorte PR s popisom improvements

---

💡 **Tip:** Začnite s jednoduchými issues na testovanie systému pred použitím na complex problems.

🤖 **AI-Powered** • ⚡ **Fast** • 🔒 **Secure** • 📈 **Scalable**