# 🤖 GitHub Actions Auto-Fix Setup Instructions

## 📋 Požiadavky

### 1. GitHub Repository Settings
- Repository musí mať zapísané práva (write access)
- GitHub Actions musia byť povolené
- Issues a Pull Requests musia byť povolené

### 2. Secrets Configuration
Potrebujete nastaviť tieto secrets v repository settings:

```
Settings → Secrets and variables → Actions → New repository secret
```

**Požadované secrets:**
- `CLAUDE_API_KEY` - API kľúč pre Claude AI (Anthropic)

### 3. GitHub Token Permissions
GitHub Actions automaticky používa `GITHUB_TOKEN` s týmito potrebnými permissions:
- `contents: write` - pre modifikáciu súborov
- `pull-requests: write` - pre vytváranie PR
- `issues: write` - pre komentovanie issues

## 🚀 Aktivácia

### Krok 1: Commit súborov
```bash
git add .github/
git commit -m "Add GitHub Actions auto-fix system"
git push origin main
```

### Krok 2: Nastavenie Claude API Key
1. Získajte API kľúč z [Anthropic Console](https://console.anthropic.com/)
2. V GitHub repository: `Settings → Secrets and variables → Actions`
3. Kliknite `New repository secret`
4. Name: `CLAUDE_API_KEY`
5. Secret: `sk-ant-api03-...` (váš API kľúč)

### Krok 3: Test workflow
1. Vytvorte nový issue s template "🤖 Auto-Fix Bug Report"
2. Vyplňte detail a označte prioritu
3. Pridajte label `auto-fix`
4. Workflow sa automaticky spustí

## 📝 Ako používať

### Pre Bug Reports:
1. Použite template "🤖 Auto-Fix Bug Report"
2. Popíšte problém detailne
3. Označte affected files
4. Pridajte label `auto-fix`
5. AI vytvorí PR s riešením

### Pre Enhancements:
1. Použite template "🚀 Auto-Fix Enhancement"  
2. Popíšte požadované vylepšenie
3. Špecifikujte acceptance criteria
4. Pridajte label `auto-fix`
5. AI implementuje enhancement

## 🧪 Testing

### Manuálne spustenie:
```yaml
# V GitHub Actions tab
Actions → Auto-Fix Issues → Run workflow
# Zadajte issue number
```

### Debug logs:
- Všetky kroky sú logované v GitHub Actions
- Claude AI response sa ukladá ako artifact
- Failed fixes dostanú komentár s vysvetlením

## 🔧 Konfigurácia

### Úprava templates:
```
.github/ISSUE_TEMPLATE/auto-fix-bug.md
.github/ISSUE_TEMPLATE/auto-fix-enhancement.md
```

### Úprava workflow:
```
.github/workflows/auto-fix-issues.yml
```

### Úprava Claude AI action:
```
.github/actions/claude-ai-fix/action.yml
```

## 🚨 Limitations

### Čo AI dokáže opraviť:
- ✅ Jednoduché syntax errors
- ✅ Parameter changes (chunk sizes, timeouts)
- ✅ Pridanie error handling
- ✅ Import fixes
- ✅ Configuration updates

### Čo vyžaduje manual review:
- ❌ Komplexné algorithmic changes
- ❌ Database schema changes  
- ❌ Security-related fixes
- ❌ API breaking changes

## 🔐 Security

### Bezpečnostné opatrenia:
- AI nemôže modifikovať `.github/` súbory
- Všetky changes prebiehajú cez PR (nie direct commit)
- Syntax validation pred vytvorením PR
- Manual review requirement pred merge

### Claude API:
- API kľúč je stored ako GitHub secret
- Žiadne sensitive data sa neposielajú do Claude
- Iba issue description a relevant code context

## 📊 Monitoring

### GitHub Actions insights:
- `Actions` tab zobrazuje všetky runs
- Failed runs majú detailed logs
- Success rate tracking

### Issue comments:
- AI automaticky komentuje progress
- Success/failure notifications
- Link na vytvorený PR

## 🛠️ Troubleshooting

### Workflow sa nespustil:
- Skontrolujte či má issue label `auto-fix`
- Overte GitHub Actions permissions
- Pozrite Actions tab pre error logs

### Claude API errors:
- Skontrolujte `CLAUDE_API_KEY` secret
- Overte API quota limits
- Pozrite workflow logs pre details

### PR creation failed:
- Skontrolujte token permissions
- Overte branch protection rules
- Pozrite či branch name nie je duplicated

---

## 🎯 Next Steps

Po úspešnom setup môžete:

1. **Testovať systém** s jednoduchým issue
2. **Customizovať templates** pre vaše potreby  
3. **Rozšíriť AI capabilities** o ďalšie fix patterns
4. **Integrovať testing** do workflow
5. **Pridať notification** channels (Slack, email)

Happy auto-fixing! 🤖✨