# Citation Guide for DeepCausalMMM

## 📚 What is CITATION.cff?

The `CITATION.cff` (Citation File Format) is a standard way to provide citation information for software. It allows:

- **GitHub** to display a "Cite this repository" button
- **Zenodo** to automatically generate citations
- **Researchers** to easily cite your software correctly
- **Citation managers** (Zotero, Mendeley) to import metadata

## ✅ What We've Created

**File:** `CITATION.cff` in the root directory

**Key Information:**
- Version: 1.0.17
- DOI: 10.5281/zenodo.16934851
- Author: Aditya Puttaparthi Tirumala
- ORCID: 0009-0008-9495-3932
- License: MIT
- Release Date: October 5, 2025

## 🔍 How It Appears on GitHub

Once pushed, GitHub will show a **"Cite this repository"** button on your repo page that displays:

### APA Format:
```
Puttaparthi Tirumala, A. (2025). DeepCausalMMM: Deep Learning and Causal 
Inference for Marketing Mix Modeling (Version 1.0.17) [Computer software]. 
https://doi.org/10.5281/zenodo.16934851
```

### BibTeX Format:
```bibtex
@software{Puttaparthi_Tirumala_DeepCausalMMM_2025,
  author = {Puttaparthi Tirumala, Aditya},
  doi = {10.5281/zenodo.16934851},
  month = oct,
  title = {{DeepCausalMMM: Deep Learning and Causal Inference for Marketing Mix Modeling}},
  url = {https://github.com/adityapt/deepcausalmmm},
  version = {1.0.17},
  year = {2025}
}
```

## 📝 How to Cite DeepCausalMMM

### For Papers/Publications:

**APA Style:**
```
Puttaparthi Tirumala, A. (2025). DeepCausalMMM: Deep Learning and Causal 
Inference for Marketing Mix Modeling (Version 1.0.17) [Computer software]. 
https://doi.org/10.5281/zenodo.16934851
```

**Chicago Style:**
```
Puttaparthi Tirumala, Aditya. 2025. DeepCausalMMM: Deep Learning and Causal 
Inference for Marketing Mix Modeling. Version 1.0.17. 
https://doi.org/10.5281/zenodo.16934851.
```

**MLA Style:**
```
Puttaparthi Tirumala, Aditya. DeepCausalMMM: Deep Learning and Causal Inference 
for Marketing Mix Modeling. Version 1.0.17, 2025, 
doi:10.5281/zenodo.16934851.
```

### For Code/Documentation:

**In README or Documentation:**
```markdown
## Citation

If you use DeepCausalMMM in your research, please cite:

Puttaparthi Tirumala, A. (2025). DeepCausalMMM: Deep Learning and Causal 
Inference for Marketing Mix Modeling (Version 1.0.17) [Computer software]. 
https://doi.org/10.5281/zenodo.16934851
```

**In Python Code:**
```python
"""
This code uses DeepCausalMMM:
Puttaparthi Tirumala, A. (2025). DeepCausalMMM v1.0.17.
DOI: 10.5281/zenodo.16934851
"""
```

## 🔄 Updating for Future Releases

### When releasing a new version (e.g., v1.0.18):

1. **Update version number:**
   ```yaml
   version: "1.0.18"
   ```

2. **Update release date:**
   ```yaml
   date-released: 2025-11-15  # New release date
   ```

3. **Update DOI (if new Zenodo release):**
   ```yaml
   doi: 10.5281/zenodo.XXXXXXX  # New DOI from Zenodo
   ```

4. **Commit and push:**
   ```bash
   git add CITATION.cff
   git commit -m "Update citation for v1.0.18"
   git push origin main
   ```

## 🔗 Integrating with Zenodo

### Step 1: Link GitHub to Zenodo

1. Go to https://zenodo.org/
2. Log in with GitHub
3. Go to Settings → GitHub
4. Enable `deepcausalmmm` repository

### Step 2: Create Release on GitHub

1. Go to https://github.com/adityapt/deepcausalmmm/releases
2. Click "Draft a new release"
3. Choose tag: v1.0.17
4. Publish release

### Step 3: Get DOI from Zenodo

1. Zenodo automatically creates archive
2. Get DOI (format: 10.5281/zenodo.XXXXXXX)
3. Update `CITATION.cff` with new DOI

### Step 4: Update CITATION.cff

```yaml
doi: 10.5281/zenodo.XXXXXXX  # Replace with actual DOI
```

## 📊 Tracking Citations

### Google Scholar

1. Create Google Scholar profile
2. Add DeepCausalMMM as a publication
3. Track citations automatically

### Zenodo

1. View your Zenodo record
2. See download statistics
3. Track citations via DOI

### GitHub

1. Check "Used by" section on GitHub
2. View dependent repositories
3. Monitor stars and forks

## ✅ Validation

### Validate your CITATION.cff:

**Online Validator:**
```
https://citation-file-format.github.io/cff-initializer-javascript/
```

**Command Line:**
```bash
pip install cffconvert
cffconvert --validate
```

**GitHub Action:**
Add to `.github/workflows/validate-cff.yml`:
```yaml
name: Validate CITATION.cff
on: [push, pull_request]
jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: citation-file-format/cffconvert-github-action@2.0.0
        with:
          args: "--validate"
```

## 📚 Additional Resources

### Official Documentation:
- **CFF Specification**: https://citation-file-format.github.io/
- **GitHub Guide**: https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files
- **Zenodo Guide**: https://help.zenodo.org/

### Tools:
- **CFF Initializer**: https://citation-file-format.github.io/cff-initializer-javascript/
- **cffconvert**: https://github.com/citation-file-format/cff-converter-python
- **Zenodo**: https://zenodo.org/

## 🎯 Best Practices

1. **Keep Updated**: Update version and date with each release
2. **Include ORCID**: Always include author ORCID IDs
3. **Add Keywords**: Help people discover your software
4. **Write Abstract**: Explain what your software does
5. **Link to Paper**: Reference JOSS paper when published
6. **Validate**: Check format before committing

## 📝 Example Citation in Papers

### In Methods Section:
```
We used DeepCausalMMM (Puttaparthi Tirumala, 2025) for marketing mix 
modeling. The package implements GRU-based temporal modeling and DAG 
causal discovery for analyzing marketing channel effectiveness.
```

### In References:
```
Puttaparthi Tirumala, A. (2025). DeepCausalMMM: Deep Learning and Causal 
Inference for Marketing Mix Modeling (Version 1.0.17) [Computer software]. 
https://doi.org/10.5281/zenodo.16934851
```

## 🎉 Benefits

**For Users:**
- ✅ Easy to cite your software correctly
- ✅ One-click citation export
- ✅ Multiple format support

**For You:**
- ✅ Track software citations
- ✅ Get academic credit
- ✅ Professional appearance
- ✅ JOSS requirement met

---

## 🚀 Next Steps

1. **Push CITATION.cff to GitHub**
   ```bash
   git add CITATION.cff
   git commit -m "Add CITATION.cff for v1.0.17"
   git push origin main
   ```

2. **Verify on GitHub**
   - Check for "Cite this repository" button
   - Test citation export

3. **Link to Zenodo** (optional but recommended)
   - Get DOI for permanent archival
   - Update CITATION.cff with DOI

4. **Add to README**
   - Include citation section
   - Link to CITATION.cff

---

**Your CITATION.cff is ready!** 🎊
