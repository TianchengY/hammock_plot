# Hammock Plot Gallery Examples

Each script in this folder reads one dataset from `data/` and writes a gallery-ready PNG to `image/gallery/`.

Run one example:

```powershell
python examples/gallery/penguins_mixed_displays.py
```

Run all examples:

```powershell
Get-ChildItem examples/gallery/*.py | ForEach-Object { python $_.FullName }
```

Build the Sphinx gallery:

```powershell
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```
