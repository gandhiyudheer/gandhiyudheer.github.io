# gandhiyudheer.github.io

Personal website and CV of Yogesh Gandhi, Senior Researcher at Braid Technologies K.K., Tokyo.

Live at: https://gandhiyudheer.github.io

## Stack

Plain HTML and CSS. No build step, no framework.

- `index.html`, single page with all sections
- `styles.css`, styling (warm cream, serif display, auto dark mode)
- `cv.html`, industry CV, two pages. This is the source of truth for the CV.
- `cv-research.html`, research CV, three pages, with the full publication list
- `cv.pdf`, generated from `cv.html`, kept for anyone who wants a file to download

Fonts from Google Fonts: Fraunces (display), IBM Plex Sans (body), JetBrains Mono (details).

## Regenerating cv.pdf

`cv.pdf` is generated, not hand edited. After changing `cv.html`:

```bash
google-chrome --headless=new --disable-gpu --no-sandbox \
  --no-pdf-header-footer --print-to-pdf=cv.pdf file://$PWD/cv.html
```

Then check it still fits two pages with `pdfinfo cv.pdf`. If it has spilled onto a third,
the cause is usually the education block failing to split across the page break rather
than length alone, so trim a line or two from the profile or the first role, above that
block, not below it.

## Local preview

```bash
python3 -m http.server 8000
```

Open http://localhost:8000

## Deploy

Any push to `main` is automatically published to GitHub Pages.
