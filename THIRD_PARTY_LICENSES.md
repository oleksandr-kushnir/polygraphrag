# Third-Party Licenses

PolyGraphRAG's own source code is MIT-licensed (see [LICENSE](LICENSE)). This
file inventories the third-party components PolyGraphRAG depends on and their
licenses.

**When this matters.** Installing from source (`pip install -r requirements.txt`)
downloads each dependency from PyPI under its own license — you are not
redistributing them, so their attribution terms are satisfied by upstream. The
obligations below apply when you **redistribute a built artifact that bundles
these dependencies** — most importantly the **Docker image** produced by
`Dockerfile` / `docker compose build`. If you publish that image (e.g. to a
container registry) or otherwise ship the dependencies as binaries, ship this
file (or the per-package license texts) with it.

> This list is curated from the declared package metadata. Before publishing a
> redistributable image, regenerate the exact resolved set from a full install:
>
> ```bash
> pip install pip-licenses
> pip-licenses --format=markdown --with-urls --with-license-file \
>   --output-file THIRD_PARTY_LICENSES_FULL.md
> ```
>
> The heavy `raganything[all]` / `mineru` stack has a large transitive tree;
> the generated file is the authoritative inventory for a given build.

---

## ⚠️ Components with obligations beyond simple attribution

### MinerU — attribution required (license auto-terminates otherwise)

- **Package:** `mineru` (pulled in transitively: `raganything[all]` → `mineru[core]`)
- **Role:** document layout parsing / OCR for uploaded files.
- **License:** [MinerU Open Source License](https://github.com/opendatalab/MinerU/blob/master/LICENSE.md) — a custom license based on Apache-2.0 (MinerU ≥ 3.1.0). **Earlier releases were AGPL-3.0**; `requirements.txt` pins `mineru>=3.1.0` to stay on the Apache-based license.
- **Obligation:** If you provide an online service to third parties based on
  MinerU, you must **clearly and prominently indicate** — in the service
  interface or public documentation — that MinerU is used. Non-compliance
  terminates the license automatically. A separate commercial license is
  required only above **100M monthly active users** or **USD 20M monthly
  revenue** (not applicable to typical deployments).
- **How PolyGraphRAG complies:** MinerU is credited in [NOTICE](NOTICE), in the
  README "Credits & license" section, and in the README ingestion docs
  ("Powered by MinerU"). **If you fork or rebrand this service, keep a visible
  "powered by MinerU" attribution.**

### psycopg2-binary — LGPL (weak copyleft)

- **Package:** `psycopg2-binary`
- **License:** LGPL-3.0-or-later, with a linking/loader exception.
- **Obligation when bundled (e.g. in the Docker image):** include the LGPL
  license text, note that psycopg2 is used and where its source is available
  (<https://github.com/psycopg/psycopg2> / PyPI), and do not prevent recipients
  from replacing it with a modified version. Using it unmodified as a library —
  as PolyGraphRAG does — is fully permitted under the exception; no PolyGraphRAG
  source needs to be released.
- **Note:** `psycopg2-binary` is currently listed in `requirements.txt` but is
  **not imported by PolyGraphRAG's code** (the Postgres access path uses
  `asyncpg`, Apache-2.0). If it is confirmed unnecessary for the LightRAG /
  Apache AGE runtime, removing it eliminates the only copyleft dependency and
  this obligation entirely.

---

## Permissive dependencies (attribution only when redistributed)

Under MIT / BSD / Apache-2.0 / HPND / PostgreSQL licenses, redistribution
requires preserving the copyright and license notice (and, for Apache-2.0, any
`NOTICE` file and a statement of changes if you modified the code — PolyGraphRAG
does not modify these). No source disclosure is required.

### Direct runtime dependencies (`requirements.txt`)

| Package | License | Project |
|---|---|---|
| raganything | MIT | https://github.com/HKUDS/RAG-Anything |
| lightrag-hku | MIT | https://github.com/HKUDS/LightRAG |
| fastapi | MIT | https://github.com/fastapi/fastapi |
| uvicorn | BSD-3-Clause | https://github.com/encode/uvicorn |
| python-multipart | Apache-2.0 | https://github.com/Kludex/python-multipart |
| asyncpg | Apache-2.0 | https://github.com/MagicStack/asyncpg |
| pyvis | BSD-3-Clause (bundles vis-network, MIT/Apache-2.0) | https://github.com/WestHealth/pyvis |

`pyvis` embeds the vis-network JavaScript library into the generated
`graph.html`; that bundled JS (dual MIT / Apache-2.0) is therefore shipped in
any exported graph HTML.

### Notable transitive dependencies (via `raganything[all]` / `mineru[core]`)

| Package | License | Project |
|---|---|---|
| transformers | Apache-2.0 | https://github.com/huggingface/transformers |
| huggingface_hub | Apache-2.0 | https://github.com/huggingface/huggingface_hub |
| modelscope | Apache-2.0 | https://github.com/modelscope/modelscope |
| paddleocr | Apache-2.0 | https://github.com/PaddlePaddle/PaddleOCR |
| opencv-python | Apache-2.0 (MIT packaging) | https://github.com/opencv/opencv-python |
| pypdfium2 | Apache-2.0 / BSD-3-Clause (bundles PDFium, BSD-3) | https://github.com/pypdfium2-team/pypdfium2 |
| Pillow | HPND (MIT-style) | https://github.com/python-pillow/Pillow |
| reportlab | BSD-3-Clause | https://www.reportlab.com/ |
| weasyprint | BSD-3-Clause | https://github.com/Kozea/WeasyPrint |
| markdown | BSD-3-Clause | https://github.com/Python-Markdown/markdown |
| pygments | BSD-2-Clause | https://github.com/pygments/pygments |
| lxml | BSD-3-Clause (bundles libxml2/libxslt, MIT) | https://github.com/lxml/lxml |
| python-docx | MIT | https://github.com/python-openxml/python-docx |
| openpyxl | MIT | https://foss.heptapod.net/openpyxl/openpyxl |
| numpy | BSD-3-Clause | https://github.com/numpy/numpy |

The transitive tree is large and version-dependent; treat `pip-licenses` output
from an actual build as authoritative (see note above).

### System / non-Python components used at runtime

| Component | License | Notes |
|---|---|---|
| PostgreSQL | PostgreSQL License (permissive) | Datastore |
| pgvector | PostgreSQL License | Postgres extension, vector storage |
| Apache AGE | Apache-2.0 | Postgres extension, graph storage |
| LibreOffice | MPL-2.0 | Invoked as an external process for Office → PDF; not bundled or linked |

`LibreOffice` is executed as a separate program (Office document conversion),
not linked into PolyGraphRAG, so its MPL-2.0 terms apply only to LibreOffice's
own files, which PolyGraphRAG does not redistribute.

### Development-only dependencies (`requirements-dev.txt`) — not shipped

| Package | License |
|---|---|
| pydantic | MIT |
| pytest | MIT |
| pytest-asyncio | Apache-2.0 |
| httpx | BSD-3-Clause |

These are test/build tooling only and are not part of any redistributed
artifact, so they carry no distribution obligation.
