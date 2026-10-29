# ContextWorld 论文

英文主稿为 `main_en.tex`，可直接阅读[英文 PDF](build/main_en.pdf)。`main.tex` 保留中文稿作为写作参考。两份源文件均记录中英文题目。

## 文件

| 文件 | 用途 |
|---|---|
| `main.tex` | 中文参考稿与编译入口 |
| `main_en.tex` | 英文主稿与编译入口 |
| `build/main_en.pdf` | 英文预览，随源稿同步更新 |
| `references.bib` | 参考文献 |
| `figures/` | 图片资源 |
| `cvpr.sty`、`ieeenat_fullname.bst` | CVPR 样式与参考文献格式 |

## 编译与预览

使用 TeX Live 2025 和 `latexmk`，在本目录编译英文稿：

```bash
make en
```

英文稿使用 pdfLaTeX，输出为 `build/main_en.pdf`。中文参考稿使用 XeLaTeX，通过 `make zh` 编译，输出为 `build/main.pdf`。

当前环境若未设置编译器路径，可先执行：

```bash
export PATH="/opt/texlive/2025/bin/x86_64-linux:$PATH"
```

在 Overleaf 中，中文稿选择 `main.tex` 和 XeLaTeX；英文稿选择 `main_en.tex` 和 pdfLaTeX。两者均使用 TeX Live 2025。中文字体使用 TeX Live 自带的 Fandol。

## 模板来源

样式文件原样取自 [CVPR 官方 author-kit](https://github.com/cvpr-org/author-kit/tree/CVPR2026-v1(latex)) 的 `CVPR2026-v1(latex)` 发布版。工程的目标会议年份为 2027；正式投稿时需按 CVPR 2027 官方模板与要求设置。中文双栏预览的篇幅不用于判断英文投稿页数。
