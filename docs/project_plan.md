# B-PINNs 报告翻译项目实施方案

## 1. 项目目标
对 `report.pdf` 进行逐字逐句的精确翻译，保留原文的格式、公式和图表引用，生成高质量的中文 PDF 报告。

## 2. 工作流程
1.  **文档分割**：将原始 PDF 按照章节分割为多个独立的小文件，便于管理和对照翻译。
2.  **环境搭建**：建立 LaTeX 项目结构，主文件 (`main.tex`) 包含所有子章节文件，确保统一的编译和样式。
3.  **逐章翻译**：
    *   针对每个分割后的 PDF 章节，创建对应的 LaTeX `.tex` 文件。
    *   读取原文内容，进行逐字逐句翻译。
    *   将翻译内容写入 `.tex` 文件，保留 LaTeX 格式（章节标题、公式、引用等）。
4.  **编译与校对**：
    *   编译主文件生成 PDF。
    *   对照原始 PDF 进行校对，修正翻译错误和格式问题。
5.  **进度追踪**：实时更新 `progress.md`，记录每一章的翻译状态。

## 3. 目录结构
```
docs/
├── project_plan.md       # 项目实施方案
├── progress.md           # 施工进度表
├── source_pdfs/          # 分割后的原始 PDF 章节
│   ├── 00_frontmatter.pdf
│   ├── 01_introduction.pdf
│   ├── ...
├── tex/                  # 翻译后的 LaTeX 源码
│   ├── main.tex          # 主入口文件
│   ├── sections/         # 各章节源码
│   │   ├── 00_frontmatter.tex
│   │   ├── 01_introduction.tex
│   │   ├── ...
│   └── images/           # 图片资源（如果需要提取或替换）
└── output/               # 编译输出目录
    └── report_cn.pdf     # 最终中文报告
```

## 4. 工具链
- **PDF 处理**: `pdftk` 用于分割 PDF。
- **文本提取**: Google Gemini Vision (多模态能力) 用于读取和理解 PDF 内容。
- **编译工具**: `latexmk -xelatex`。
- **LaTeX 模板**: `ctexart` 文档类，配合 `geometry`, `amsmath` 等宏包。
