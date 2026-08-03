# OccPlanner project page

这是可以直接上传到 `hbl-0624.github.io` 仓库根目录的静态网站。

## 上传文件

将下面所有内容上传到仓库根目录：

- `index.html`
- `style.css`
- `occplanner-paper.pdf`
- `assets/` 文件夹

## 修改内容

- 页面文字、指标、按钮链接：编辑 `index.html`
- 配色、字号和布局：编辑 `style.css`
- 图片和视频：用新文件覆盖 `assets/` 中的同名文件
- 论文：用新 PDF 覆盖 `occplanner-paper.pdf`

代码公开后，在 `index.html` 中搜索 `Code · coming soon`，将该 `<span>` 替换为：

```html
<a class="button button-light" href="你的代码仓库链接" target="_blank" rel="noreferrer">Code ↗</a>
```

论文作者和引用信息确定后，搜索 `Binling Huang et al.` 与 `Citation details` 进行替换。
