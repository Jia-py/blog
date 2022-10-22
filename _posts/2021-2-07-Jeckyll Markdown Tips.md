---

layout:     post
title:      "Jeckyll Markdown Tips"
date:       2021-02-07 21:43:00
author:     "Jpy"
header-img: "img/post-bg-2015.jpg"
# 生活，工作，笔记（个人理解消化心得），文档（方便后续查阅的资料整理），其他
tags:
    - 文档
---




# Jeckyll插入MathJax

1. 在`_includes/head.html`中添加 或 在markdown开头添加

```
<!-- 数学公式 -->
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {
      skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
      inlineMath: [['$','$']]
    }
  });
</script>
```
2. 使用`<img src="https://www.zhihu.com/equation?tex=e^2 />"`
3. 在markdown中调用`$ code $`即可
3. 编辑时注意行内公式使用`$$ 内容 $$ `，行间公式请前后打上空行，typora默认不加空行

# kramdown插入自动目录

```markdown
* 目录
{:toc}
```

