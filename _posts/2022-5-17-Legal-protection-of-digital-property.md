---
layout:     post
title:      "HKU notebook - legal protection of digital property"
subtitle:   ""
date:       2022-05-17 23:00:00
updatedate:
author:     "Jpy"
header-img: "img/post-bg-2015.jpg"
no-catalog: False
onTop: false
latex: true
# 生活，工作，笔记（个人理解消化心得），文档（方便后续查阅的资料整理），项目，其他
tags:
    - 生活
---

# Basic Legal Concepts

Criminal (Offence) vs civil laws

Criminal prosecution is by govt, once started, can only be discontinued by govt.

# Copyright CO

属不属于copyright

1. intangible(无形的)
2. categories
3. protect expressions of an idea
4. 文学，戏剧，音乐作品在以writing方式记录下来之前不存在copyright
5. Literary, dramatic, musical and artistic works must be "original", creativity
6. Copyright arises automatically
7. ...

8. ![image-20221022185123539](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20221022185123539.png)

software copyright: 属于copyright中的literary work s.4(1)

* Source code and object code (protected)
* Algorithms: protect if detailed idea or 作者的很大一部分的技能和精力体现在算法中 (P5 lecture 2). 以自然语言传播算法才算侵权，在程序中**使用**算法不算侵权

判断能否copyrightable的标准：1. originality 原创性 【lecture1 P18 两种conditions. 1. originate from the author 2.has involved the author’s skill and labour 】 2. fixation 有实物媒介存储下来

authorship & ownership

employee work：可能是author但不是owner

commissioned work: 一种自雇佣关系，不是employer与employee的关系

term of protection: copyright保护的**期限**

rights of owner s.22(1) （同时提及的是侵权行为）

Copying

Public相关侵权行为

Authorisation

Secondary Infringement？1. infringing copy？2. required knowledge？

Exceptions to Infringement of Software Copyright

Civil Remedies(赔偿) and Criminal Sanctions

Bit Torrent

Hyperlinking

思路：

1. 属不属于copyright，lecture1 P11
2. copyrightable? lecture 2 P7（原创性 2点；fixation）
3. copyright属于谁,authorship(有多作者吗) and ownership, lecture 2；属于employee work还是commissioned work (s.15)吗，那么copyright属于谁
4. Exceptions to Infringement of Software Copyright?
5. 是否侵犯了owner的copyright？(Infringement)
   1. copying部分s.23（是否属于copy；是否符合substantial part,copy的比例，包含sufficient skill and labour from owner；是否符合causation的两方面，1. similarity 2. chance to access?）
   2. public相关（issue copies to public; rent to public; Making Available of Copies to the Public (not important for CS);  broadcasting (not important for CS); making adaption; Act Done in Relation to an Adaptation)
   3. Authorisation s.22(2) 两个案例. 对他人侵权有positive作用，授权他人侵权行为，并且不是copyright owner，则构成Authorisation; 出售给客户及放在网上给他人下载也属于该点。
   4. 判断存在Secondary Infringement吗？（**是infringing copy吗 lecture3 p25？是否有required knowledge lecture3 p24?** ）
      1. Importing or Exporting infringing copies s.30
      2. ==Possessing or Dealing with Infringing copies s.31==
      3. Providing Means for making infringing copies s.32 注意该工具只能用于制作侵权产品才能定罪
   5. ==offence s.118(1)==
6. Civil Remedies(赔偿) and Criminal Sanctions ==(s.118 offences version of s.30,s.31,s.32) lec4 p8==?

# privacy PDPO

* what is data s.2
* what is personal data s.2
  * 注意是否practical，数据在不同的data user手上是不同的
* what is data subject & data user.

DPP1(collecting data)

DPP2(keeping data)

DPP3(using data)

合法的数据交易

![image-20220420164430569](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20220420164430569.png)

user1与user2之间交易数据，user1需要保证user2的purpose与其一致，只有图中的a,b,c。另外，在该过程中，user1符合DPP1（采集数据）与DPP3（使用数据），user2只适用于DPP1（采集数据）。

* what is direct marketing? S.35A(1)
* 如果是direct marketing（1. 是否是给specific person 2. 是否通过电子的手段），需要考虑S.35下的条约；如果不是direct marketing，直接考虑DPP3
* 考虑是否犯罪

DPP4（data user should keep data secure)

DPP5 （data user告诉data subject，拿了什么数据，用来干什么）

DPP6 (this DPP focus on data subject, 有啥权利) **access & correction right**

S.20 Data user拒绝数据访问的情形

* 访问者是否为data subject

* 是否有多个subject参与，是否只需隐去其他subject的姓名即可提供
* lecture6 p27-28的额外情况

S. 24 Refusal to Comply（遵从） with Data Correction Request:

Privacy Commissioner

**Offence**: 'Doxxing', Doxxing(人肉搜索)

**Extra-territorial Doxxing**

**other offences**

**Exemptions**: 一些例外，一般是豁免遵从DPP3（使用限制）及DPP6（访问和修改权利）

思路：

1. 属于personal data吗 s.2(1)
2. 在这里谁是data subject，谁是data user s.2(1)
3. 属于什么DPP，是否符合Exemptions中一些例外
4. 是否存在direct marketing(s.35G都可以提一提)
5. 有无Doxxing
6. 是否有额外的犯罪行为，需要处以什么惩罚

# Patent PO CH514

What is patent, what is the difference between patent and copyright

2 categories of claims: product and process.

Dependent claims and Independent claims

Ownership s.9E

Employee invention s.57

Patentable Invention: s.9A 4 conditions (susceptible of industrial application; new; inventive; **exemption**)

How can computer program be patentable? P18 lecture 9

Patent Infringement(侵犯) s.73: product patent, process patent

How to judge whether infringe or not? Basic Question on Infringement. 如何去test一些essential elements变种是否属于侵权，**3 tests**

Exceptions s.75

思路：

1. Patentable申请patent吗（4 conditions），是属于被exclude patent吗？
   1. 如果是computer program:  technical contribution(3点)，且不用于trade或心理作用的program，则是patentable

2. 如果属于patentable，属于product还是process（program）

3. ownership是谁
4. 是否有patent infringement，如何judge？如何test（有多个组件的话分开讨论）
5. s.73 侵权

# Exam notice

copyright s.120 PPT中提到推荐阅读的章节也得打印出来

答题不需要重复法条，只需要写出具体的侵权行为或违法行为

考试中每道题对应一个知识点