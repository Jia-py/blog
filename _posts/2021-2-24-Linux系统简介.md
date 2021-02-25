---
layout:     post
title:      "Linux基础"
subtitle:   "Linux Base"
date:       2021-02-24 17:17:00
author:     "Jpy"
header-img: "img/post-bg-2015.jpg"
tags:
    - CS
    - Study
---

linux是一个操作系统，负责系统调度和内核。Linux开源免费，相比windows，没有那么好上手。

# Linux基础命令

* **输入命令类**
  * `tab`补全
  * `↑`历史命令
  * `*`匹配任意多个任意字符 `?`匹配一个任意字符 `[list]`匹配list中的任意单一字符  `[^list]`匹配除list中的任意单一字符外的字符
  * `man <command_name>`使用帮助  `<command_name> --help`使用参数帮助
  * `touch`创建文件 `cat`显示文件 `echo {内容} > {文件名}`写入内容
* **进程管理类**
  * `Ctrl+c`中断当前进程
  * `Ctrl+s`一次暂停，二次恢复
* **显示编辑类**
  * `Shift+PgUp`将终端向上滚动	`Shift+PgDn`向下滚动
* **安装类**
  * `sudo apt-get install <library_name>` 安装包

# Linux用户操作及文件权限设置

**查看用户** `whoami` `who i am` `who mom likes`

**切换用户** `su <user>`

**新建用户及工作目录** `sudo adduser <user_name>`  **只新建用户** `useradd <user-name>`

**退出用户** `exit`

**给新用户sudo权限**：在拥有root权限的用户下键入`sudo usermod -G sudo <user>`

**删除用户**  `sudo deluser <user name> --remove-home(这里是把工作目录一并删除)`

**删除用户组** `sudo groupdel <group>` 但需要先把用户组中的用户删除干净才能使用

**显示目录** `ls (-l)`加上-l为以长目录形式展示

![ls](https://s3.ax1x.com/2021/02/24/yXoNrD.png)

`ls .`显示当前目录 `ls ..`显示上一级目录

修改文件权限 `sudo chown <user> <file> `

```bash
# 二进制表示法
# 这里的三位数字分别表示的是用户、用户组、其他用户的权力，每位数字是一个二进制转为十进制后的值。
# 这个二进制有三位，第一位表示r，第二位w，第三位x。若都有权力，则为111（7），都没有权力，则为000（0）
chmod 600 file-name
# 加减赋权法
# 这里的go代表group、others，还有一个没写的是user。
# + - 很好理解，就是加上什么权力，减去什么权限。如ugo-w，表示三种用户都删去写入权。
chmod go-rw file-name
```

