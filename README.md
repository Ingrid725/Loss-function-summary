{% spoiler "隐藏内容的标题" %}

隐藏文字隐藏文字隐藏文字。  
支持 `markdown` 语法

- 支持 **粗体**、*斜体*
- 支持列表
- 支持md插入图片语法：![1](hexo-pictures/hexo-6_1.png)
- 支持html插入图片语法：<img src = "hexo-pictures/hexo-6_1.png">
- 支持行内代码 `markdown`
- 支持代码块
  
    ```cpp
    #include <initializer_list>
    #include <iostream>
    struct A {
        A() { std::cout << "1"; }
        A(int) { std::cout << "2"; }
        A(std::initializer_list<int>) { std::cout << "3"; }
    };
    int main(int argc, char *argv[]) {
        A a1;
        A a2{};
        A a3{ 1 };
        A a4{ 1, 2 };
    }
    ```// 此处加点文字，防止排版错误：因为这是代码块内的代码。使用时可删除

- 支持表格

    |文字|文字|
    |-|-|
    |文字|文字|

{% endspoiler %}
