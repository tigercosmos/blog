---
title: "まだ const std::string& を使ってる？ std::string_view を試そう！"
date: 2023-06-07 00:01:00
tags: [c++, string_view, ]
des: "本記事では C++17 の std::string_view の特徴を紹介し、例も示します。"
lang: jp
translation_key: stringview
---

![COVER IMAGE](https://github.com/tigercosmos/blog/assets/18013815/e93fadda-2edc-4ba8-87fd-4df5569939e4)

## 概要

C++ 入門で最も基本的な標準ライブラリの 1 つである `std::string` は、皆さんかなり使い慣れていると思います。文字列を他の関数に渡す方法として一般的なのは `const std::string &` あるいは `const char *` です。つまり参照（reference）やポインタ（pointer）でアドレスを渡すことで、メモリのコピーを避けるという考え方です。

たとえば次のコードは非常に一般的です：

```c++
#include <string>
#include <iostream>

void print(const std::string &input) {
    std::cout << input << std::endl;
}

int main() {
    std::string message("hello world");
    print(message);
}
```

そして C++17 以降では、新しい選択肢として `std::string_view` があります。ソフトウェア工学における View（ビュー）の概念は「開発者が同じデータ配列（Array）を、異なる視点や方法で参照・アクセスできるようにする」ことです。View は軽量な抽象インターフェースを提供し、データのコピーや再配置をせずに配列データを操作できます。この仕組みによって、追加メモリを消費せずにスライス、リシェイプ、並べ替え、リマップなどの操作ができ、不要なオブジェクト生成やメモリ操作を避けられます。Array View でも String View でも（底層は配列です）、本質は「軽量な操作で内部要素にアクセスする」ことです。たとえば Python の NumPy や C++ の Eigen にもこの概念があります。

では、先ほどのコードを少しだけ変更してみます：

```c++
#include <string>
#include <iostream>
#include <string_view> // 忘れずに include

void print(std::string_view input) {
    std::cout << input << std::endl;
}

int main() {
    std::string message("hello world");
    print(std::string_view(message));
}
```

なぜこれが良いのかを説明していきます。

## std::string_view の利点

`std::string_view` には次の利点があります：

- **軽量で低オーバーヘッド**：`string_view` の生成・コピー・受け渡しは、元の文字列データのコピーを一切行いません。一方、通常の `std::string` をコピーするとコストが高くなります。
- **`std::string` の関数と互換**：普段 `string` に対して行う操作の多く（Iterator、`cout` 出力、`substr`、`find` など）を利用できます。
- **より安全**：`string_view` は所有権を持ちません。`string_view` 自体は気軽に破棄できます。
- **より高い柔軟性**：`string_view` は異なる文字列型との互換性が高いです。たとえば `std::wstring` や `winrt::hstring` を渡してもエラーにならない一方、`const std::string &` では型の不一致でコンパイルできない可能性があります。
- **高速な文字列操作**：`std::string::substr` は新しい `string` オブジェクトを生成するため非常に高コストで、あまり使われません。しかし `std::string_view` なら `substr` を使っても高速です。私の実験では 17 倍も差が出ることがあります。さらに、`string_view` は前綴・後綴操作もでき、`const std::string &` では実現できません。
- **よりモダン**：`const std::string &` や `const char *` に頼るスタイルから脱却できます。旧来の方法も性能的には同等ですが、それでも `string_view` には上記の利点があります。

## std::string_view の例

### substr（部分文字列）

前述のとおり、`std::string::substr` と `std::string_view::substr` では性能差が大きくなり得ます。私の実験では、`string` の `substr` は `string_view` より 17 倍遅くなることがありました。理由は単純で、`string` の `substr` は新しい `string` を生成するからです。

> `substr` の使い方は `substr(start, length)` です。先頭と末尾を指定するものではありません。

では、`string` を使う場合に `substr` を使わず部分文字列比較をしたいならどうすればよいでしょうか？自分で `string` の内部メモリを直接操作する方法もありますが、便利とは言えません。

```c++
void print1(const std::string &input) {
    if(input.substr(0,3) == "123") { // この場合の部分文字列は std::string
      // ...
    }
}

void print2(std::string_view input) {
    if(input.substr(0,3) == "123") { // この場合の部分文字列は std::string_view
      // ...
    }
}
```

### 前綴／後綴を削除する操作

前綴（prefix）や後綴（suffix）の操作もよくあります。たとえばファイルパスからディレクトリ部分を取り除いたり、拡張子を削除したりするケースです。

方法 1：

```c++
// g++ test.cpp -std=c++17; ./a.out 
#include <iostream>
#include <string_view>

int main()
{
    std::string_view path("my_folder/log.txt");

    size_t pos1 = path.find('/');
    path.remove_prefix(pos1 + 1);
    size_t pos2 = path.rfind('.');
    path.remove_suffix(path.size() - pos2);

    std::cout << pos1 << ", " << pos2 << ": " << path << std::endl; // 9, 13: log
}
```

方法 2：

```c++
// g++ test.cpp -std=c++17; ./a.out 
#include <iostream>
#include <string_view>

int main()
{
    std::string_view path("my_folder/log.txt");

    size_t pos1 = path.find('/');
    size_t pos2 = path.rfind('.');
    std::string_view new_path = path.substr(pos1 + 1, path.size() - pos2 - 1);

    std::cout << pos1 << ", " << pos2 << ": " << new_path; // 9, 13: log
}
```

ここでは 2 つの例を示しました。上の例は `remove_prefix` と `remove_suffix` を使っていますが、これらは「元の」 `string_view` オブジェクトを変更する点に注意してください。下の例は `substr` で同様の効果を得る方法です。

最初から最後まで文字列コピーは一切行っておらず、全て `string_view` だけで完結しています。しかも操作は非常に「高レベル」です。`string_view` の助けがなければ、オブジェクトを生成したりメモリをコピーしたりせずに、`cout` でこうも綺麗に出力するのは難しいでしょう。

## 結論

総じて `std::string_view` を使う主な目的は、性能向上とリソース節約です。`string_view` は既存文字列へのポインタと長さを保持するだけで、新しいメモリ確保や文字列コピーが不要なので、生成が速く、メモリ消費も少なくて済みます。また `std::string_view` は軽量な読み取り専用 View を提供し、大きな文字列を効率的に参照できます。そのため、文字列参照や関数引数として文字列を渡す場面で理想的な選択肢になります。

一方で、`const std::string &` にも独自の用途があります。特に文字列を変更する必要がある場合や `std::string` 固有の機能を使いたい場合です。そのような状況では `std::string_view` は読み取り専用なので要件を満たせません。また、他ライブラリの API が `const std::string &` を要求する場合、`string_view` を渡すことはできません。

文字列操作をする際には、`std::string` 自体を操作する必要がある、または新しい所有権が必要な場合を除き、できるだけ `std::string_view` を使うべきです。

## 参考資料

- [std::string_view: The Duct Tape of String Types](https://devblogs.microsoft.com/cppblog/stdstring_view-the-duct-tape-of-string-types/)
- [class std::string_view in C++17](https://www.geeksforgeeks.org/class-stdstring_view-in-cpp-17/)
- [std::basic_string_view (from cppreference)](https://en.cppreference.com/w/cpp/string/basic_string_view)
