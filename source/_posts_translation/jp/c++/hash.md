---
title: "C++ の std::hash 入門"
date: 2024-12-27 01:00:00
tags: [c++, hash, std::hash, std::unordered_map, std::unordered_set]
des: "本記事では std::hash の基本的な使い方を紹介し、カスタム型にハッシュを実装して std::unordered_map / std::unordered_set で使う方法を説明します。"
lang: jp
translation_key: hash
---

データ構造とアルゴリズムでハッシュ（hash）を学んだことがあると思います。最大の利点は（平均的には）テーブル参照が $\mathbf{O}(1)$、つまり定数時間でできることです。C++ でハッシュベースのデータ構造を使いたい場合、一般的には `std::unordered_map` や `std::unordered_set` を使って高速な探索を実現します。

通常、`std::unordered_map` / `std::unordered_set` を使っていて困ることはありません。たとえばキーとして `std::string`、`int`、`float` などの型を使う場合、C++ はすでにハッシュ値の計算方法を用意しています。しかし、キーに自作のクラス／型を使いたい場合、C++ はそのオブジェクトのハッシュ値をどう計算すればよいかを知りません。そのときに「この型はこうやってハッシュを計算する」という定義が必要になり、そこで `std::hash` を使います。

<img src="https://github.com/user-attachments/assets/28b99953-2b57-42da-a029-00d3ad6144d8" rel="img_src" alt="cover picture">

## 基本的な使い方

`std::hash` を使うと、整数・文字列・ポインタなど、さまざまな型のハッシュ値を計算できます。次は簡単な例です：

```cpp
#include <iostream>
#include <string>
#include <functional>

int main() {
    std::hash<int> int_hash;
    std::hash<std::string> string_hash;

    int number = 42;
    std::string text = "hello";

    std::cout << "Hash of number: " << int_hash(number) << std::endl;
    std::cout << "Hash of text: " << string_hash(text) << std::endl;

    return 0;
}
```

```
Hash of number: 42
Hash of text: 2762169579135187400
```

ここでは整数と文字列のハッシュ値を計算しています。`std::hash` の出力は `std::size_t` 型の数値で、ハッシュテーブル内の高速な探索に利用できます。

## カスタム型のハッシュ

`int` や `float`、`std::string` のような標準的な型は、すでにハッシュの実装が用意されています。一方で、新しい型を自分で定義した場合は、その型のハッシュアルゴリズムを自分で実装する必要があります：

```cpp
#include <iostream>
#include <string>
#include <functional>

struct Person {
    std::string name;
    int age;

    bool operator==(const Person& other) const {
        return name == other.name && age == other.age;
    }
};

// Person 型の std::hash
namespace std {
    template <>
    struct hash<Person> {
        std::size_t operator()(const Person& p) const {
            return std::hash<std::string>()(p.name) ^ (std::hash<int>()(p.age) << 1);
        }
    };
}

int main() {
    Person person{"Alice", 30};
    std::hash<Person> person_hash;

    std::cout << "Hash of person: " << person_hash(person) << std::endl;

    return 0;
}
```

この例では `Person` 型のハッシュ計算を定義しています。ハッシュ値の計算方法は用途に応じて決められますが、ここでは `name` と `age` のハッシュを XOR とビットシフトで組み合わせています。一般にハッシュは高速であるほど望ましいため、簡単な演算で済むことも多いです。ただし、衝突（collision）が起きやすい設計になっていないかにも注意が必要です。理想的には、異なるオブジェクトはそれぞれユニークなハッシュ値を持つべきです。

## `std::unordered_map` / `std::unordered_set` のキーのハッシュを定義する

`std::unordered_map` や `std::unordered_set` のキーとして特殊なオブジェクトを使う場合、そのオブジェクトのハッシュ値の計算方法を定義しなければなりません。ここで `std::hash` を使います。

ここでは `Person` を `std::unordered_map` のキーとして使います。このコンテナは `Person` のハッシュを計算する必要があるため、ハッシュを定義しないとコンパイラは `Person` をどうハッシュすればよいか分からず、コンパイルできません。

定義方法は簡単で、`struct hash<T>::operator()` を実装するだけです。`Person` の例と同じです：

```c++
namespace std {
    template <>
    struct hash<Person> {
        std::size_t operator()(const Person& p) const {
            return std::hash<std::string>()(p.name) ^ (std::hash<int>()(p.age) << 1);
        }
    };
}
```

これで、カスタムオブジェクトをキーとして `std::unordered_map` / `std::unordered_set` を利用できるようになります。

結果は次のとおりです：

```cpp
#include <iostream>
#include <string>
#include <functional>
#include <unordered_map>

struct Person {
    std::string name;
    int age;

    bool operator==(const Person& other) const {
        return name == other.name && age == other.age;
    }
};

// std::hash<Person> をコメントアウトするとコンパイルできない
namespace std {
    template <>
    struct hash<Person> {
        std::size_t operator()(const Person& p) const {
            return std::hash<std::string>()(p.name) ^ (std::hash<int>()(p.age) << 1);
        }
    };
}

int main() {
    std::unordered_map<Person, int> map;

    map[Person{"John", 25}] = 1;
    map[Person{"Jane", 22}] = 2;

    return 0;
}
```

## ハッシュ衝突（Hash Collision）

ハッシュの原理を学ぶとき、必ずハッシュ衝突（hash collision）を学びます。簡単に言えば、各オブジェクトのハッシュ値は本来それぞれ異なるのが望ましいですが、偶然同じになってしまうことがあります。そのとき衝突が発生し、コンテナ側で衝突を処理する必要があります。よくある直感的な方法としては、衝突した要素を線形に探索して解決する方式があります。

では先ほどの例に戻って、`Person` のハッシュが衝突したらどうなるでしょうか？

たとえば `Person` のハッシュを常に定数にする場合：

```c++
namespace std {
    template <>
    struct hash<Person> {
        std::size_t operator()(const Person& p) const {
            return 1;
        }
    };
}
```

それでもハッシュテーブルから正常にデータを取得できるでしょうか？

```c++
#include <iostream>
#include <string>
#include <functional>
#include <unordered_map>

struct Person {
    std::string name;
    int age;

    bool operator==(const Person& other) const {
        return name == other.name && age == other.age;
    }
};

namespace std {
    template <>
    struct hash<Person> {
        std::size_t operator()(const Person& p) const {
            return 1;
        }
    };
}

int main() {
    std::unordered_map<Person, int> map;

    Person p1{"John", 25};
    Person p2{"Jane", 22};
    Person p3{"Foo", 55};

    map[p1] = 1;
    map[p2] = 2;
    map[p3] = 3;

    std::cout << map[p1] << std::endl; // 1
    std::cout << map[p2] << std::endl; // 2
    std::cout << map[p3] << std::endl; // 3


    return 0;
}
```

実際に試すと、`std::hash` の定義で衝突が発生していても、C++ は正しく値を見つけられることが分かります。ただし予想どおり、衝突が起きると探索は線形になり、ハッシュの速度面のメリットは失われてしまいます。
