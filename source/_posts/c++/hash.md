---
title: C++ 雜湊值函式庫 std::hash 介紹
date: 2024-12-27 01:00:00
tags: [c++, hash, std::hash, std::unordered_map, std::unordered_set]
des: "本文簡單介紹 std::hash 的使用方式，瞭解如何替客製化型別加入雜湊算法，並使用在 std::unordered_map 或 std::unordered_set 中。"
lang: zh
translation_key: hash
---

大家在資料結構與演算法中應該學過雜湊（hash），其最大功用是查表時效率為 $\mathbf{O}(1)$，換句話說就是常數時間的速度，也就是最快的意思。在 C++ 中如果我們想要使用雜湊的資料結構，通常會使用 `std::unordered_map` 或 `std::unordered_set`，藉由這兩個資料結構達到快速查表的效果。

一般使用 `std::unordered_map` 或 `std::unordered_set` 時我們並不會碰到任何問題，例如我們使用 `std::string`、`int`、`float` 等型別當鍵值時，C++ 已經內建好算雜湊值的方式，但是當我們用客製化的類別、型別當作鍵值時，C++ 並不知道要怎樣去計算其雜湊值，我們就會需要定義如何去算該物件的雜湊值，此時就會使用 `std::hash`。

<img src="https://github.com/user-attachments/assets/28b99953-2b57-42da-a029-00d3ad6144d8" rel="img_src" alt="cover picture">


## 基本用法

我們可以直接使用 `std::hash` 來計算不同型別的雜湊值，例如整數、字串、指標等。以下是一個簡單的例子：

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

在這裡，我們分別計算了一個整數和字串的雜湊值。`std::hash` 的輸出是一個 `std::size_t` 型別的數字，這個數字可用來在雜湊表中進行快速查找。

## 客製化型別的雜湊

預設的 C++ 物件都已經實做好雜湊的算法，像是 `int`、`float`、`std::string` 等，但如果我們有定義新物件，則需要自己去實做該物件的雜湊算法：

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

// Person 型別的 std::hash
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

在這個例子中我們定義了 `Person` 型別的雜湊值的計算。雜湊值的計算方式可以根據根據需求來決定，例如這裡使用了 `XOR` 和位移運算來組合 `name` 和 `age` 的雜湊值。雜湊值一般來說效能越高越好，所以通常簡單的運算就可以，但同時也要注意該算法是否會造成衝突，每一個物件應該都要有獨一無二的雜湊值。

## 定義 `std::unordered_map` 或 `std::unordered_set` 的鍵值雜湊

當我們使用特別的物件當作 `std::unordered_map` 或 `std::unordered_set` 的鍵值時，則必須自己定義該物件的雜湊值算法，此時要用到 `std::hash`。

這邊我們使用 `Person` 來當作 `std::unordered_map` 的鍵值，由於該容器需要計算 `Person` 物件的雜湊，如果我們不定義該物件的雜湊算法的話，編譯器會不知道要如何對該物件做雜湊，自然就無法編譯。

定義方式也很簡單，只要宣告 `struct hash<T>::operator()` 即可，如同 `Person` 的範例：

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

如此一來我們就能夠順利編譯，使用客製化物件來當作 `std::unordered_map` 或 `std::unordered_set` 的鍵值。

成果如下：

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

// 註解掉 std::hash<Person> 則無法編譯
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

## 雜湊碰撞

我們在學雜湊原理時，肯定學過雜湊碰撞（hash collision），簡單來說每個東西的雜湊值應該都不一樣，但萬一剛剛好一樣的時候，就會發生碰撞了，此時我們就需要去處理這個碰撞情況，比較常見直觀的作法是將碰撞的元素以線性的方式再進行檢索。

回到我們剛剛的例子，如果 `Person` 的雜湊出現碰撞會怎樣？

例如我們永遠將 `Person` 回傳常數：

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

我們是否還能正常取得雜湊表的資料呢？

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

實際測試之後會發現就算 `std::hash` 定義的雜湊有碰撞時，C++ 依舊能正確將答案找出來，但可以預期因為發生碰撞，查表就會是線性的，也就失去雜湊原本速度上的優勢。
