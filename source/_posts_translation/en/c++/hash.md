---
title: "An Introduction to std::hash in C++"
date: 2024-12-27 01:00:00
tags: [c++, hash, std::hash, std::unordered_map, std::unordered_set]
des: "This post briefly introduces how to use std::hash, including how to add hashing for custom types and use them in std::unordered_map or std::unordered_set."
lang: en
translation_key: hash
---

You have probably learned about hashing in data structures and algorithms. Its biggest benefit is that (on average) lookups in a hash table can be $\mathbf{O}(1)$—constant time, i.e., very fast. In C++, when we want to use hash-based data structures, we typically use `std::unordered_map` or `std::unordered_set` to achieve fast lookups.

In most cases, there is no issue using `std::unordered_map` or `std::unordered_set`. For example, when you use built-in types such as `std::string`, `int`, or `float` as keys, C++ already provides built-in ways to compute hash values. However, when you want to use a custom class/type as the key, C++ does not know how to compute its hash value. In that case, you need to define how to hash that object, and `std::hash` is what you use.

<img src="https://github.com/user-attachments/assets/28b99953-2b57-42da-a029-00d3ad6144d8" rel="img_src" alt="cover picture">

## Basic Usage

You can directly use `std::hash` to compute hash values for different types such as integers, strings, and pointers. Here is a simple example:

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

Here, we compute the hash values of an integer and a string. The output of `std::hash` is a `std::size_t` number, which can be used for fast lookups in a hash table.

## Hashing Custom Types

Default C++ types already have hashing implementations, such as `int`, `float`, and `std::string`. But if you define a new type, you need to implement a hash function for it yourself:

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

// std::hash for Person
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

In this example, we define how to compute the hash value for the `Person` type. The way you compute the hash depends on your needs; here we use XOR and bit shifting to combine the hashes of `name` and `age`. In general, you want hashing to be efficient, so simple operations are often enough—but you also need to pay attention to collisions. Ideally, different objects should have unique hash values.

## Defining Key Hashes for `std::unordered_map` or `std::unordered_set`

When you use a special object as the key in `std::unordered_map` or `std::unordered_set`, you must define how to hash that object. This is where `std::hash` comes in.

Here we use `Person` as the key type in `std::unordered_map`. Since the container needs to compute hashes for `Person`, if you do not define a hashing function, the compiler will not know how to hash `Person`, and compilation will fail.

The definition is straightforward: implement `struct hash<T>::operator()`—as shown in the `Person` example:

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

With this, you can compile successfully and use custom objects as keys in `std::unordered_map` or `std::unordered_set`.

The result looks like this:

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

// Comment out std::hash<Person> and it will not compile
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

## Hash Collisions

When learning the basics of hashing, you definitely hear about hash collisions. In short, each object’s hash value should ideally be different, but if two objects happen to produce the same hash value, a collision occurs. In that case, the collision needs to be handled. A common and intuitive approach is to linearly probe the colliding entries.

Back to our example: what happens if `Person` hashes collide?

For instance, what if we always return a constant for `Person`:

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

Can we still retrieve data from the hash table correctly?

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

After testing, you will find that even if your `std::hash` definition produces collisions, C++ can still find the correct answers. However, as expected, because collisions occur, lookups become linear and you lose the performance advantage that hashing is supposed to provide.
