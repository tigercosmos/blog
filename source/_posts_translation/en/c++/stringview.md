---
title: "Still Using const std::string&? Try std::string_view!"
date: 2023-06-07 00:01:00
tags: [c++, string_view, ]
des: "This post introduces C++17 std::string_view and provides examples."
lang: en
translation_key: stringview
---

![COVER IMAGE](https://github.com/tigercosmos/blog/assets/18013815/e93fadda-2edc-4ba8-87fd-4df5569939e4)

## Introduction

As one of the most basic standard library components in C++, `std::string` should be very familiar to everyone. When it comes to passing a string into another function, common approaches are `const std::string &` or `const char *`. In other words, you pass by address—either a reference or a pointer—so you do not need to copy memory.

For example, the following code is very common:

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

Starting from the C++17 standard, we have a new option: `std::string_view`. In software engineering, the idea of a “view” is that developers can observe and access the same underlying data array from different perspectives or in different ways. A view provides a lightweight abstraction that lets you operate on array data without copying or rearranging it. With this mechanism, you can slice, reshape, reorder, or remap data without consuming additional memory, which helps avoid unnecessary object creation or memory operations. So whether it is an Array View or a String View (which is still an array underneath), the core idea is to access the internal elements with lightweight operations. You can see this concept in Python’s NumPy library and C++’s Eigen library.

Now, let’s slightly modify the previous code:

```c++
#include <string>
#include <iostream>
#include <string_view> // remember to include this

void print(std::string_view input) {
    std::cout << input << std::endl;
}

int main() {
    std::string message("hello world");
    print(std::string_view(message));
}
```

Next, let’s explain why this can be better.

## Advantages of std::string_view

Using `std::string_view` has the following benefits:

- **Lightweight, low overhead**: Creating, copying, and passing around a `string_view` does not copy the underlying string data. In contrast, copying a normal `std::string` can be expensive.
- **Compatible** with `std::string` operations: Most operations you commonly use on `string` are supported, such as iterators, `cout` output, `substr`, `find`, etc.
- **Safer**: A `string_view` never owns the data. When you use `string_view`, you can delete it without worrying about freeing the underlying memory.
- **More flexibility**: `string_view` can be more compatible with different string-like types. For example, you can pass in `std::wstring` or `winrt::hstring` without causing errors, while `const std::string &` might not compile due to type mismatch.
- **Faster string operations**: We rarely call `std::string::substr` because it creates a new `std::string` object, which is very costly. With `std::string_view`, you can use `substr` and it remains fast. According to my experiments, the difference can be as large as 17x. Also, `string_view` supports prefix/suffix operations, which you cannot do when you take `const std::string &`.
- **More modern**: It moves away from always using `const std::string &` or `const char *`. Even though the old methods can have similar performance, `string_view` brings many benefits as described above.

## std::string_view Examples

### substr

As mentioned above, `std::string::substr` and `std::string_view::substr` can have very different performance. In my experiments, calling `substr` on `string` can be 17x slower than `string_view`. The reason is simple: `string::substr` creates a new `string`.

> Note that `substr` is used as `substr(start, length)`, not “start and end”.

So if you do not want to use `substr` with `string` for substring matching, what can you do? You could manually operate on the underlying memory of the string, but that is not very convenient.

```c++
void print1(const std::string &input) {
    if(input.substr(0,3) == "123") { // substring is std::string
      // ...
    }
}

void print2(std::string_view input) {
    if(input.substr(0,3) == "123") { // substring is std::string_view
      // ...
    }
}
```

### Removing Prefix and Suffix

Prefix and suffix operations are also common. For example, you might want to remove the directory prefix from a file path, and remove the file extension suffix.

Method 1:

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

Method 2:

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

Here are two examples. The first uses `remove_prefix` and `remove_suffix`. Note that these functions modify the original `string_view` object. The second example shows how to achieve the same effect with `substr`.

From start to finish, we do not copy any string data. The whole flow uses only `string_view`, and the operations are very “high-level”. Without `string_view`, it would be difficult to avoid creating new objects or copying memory while still printing the desired substring with `cout` in such an elegant way.

## Conclusion

Overall, `std::string_view` is mainly used to improve performance and save resources. Since it only holds a pointer and a length into an existing string, it does not require new allocations or string copies, so it is fast to create and uses less memory. At the same time, `std::string_view` provides a lightweight, read-only view, which enables efficient access to large strings. This makes it an ideal choice for referencing strings or passing string parameters to functions.

However, `const std::string &` also has its own use cases—especially when you need to modify the string or rely on `std::string`-specific functionality. In those cases, `std::string_view` cannot satisfy the requirements because it provides only a read-only view. Also, when you pass a string to APIs from other libraries, if the API requires a `const std::string &`, you cannot pass a `string_view` instead.

When doing string manipulation, unless you need to operate on `std::string` itself or you need to create new ownership, you should use `std::string_view` as much as possible.

## References

- [std::string_view: The Duct Tape of String Types](https://devblogs.microsoft.com/cppblog/stdstring_view-the-duct-tape-of-string-types/)
- [class std::string_view in C++17](https://www.geeksforgeeks.org/class-stdstring_view-in-cpp-17/)
- [std::basic_string_view (from cppreference)](https://en.cppreference.com/w/cpp/string/basic_string_view)
