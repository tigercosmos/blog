---
title: "C++ std::string Advanced: One Article to Get You Comfortable with Strings!"
date: 2024-11-21 02:01:00
tags: [c++, string, ]
des: "This post introduces advanced string techniques in an approachable way, including advanced std::string usage, std::stringview, std::stringstream, std::regex, std::format, std::wstring, short string optimization, and the <charconv> library."
lang: en
translation_key: std-string-advanced
---

As one of the most important features of a programming language, strings deserve serious attention. In this post, we will learn advanced string techniques in C++, including advanced APIs of `std::string`; how to use `std::stringview` to work with strings more efficiently; how to use `std::stringstream` for data streams; how to use `std::regex` for regular expressions; how to use `std::format` for string formatting; how to use `std::wstring` for wide characters; what short string optimization is; and how to use the `<charconv>` library.

Previous: [C++ std::string for Beginners: One Article to Get You Comfortable with Strings!](/post/2023/06/c++/std-string-beginner/)
Next: [C++ std::string Advanced: One Article to Get You Comfortable with Strings!](/post/2024/11/c++/std-string-advanced/)

![封面照片](https://github.com/user-attachments/assets/0e911498-715e-4e9c-84fb-2c1db0e92573)
(Taken in Jiuzhaigou, 2024)

## 1. std::string as a container

In C++, `std::string` is not just “a string”. It is a powerful container with flexible resizing behavior. If you understand how `std::string` manages capacity, you can handle string data more efficiently. This section introduces important container-related operations such as `capacity`, `size`, `resize`, `reserve`, as well as `push_back` and `pop_back`.

### 1.1 capacity vs size

When working with `std::string`, you will frequently see `size()` and `capacity()`. `size()` is the actual length of the string (the number of characters you stored). `capacity()` is the size of the allocated buffer (how many characters it can hold). The size is always less than or equal to the capacity.

For example, the following demonstrates the difference between `size()` and `capacity()`:

```cpp
#include <iostream>
#include <string>
int main() {
    std::string str = "Hello";
    std::cout << "Size: " << str.size() << std::endl; // 5
    std::cout << "Capacity: " << str.capacity() << std::endl; // 15
    return 0;
}
```

`std::string` usually allocates more memory than it currently needs to reduce the cost of frequent reallocations. In this example, `capacity` is 15 while `size` is 5. If you append more characters, it will not reallocate immediately; it will reallocate only when the capacity is exhausted. Reallocation is expensive, because the typical mechanism is to allocate a new buffer (often with doubled capacity) and then copy the old data into the new buffer.

### 1.2 resize

The `resize` function lets you manually adjust the string length. If the new length is longer than the old length, `std::string` fills the additional space (by default with `\0`). If the new length is shorter, it truncates the string.

```cpp
std::string str = "Hello";
str.resize(10); // str becomes "Hello\0\0\0\0\0"
std::cout << "New Size: " << str.size() << std::endl; // 10
str.resize(3); // str becomes "Hel"
std::cout << "New Size: " << str.size() << std::endl; // 3
```

Notice that `resize` “fills” the string to the requested length. So if you `str.resize(10)`, then `size` and `capacity` will both be 10.

### 1.3 reserve

When `std::string` runs out of capacity, reallocating memory is very expensive. So if you already know how much memory you will need, you can allocate it once ahead of time using `reserve`. `reserve` only affects `capacity`; it does not change `size`.

```cpp
std::string str;
str.reserve(100); // pre-allocate space for 100 characters
std::cout << "Capacity after reserve: " << str.capacity() << std::endl; // at least 100
std::cout << "Size after reserve: " << str.size() << std::endl; // 0
```

After reserving space, you can use `str += str2` or `str.push_back(c)` to append new content to `str`. Since the buffer is already reserved, there will be no allocations until you fill up the reserved capacity. Therefore, when you use `reserve`, you should be careful not to exceed the planned capacity too soon.

Using `reserve` can improve performance for large string workloads by reducing the number of reallocations. If you know you are going to do many string operations, reserving memory first can avoid unnecessary reallocations later.

> Tip: When you use `size()`, the string is already “filled”, so you usually use `str[i]` to modify characters. When you use `reserve()`, you are only reserving space; in that case, you should append via `str += str2` or `str.push_back(c)`.

### 1.4 push_back and pop_back

`push_back` appends a single character to the end of `std::string`. If there is not enough capacity, it will trigger reallocation.

```cpp
std::string str = "Hello";
str.push_back('!'); // same as str += '!'
std::cout << str << std::endl; // Hello!
```

`pop_back` removes a single character from the end of `std::string`.

```cpp
std::string str = "Hello";
str.pop_back();
std::cout << str << std::endl; // Hell
```

Single-character operations are sometimes very convenient. For example, when you iterate through characters in a `for` loop, and you combine it with `reserve`, you can often ensure you stay within capacity.

> You will notice that these functions are used exactly like `std::vector`. That is because you can think of `std::string` as `std::vector<char>`—their nature is very similar.

## 2. std::string vs C-style string

In C++, `std::string` is a powerful container, but traditional C-style strings (`char*`) are still unavoidable in some scenarios—for example, when interacting with C libraries that only accept C-style strings.

### 2.1 Convert std::string to C-style string

Converting `std::string` to a C-style string is easy: just call `.c_str()` to get a `const char*`.

```cpp
std::string filename = "data.txt";
const char* c_filename = str.c_str();
FILE* file = fopen(c_filename, "w"); // POSIX API only accepts C-style strings
```

If you need a non-`const` `char*`, you can allocate a new memory buffer using `std::vector<char>` or `std::array<char>`, and copy the string into it (e.g., using `memcpy`).

### 2.2 Convert C-style string to std::string

Converting from a C-style string to `std::string` is also easy—just use a `std::string` constructor:

```cpp
const char* c_str = "Hello";
std::string str(c_str);
std::cout << str << std::endl; // prints "Hello"
```

> In modern C++ development, we should prefer `std::string` for string handling.

## 3. string_view

`std::string_view` is a type introduced in C++17. It provides a lightweight, non-owning string view. It is pointer-like: it can refer to an existing string without copying the data, reducing unnecessary memory overhead. `std::string_view` is especially suitable for partial string operations such as trimming prefixes/suffixes and taking substrings. Traditionally, doing those operations with `std::string` often incurs extra allocations and copies.

> Further reading: [Still Using `const std::string &`? Try `std::string_view`!](/post/2023/06/c++/stringview/)

### 3.1 Basic usage of string_view

`std::string_view` can be initialized from `std::string` or a C-style string, and you can use many operations similar to `std::string`. The difference is: using `std::string_view` can avoid unnecessary string copying.

```cpp
#include <string_view>

void print_string(std::string_view sv) {
    std::cout << sv.substr(0,3) << std::endl; // print substring
}

int main() {
    std::string str = "Hello, world!";
    print_string(str); // pass std::string
    print_string("Temporary C-string"); // pass C-style string
    return 0;
}
```

In the example above, we use `substr()` to obtain a substring. If this were `std::string::substr`, it would involve a memory copy. But `std::string_view` is a view: it refers to the original string’s memory. Therefore, the substring returned by `std::string_view::substr` is also a `std::string_view`, and there is no memory copy during the process!

> You must carefully manage the lifetime of `std::string_view`. It does not own the underlying data, so the underlying data must outlive the `std::string_view`. Avoid pointing `std::string_view` to temporary data (e.g., strings from local variables that go out of scope).

## 4. stringstream

`std::stringstream` is a string-based data stream. The concept is similar to `std::cin` and `std::cout`, except that `stringstream` reads/writes to a “string stream” while `cin/cout` read/write to I/O. `std::stringstream` is very useful for formatting, input/output, and data conversion.

### 4.1 Basic usage

We can use `std::stringstream` to build strings, for example:

```cpp
#include <sstream>
#include <iostream>
#include <string>
#include <vector>

int main() {
    std::vector<int> v = {1, 2, 3};

    std::stringstream ss;
    ss << "[";

    for (int i = 0; i < v.size(); i++) {
        ss << v[i];
        if (i != v.size() - 1) {
            ss << ", ";
        }
    }

    ss << "]";

    std::cout << ss.str();  // [1, 2, 3]
}
```

In this example, we output an array as a string. You can see that `std::stringstream` lets us use `<<` just like `std::cout` to insert data into the stream. After building the content, we call `ss.str()` to get a `std::string`.

In addition, `std::stringstream` can also output and read data like `std::cout` and `std::cin`. For example, it can convert numbers to strings, or parse numbers from a string:

```cpp
#include <sstream>
#include <iostream>
#include <string>
int main() {
    std::stringstream ss;
    ss << 123 << " " << 234; // write into stringstream
    std::string result = ss.str(); // convert to std::string
    std::cout << result << std::endl; // prints "123 234"
    
    // ss still holds "123 234"
    int number;
    ss >> number; // read from stringstream, left to right
    std::cout << number << std::endl; // prints "123"
    return 0;
}
```

### 4.2 Applications of stringstream

For example, when handling CSV or other structured data, we can use `std::stringstream` as the data source.

```cpp
#include <sstream>
#include <string>
#include <vector>
#include <iostream>

int main() {
    std::string line = "apple,banana,orange";
    std::stringstream ss(line); // a data stream, could come from network or a file
    
    std::string item;
    std::vector<std::string> items;

    while (std::getline(ss, item, ',')) {
        items.push_back(item);
    }

    return 0;
}
```

The definition of `std::getline` is `istream& getline(istream& input_stream, string& output, char delim);`. The first parameter is an `std::istream`, and `std::stringstream` inherits from `std::istream`, so we can use `getline` directly on it. Similarly, if you have used `std::getline(std::cin, ...)`, `std::cin` also inherits from `std::istream`.

## 5. regex

Regex (regular expression) is commonly used for pattern matching and searching in strings, including validating formats, searching for patterns, and doing string replacements. The C++ standard library provides `<regex>` for regex operations.

> If you have never used regex, you can first read [MDN’s basic syntax guide](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Regular_expressions).

### 5.1 Basic usage

The most common regex usage is to use a `std::regex` object together with `std::regex_match` or `std::regex_search`:

```cpp
#include <iostream>
#include <regex>
#include <string>

int main() {
    std::string input = "hello123";
    std::regex pattern("[a-z]+\\d+"); // define regex pattern
    if (std::regex_match(input, pattern)) {
        std::cout << "Input matches the pattern!" << std::endl;
    }
    return 0;
}
```

In this example, the pattern `[a-z]+\\d+` means “one or more lowercase letters followed by one or more digits”. With `std::regex_match`, we can check whether the entire string matches the pattern.

### 5.2 Searching and replacing

Besides `std::regex_match`, we can use `std::regex_search` to find matching parts inside a string, or use `std::regex_replace` to replace:

```cpp
#include <iostream>
#include <regex>
#include <string>

int main() {
    std::string input = "abc123def456";
    std::regex pattern("\\d+");

    // search
    std::smatch match;
    if (std::regex_search(input, match, pattern)) {
        std::cout << "Found number: " << match.str() << std::endl;
    }

    // replace
    std::string replaced = std::regex_replace(input, pattern, "#");
    std::cout << "Replaced string: " << replaced << std::endl;

    return 0;
}
```

Here we use `std::regex_search` to find the first substring matching `\\d+` (one or more digits), and then use `std::regex_replace` to replace all digits with `#`.

> Although `regex` is powerful, it is very slow. This is intuitive: pattern matching repeatedly validates the string, so the complexity cannot be low. If you can solve the problem with a better method, try to avoid `regex`.

## 6. format

Many languages have built-in string formatting tools, such as Python, Rust, and Golang. Finally, C++20 introduced `std::format`, a modern and safer formatting tool compared to traditional `std::sprintf` or `std::ostringstream`. `std::format` lets you format strings in a concise and readable way.

Note that `std::format` requires C++20. In practice, you need at least `g++-13` (Ubuntu 24 default) to support it. If you can only use C++17, you can consider [libfmt](https://github.com/fmtlib/fmt), which provides basically the same functionality, but is not part of the standard library, so project setup is a bit more work.

### 6.1 Basic usage

Let’s start with the most basic usage:

```cpp
#include <iostream>
#include <format>

int main() {
    int number = 42;
    std::string name = "Alice";

    // format using std::format
    std::string result = std::format("Hello, {}! Your number is {}.", name, number);
    std::cout << result << std::endl;

    return 0;
}
```

In this example, `std::format` substitutes parameters into `{}`, similar to Python’s `str.format` or Rust’s `format!`. `std::format` automatically converts parameters into strings and inserts them.

### 6.2 Format specifiers

`std::format` supports various format specifiers, such as numeric base, width, padding, etc.

```cpp
#include <iostream>
#include <format>

int main() {
    int number = 255;

    std::cout << std::format("Hexadecimal: {:#x}\n", number); // hex
    std::cout << std::format("Padded number: {:08}\n", number); // pad with 0 to width 8
    std::cout << std::format("Scientific notation: {:.2e}\n", 12345.6789); // scientific notation

    return 0;
}
```

Output:

```
Hexadecimal: 0xff
Padded number: 00000255
Scientific notation: 1.23e+04
```

In practice, it feels similar to `boost::format` or even `std::cout` in that there are many ways to configure formatting.

## 7. wstring

For a normal `std::string`, each `char` occupies one byte, so it can represent at most 256 values. This is far from enough to represent Unicode (tens of thousands of characters). Depending on the language, characters may require different byte lengths. For example, many CJK characters are multibyte and may require 2 (UTF-16) to 4 (UTF-32) bytes. In such cases, we use `std::wstring`, which is the wide-character version of `std::string`. It stores each character as `wchar_t`, which is suitable for Unicode or other multibyte encodings.

### 7.1 Basic usage

Using `std::wstring` is very similar to `std::string`, but you need to initialize with `L""` literals:

```cpp
#include <iostream>
#include <string>

int main() {
    std::ios::sync_with_stdio(false);
    std::wcout.imbue(std::locale("en_US.utf8"));

    std::wstring ws = L"你好，世界"; // initialize using L"" literal
    std::wcout << ws << std::endl;
    std::wcout << "Length: " << ws.size() << std::endl; // 5

    return 0;
}
```

Here we use `std::wcout` to output wide strings. When dealing with wide characters, you should be careful about whether your output stream and locale settings properly support them.

> Note that we don’t actually know which encoding the compiler/runtime uses internally—it could be UTF-16, UTF-32, or something else.

> We use `sync_with_stdio(false)` here because C++ is compatible with C by default. If you don’t disable that compatibility, wide characters may still be interpreted as narrow characters, causing incorrect output. See [here](https://stackoverflow.com/a/31577195/6798649) for details.

Although we can store wide characters in a normal `std::string`, the wide characters will be stored as a sequence of `char`. If you want to process wide characters one by one, using `std::string` becomes inconvenient.

For example, try the following:

```c++
#include <iostream>
#include <string>

int main() {
    std::ios::sync_with_stdio(false);
    std::wcout.imbue(std::locale("en_US.utf8"));

    std::wstring ws = L"你好，世界"; // size = 5
    std::string ns = "你好，世界"; // size = 15

    for(auto c : ws) {
        std::wcout << c << std::endl;
    }

    for(auto c : ns) {
        std::cout << c << std::endl;
    }

    return 0;
}
```

You will see that with wide characters, each character prints correctly. In contrast, with a narrow `std::string`, the output becomes gibberish.

### 7.2 Encoding conversion with codecvt

When converting between `std::string` and `std::wstring`, the `<codecvt>` library provides encoding conversion tools, including conversions between narrow and wide strings.

Although `<codecvt>` was deprecated in C++17, it is still used in many existing projects. We can use `std::wstring_convert` and `std::codecvt_utf8` to convert between UTF-8 and wide strings:

```cpp
#include <iostream>
#include <string>
#include <locale>
#include <codecvt>

int main() {
    std::wstring wide_string = L"こんにちは";
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;

    // wide string to UTF-8
    std::string utf8_string = converter.to_bytes(wide_string);
    std::cout << "UTF-8: " << utf8_string << std::endl;

    // UTF-8 to wide string
    std::wstring converted_back = converter.from_bytes(utf8_string);
    std::wcout << L"Wide: " << converted_back << std::endl;

    return 0;
}
```

Although `<codecvt>` still works, since it is deprecated, starting from C++20 it is recommended to use other Unicode libraries (e.g., Boost.Locale) for string encoding conversion.

> Before a new standard solution is defined, `<codecvt>` will not be removed. Therefore, it still exists in C++20.

## 8. small string

Short String Optimization (SSO) is an optimization technique for short strings. Typically, `std::string` reserves a small fixed buffer internally to avoid dynamic allocations for short strings. If the string length is below some threshold (often 15 or 23 characters, depending on compiler and implementation), it stores the string data directly on the stack instead of allocating heap memory.

Example:

```cpp
#include <iostream>
#include <string>

int main() {
    std::string small_string = "short"; // may use SSO
    std::string large_string = "this is a very long string that might not fit in SSO";

    // Small string: size: 5, capacity: 15
    std::cout << "Small string: size: " << small_string.size() << ", capacity: " << small_string.capacity() << std::endl;

    // Large string: size: 52, capacity: 52
    std::cout << "Large string: size: " << large_string.size() << ", capacity: " << large_string.capacity() << std::endl;

    return 0;
}
```

In this example, `small_string` likely uses SSO because it is short. You can see that even though it has only 5 characters, the capacity is 15. In contrast, `large_string` triggers heap allocation.

SSO can significantly reduce allocation costs for short-string operations and thus improve performance. In most cases we do not need to worry about SSO, but in performance-sensitive scenarios, it is still worth understanding its impact.

> Further reading: [Short String Optimization (SSO) in C++](/post/2022/06/c++/sso/)

## 9. to_chars & from_chars

C++17 introduced a new numeric conversion library `<charconv>`. In particular, `std::to_chars` and `std::from_chars` provide efficient conversions between strings and numbers. `<charconv>` is header-only, so it is lightweight, and it uses modern algorithms with very high performance.

> Further learning: Talk [Stephan T. Lavavej “Floating-Point ＜charconv＞: Making Your Code 10x Faster With C++17's Final Boss”](https://www.youtube.com/watch?v=4P_kbF0EbZM). Around minute 45, you can see benchmark results showing `<charconv>` can be several times faster than older approaches.

### 9.1 Basic usage

#### 9.1.1 std::to_chars

We can convert a number into a string. Here is a `std::to_chars` example:

```cpp
#include <iostream>
#include <charconv>
#include <array>

int main() {
    int number = 12345;
    std::array<char, 20> buffer; // pre-allocated buffer

    // convert using std::to_chars
    auto [ptr, ec] = std::to_chars(buffer.data(), buffer.data() + buffer.size(), number);

    if (ec == std::errc()) { // check success
        std::cout << "Converted number: " << std::string(buffer.data(), ptr) << std::endl;
    } else {
        std::cout << "Conversion failed." << std::endl;
    }

    return 0;
}
```

In this example, we define a `buffer` and use `std::to_chars` to convert an integer to a string. `std::to_chars` returns a `std::to_chars_result` that contains the result pointer (`ptr`) and an error code (`ec`).

`ptr` points to the end of the successfully written output. Therefore, if conversion succeeds, `ptr` will point to the end of the converted string, and we can create the output string with `std::string(buffer.data(), ptr)`.

#### 9.1.2 std::from_chars

We can also convert a string into a number. Here is a `std::from_chars` example:

```cpp
#include <iostream>
#include <charconv>
#include <array>

int main() {
    std::string intStr = "12345 abc";
    int resultInt = 0;
    
    // Perform the conversion from string to int
    auto [ptr, ec] = std::from_chars(intStr.data(), intStr.data() + intStr.size(), resultInt);
    
    if (ec == std::errc()) {
        std::cout << "Integer conversion successful: " << resultInt << ", ptr:" << ptr << std::endl;
    } else {
        std::cout << "Integer conversion failed. ptr:" << ptr << std::endl;
    }
    return 0;
}
```

If conversion succeeds, `resultInt` contains the correct result. Otherwise, you can use `ptr` to check the last processed position, which means from that position onward, the original `intStr` is not a valid number.

For example, if you pass `12345 abc`, you will get a success message and `ptr` will point to ` abc`, because the latter part cannot be parsed. In contrast, if you pass `abc123`, you will get a conversion error and `ptr` points to the beginning of `abc123`.

### 9.2 Advanced usage

`std::to_chars` also supports different bases (e.g., hexadecimal) and floating-point conversions:

```cpp
#include <iostream>
#include <charconv>
#include <array>

int main() {
    double value = 3.14159;
    std::array<char, 20> buffer;

    // floating-point conversion
    auto [ptr, ec] = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);

    if (ec == std::errc()) {
        std::cout << "Converted float: " << std::string(buffer.data(), ptr) << std::endl;
    } else {
        std::cout << "Conversion failed." << std::endl;
    }

    int value = 8;
    std::array<char, 20> buffer;

    // different base
    [ptr, ec] = std::to_chars(buffer.data(), buffer.data() + buffer.size(), 8 /* 八進位 */);

    if (ec == std::errc()) {
        std::cout << "Converted int: " << std::string(buffer.data(), ptr) << std::endl;
    } else {
        std::cout << "Conversion failed." << std::endl;
    }

    return 0;
}
```

The results will be the string `3.14159` and the string `10` (decimal 8 equals octal 10).

> The `<charconv>` library is more suitable for simple conversions with high performance, but it cannot do complex formatting. If you need formatting, consider using `std::format`.
