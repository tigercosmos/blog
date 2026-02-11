---
title: "C++ std::string for Beginners: One Article to Get You Comfortable with Strings!"
date: 2023-06-21 00:01:00
tags: [c++, string, ]
des: "This post introduces std::string in a clear and approachable way, covering a complete set of beginner-level string usage."
lang: en
translation_key: std-string-beginner
---

## 1. Introduction to string

Whether C++ is your first programming language, or you are a developer coming from C or Python, you have probably realized how important “strings” are in a programming language. In C++, strings are handled with the `std::string` library (in this post, I will refer to it as `string`). First time hearing this? Don’t worry—this article will explain everything you should know about `string`, including details and pitfalls.

The goal of this article is to help beginners who have some programming experience become truly comfortable with using `string`. I recommend that beginners at least skim through the first half of any C++ tutorial book (at least know what variables are and what loops are). Many technical terms will appear along the way. I will try to explain them in simple terms, but due to space limitations I cannot explain every single term. The best solution is: if you see a term you don’t understand, just ask ChatGPT. For more advanced readers: you may already know the basic parts, but you might still have blind spots that this post can fill in.

Why do we need `string`?

Think about how you would represent a string without `string`:

```cpp
const char *a = "abcdfe";
char b[] = "123345";
```

It’s not impossible—this is basically how we do it in C. But it’s not convenient. For example, if you want to know the length, search for a substring, or concatenate strings, it’s all annoying!

With `string`, life becomes much simpler:

```cpp
string s = "aabbcc";
s.size(); // return 6
s.find("bb"); // return 2
s += "123"; // s = aabbcc123
```

Before we start, remember: whenever you use `std::string`, you must include the `<string>` header!

Also, `string` is part of the standard library, so normally you should write `std::string`. To keep things concise, this post assumes you have already declared `using std::string;`, so you don’t have to type `std::` everywhere. If you are not familiar with this, look up the `using` keyword and the `namespace` keyword.

So the example code in this post assumes:

```c++
#include <string>
using std::string;
```

This article is the **beginner** part. It will take you through the basic operations and usage of `string`. Let’s learn `string`!

Previous: [C++ std::string for Beginners: One Article to Get You Comfortable with Strings!](/post/2023/06/c++/std-string-beginner/)
Next: [C++ std::string Advanced: One Article to Get You Comfortable with Strings!](/post/2024/11/c++/std-string-advanced/)

![cover image](https://github.com/tigercosmos/blog/assets/18013815/3fdf2165-cb58-4a01-8698-bca0fec93acb)

## 2. Declaring and initializing string

Let’s see how to declare a `string`:

```c++
string s1; // default initialization, an empty string
string s2 = s1; // s2 is a copy of s1
string s3 = "hello world"; // s3 copies the string value (implicitly converts to string first)
string s4("hello world"); // s4 initializes from the given string
```

Next, we should talk about direct initialization, copy initialization, move initialization, copy assignment, and move assignment (often discussed under the umbrella of the [Rule of Three](https://en.cppreference.com/w/cpp/language/rule_of_three)).

> The following explanation can be a bit complex. If you are a beginner, just remember that any of the approaches s5~s9 below are fine for creating a string.

```c++
string s5("abc"); // direct initialization
string s6 = "abc"; // copy assignment
string s7 = string("abc"); // copy assignment
string s8(s7); // copy initialization
string s9 = s8; // copy initialization
string s10(std::move(s9)); // move initialization
string s11 = std::move(s10); // move assignment
```

Let’s start with s5: it’s straightforward—you directly tell `string` what value to initialize with.

s6 is a very common declaration style. In fact, s6 and s7 are completely equivalent. In s6, the rvalue performs an implicit conversion, which makes it the same as s7.

s6 or s7 performs a copy conversion, meaning you create a new `string` object (the lvalue) from another `string` object (the rvalue). That suggests you might “construct” a string twice. In theory, this syntax is inefficient, but fortunately most modern compilers optimize it away, and in practice it may make no difference. Still, you should understand the differences when compiler optimizations are not available.

s8 and s9 both perform copying. Usually when you write this, you know you need an actual string copy, so it’s fine.

s10 in plain terms means: transfer the data inside s9’s string directly to s10. After that, s9 should no longer be used, and s10 can be initialized more efficiently because it directly takes ownership of the data. s11 is the same idea. This involves concepts like `std::move` and rvalues, which I’m only mentioning briefly here. Interested readers can explore further using these keywords.

For complete initialization options, you can refer to the [constructor list](https://en.cppreference.com/w/cpp/string/basic_string/basic_string). Most constructors are more advanced. For now, it’s enough to remember the common ways above. If you know how to use `std::vector`, many constructor patterns also apply to `string`.

Finally, here is a special initialization method:

```cpp
using namespace std::literals;
string s3_2 = "hello world"s; // another way to write s3; ""s makes the rvalue a string directly
```

With the `""s` operator, you can directly turn character string literals into `string`. But remember: you must add `using namespace std::literals;`.

## 3. Basic operations on string

Here are common operations you can perform on `string`:

```c++
os << s // output s 
is >> s // input s
s.empty() // check whether s is empty
s.size() // current length of s
s[n] // access the n-th element of s directly
s1 + s2 // concatenate s1 and s2 and get a new string
s1.append(s2) // append s2 to the end of s1
s1 = s2 // copy s2
s1 != s2 // compare whether s1 and s2 are different
<, <=, ==, >=, > // lexicographical comparison (dictionary order)
```

### 3.1 Input and output

We can write a simple program to demonstrate input and output:

```c++
#include <iostream>
#include <string>
int main() {
    std::string input;
    while(std::cin >> input) { // keep reading until EOF (end of file)
        std::cout << input << std::endl; // print the input we just read
    }
    return 0;
}
```

> If you don’t know `std::cin` and `std::cout`, look up how `iostream` works. `std::endl` represents a newline.

You can also read an entire line:

```c++
#include <iostream>
#include <string>
int main() {
    std::string line;
    while(std::getline(std::cin, line)) { // read line by line (split by \n) until EOF
        std::cout << line << std::endl; // print the line we just got
    }
    return 0;
}
```

Of course, it doesn’t have to be `ostream` (`std::cin`, `std::cout`). For example, later we will introduce `stringstream`, which also works with `>>` and `<<`. As long as it is a stream, it generally works. You can even overload `>>` and `<<` for custom C++ objects—but that’s beyond the scope here.

### 3.2 `empty()` and `size()`

`empty()` and `size()` are commonly used when working with strings.

```c++
string s = "";
if(s.empty()) {
    std::cout << "it's empty!"; 
}

s = "12345678910";
if(s.size() > 5) {
    std::cout << "more than 5!"; 
}
```

For example, in the `getline` example above, if you want to skip empty lines, you can write:

```c++
    while(std::getline(std::cin, line)) {
        if(!line.empty()) { // make sure it's not an empty string
            std::cout << line << std::endl;
        }
    }
```

Note that `!s.empty()` is equivalent to `s.size() > 0`, but `!s.empty()` is often considered more concise and expressive.

Also note: `size()` returns `string::size_type`. The exact underlying type depends on the standard library implementation, but in practice it is usually `size_t` (a non-negative integer). So `s.size()` is not an `int`!

That means the following loop is wrong:

```c++
// wrong!
for(int i = 0; i < s.size(); ++1) {
    std::cout << s[i];
}
```

The correct version would be:

```c++
// correct
for(std::string::size_type i = 0; i < s.size(); ++1) {
    std::cout << s[i];
}
```

> Don’t worry—we will explain `for` very soon!

You can also use `size_t`, which is generally correct, or if you are lazy, just use `auto`.

In many cases you can still use `int`, because `int` and `size_t` (`s.size()`) can be compared. But in some cases it can cause bugs. For example, if you compare `s.size() < n` and `n` is a negative `int`, then `n` will be converted to `size_t` (a huge positive number), and the comparison will always be `true`.

### 3.3 Accessing characters

How do we access characters in a string?

The simplest two ways are `s[]` and `s.at()`.

```c++
string s("0123456789");
 
s[2] = 'a'; // s = "01a3456789"
std::cout << s[9]; // 9

s.at(3) = '6'; "01a6456789"
std::cout << s.at(3); // 6
```

`[]` and `at()` look the same, right?

The difference is boundary checking:

```c++
std::cout << s[100]; // intentionally out of bounds
// undefined behavior: could be garbage, could be a segmentation fault

std::cout << s.at(100); // intentionally out of bounds
// terminate called after throwing an instance of 'std::out_of_range'
//   what():  basic_string::at: __n (which is 100) >= this->size() (which is 10)
// Aborted
```

You can see that `at()` performs bounds checking and clearly tells you the code is wrong. You can also combine it with try-catch for error handling. In contrast, accessing out of range with `[]` is undefined behavior, and most of the time you get a segmentation fault. Does that mean `[]` is bad? Not really—bounds checking has a performance cost. `[]` is more direct and often faster, but then it’s the developer’s responsibility to ensure indices are valid.

You can check bounds like this:

```c++
string s("abcd");
size_t index = /* any number */;
if(index >=0 && index < s.size()) {
    std::cout << s[index];
}
```

Two other commonly used functions are `s.front()` and `s.back()`. As the names suggest, they access the first and the last character.

```c++
string s("abc");
std::cout << s.front(); // a
std::cout << s.back(); //c
```

You could also use `s[0]` and `s[s.size() - 1]`, but that is less intuitive and uglier.

### 3.4 Concatenating strings

#### 3.4.1 Concatenating two strings

Concatenate `"abc"` and `"defg"` into `"abcdefg"`.

One simple approach is the `+` operator, such as `s1 + s2`. Another approach is `s1.append(s2)`.

Let’s look at an example:

```c++
string s1("aaa");
string s2("bbb");

string s3 = s1 + s2; // s3 = "aaabbb"

s1 = s1 + s2; // case 1 inefficient
s1 += s2; // case 2 efficient
s1.append(s2); // case 3 efficient
```

Consider case 1: `s1 + s2` produces a new string and then copies it back into `s1`. Obviously inefficient. In case 2 and case 3, the idea is to append `s2` to the end of `s1`, so the actual performance of case 2 and case 3 is the same. The difference between case 1 and case 2 (or case 3) is that case 1 copies `s1` first, then copies `s2`, and produces a new string. The latter only needs to copy `s2` and place it after `s1`.

You can see that case 1 copies `s1` and also creates an extra string object. If you do not want to modify `s1` and `s2`, then writing `string new_str = s1 + s2` is fine.

If you don’t care whether `s1` is modified, use `append` or `+=` to reduce unnecessary copies.

In the example above, `s1 = s1 + s2` is very inefficient—please don’t write it like that!

#### 3.4.2 Concatenating string and literals

`string` can also be concatenated with character literals and character string literals. The idea is simple: the literals are converted automatically.

However, when mixing `string` and literals with `+`, at least one side of `+` must be a `string`.

```c++
string s1 = "hello";
string s2 = "world";
string s3 = s1 + ' ' + s2 + "!\n"; // OK
string s4 = "123" + "567"; // error: cannot add two literals directly
string s5 = "123"s + "567"s; // OK: same as string + string
string s6 = s1 + "aaa" + "bbb"; // OK: s1 + "aaa" creates a new string, then adds "bbb"
string s7 = "aaa" + "bbb" + s1; // error: "aaa" + "bbb" happens first; cannot add two literals
string s8 = s1 + "aaa"; // OK
string s9 = "aaa" + s1; // OK
```

### 3.5 Comparing two strings

Next, let’s introduce `<, <=, ==, !=, >=, >`.

`==` is easy: if the two strings have the same length and the same content, then `s1 == s2` is true.

For ordering comparisons, strings follow lexicographical (dictionary) order. There are two rules:

1. If `s1` and `s2` have different lengths, but from the start their content is the same, then the longer string is larger.
2. If `s1` and `s2` differ in content, then scanning from the start, whichever has the larger character at the first differing position is larger.

Examples:

```c++
// all true
"aaa" == "aaa" // equal
"aaa" != "bbb" // not equal
"abcd" < "abcde" // rule 1
"abcd" > "abcc" // rule 2, d > c
"abcd" > "abcceeeeee" // rule 2, d > c, even if the right side is longer
```

The most common comparisons are `==` and `!=` to check equality. For ordering comparisons, use them when you need dictionary ordering.

An example use case for dictionary ordering:

```c++
std::vector<std::string> words; // contains many word strings: "aaa", "abc", "bbb", ....

std::sort(words.begin(), words.end(), [](auto& s1, auto& s2){
    return s1 > s2; // sort from larger letters to smaller letters
});
```

## 4. Single-character operations on string

When processing strings, it is very common to handle characters one by one. For example, given `"abcdefg"`, you might want to check whether it contains `'f'`. Or you might want to shift every character to get `"bcdefgh"`. Or you might want to detect special symbols. Either way, you have to traverse the string.

When we talk about traversal, we naturally talk about `for`. Here are the two most important forms you will need.

First: traverse a range by index:

```c++
string s("aaabbbccc");
for(size_t i = 3 ; i < s.size() ; i++) {
    std::cout << s[i];
}
// prints bbbccc
```

Here you can choose the start and end indices. In this example, `i` starts from 3.

Second: traverse all characters. Here we can use the iterator/range-based for syntax `for(declaration : expression)`. Left of the colon is the declaration of the element, and right side is the source string.

```c++
for(char c : s) {
    std::cout << c;
}
// aaabbbccc
```

Here we traverse one character at a time, copying it into `char c`, so the original `s` cannot be modified.

If you want to modify `s`, change it to `char &c`, so you access each character by reference:

```c++
for(char &c : s) {
    c += 1;
}
// s = bbbcccddd
```

## 5. Some string APIs

A very common use case is checking whether a string contains a substring, or performing operations like splitting/inserting.

Here are some common APIs:

```c++
s.find(sub_string); // find: returns the position of the first occurrence
s.replace(pos, length, new_string); // replace: from pos, replace length characters with new_string
s.substr(pos, length); // substring: from pos, take length characters
s.insert(pos, new_string); // insert: insert new_string at pos
s.contains(sub_string); // contains: whether it contains sub_string (note: only available after C++23)
```

Here is a simple example:

```cpp
std::string http_url = "http://tigercosmos.xyz/about/";

// Insert at position 4, i.e. after 'p', to get https://tigercosmos.xyz/about/
http_url.insert(4, "s"); 
// Check substring from position 0 of length 5
assert(http_url.substr(0, 5) == "https");
// Check whether the string contains "about"
assert(http_url.contains("about") == true);

// Find the starting position of substring "xyz"
size_t pos = http_url.find("xyz");
// From pos, replace 3 characters with "co.jp", to get https://tigercosmos.co.jp/about/
http_url.replace(pos, 3, "co.jp");
```

`std::string` provides many more APIs, and each API often has many overloads. In plain terms: there are multiple ways to use them. For example, `insert` can insert a string or a char. In general, when you need a feature, check whether the [string standard library](https://en.cppreference.com/w/cpp/string/basic_string) provides it. If not, implement it yourself! (Sometimes I’m lazy and want a ready-made API. Sometimes I really enjoy reinventing the wheel!)

## 6. Converting between string and numbers

Want to convert a string to a number?

You need the following functions (all in `<string>`):

```c++
std::stoi // to int
std::stol // to long int
std::stoll // to long long int
std::stoul // to unsigned long int
std::stoull // to unsigned long long int
std::stof // to float
std::stod // to double
std::stold // to long double
```

```c++
int a = std::stoi(string("5"));
double b = std::stod(string("5.5555"));
```

Note: there is no `stou`. This is a [mysterious riddle](https://stackoverflow.com/questions/8715213/why-is-there-no-stdstou).

If you want to convert a number to a string, you can use `std::to_string()`.

Here is a simple example:

```c++
int a = 5;
std::string s = "a: " + std::to_string(a);
// s = "a: 5"
```

If you just want to read a numeric string, such as `"123456"`, and operate on each digit, you don’t actually need `stoi`. Here is a small trick:

```cpp
string s("123456");
for(size_t i = 0; i < s.size(); i++) {
    int a = s[i] - '0';
}
```

Characters in a string are `char`, and each character is an [ASCII code](https://zh.wikipedia.org/wiki/ASCII). For example, the code for “A” is 65, and the code for “0” is 48.

So how do you get the digit value from a string? In the example above, `s[2]` is `'3'`. If we take the ASCII code of `'3'` (51) and subtract the ASCII code of `'0'` (48), the difference is exactly 3.

This example just shows that in many cases you do not need to convert the entire string into a number (or a number into a string). Instead, processing it character by character can sometimes inspire new algorithm ideas.

## Conclusion

This post covered many basic usage patterns for `std::string`. If there is something you don’t know how to use, you can check the string API. But in practice, the most common and simplest approach is not Google or Stack Overflow—let’s ask the almighty ChatGPT!

> **In:** help me write a C++ program that can replace all "." in a std::string "192.168.3.4" with '-'
> **Out:** Certainly! Here's a C++ program that replaces all occurrences of "." with "-" in a std::string:

```cpp
#include <iostream>
#include <string>

int main() {
    std::string str = "192.168.3.4";
    
    for (size_t i = 0; i < str.length(); i++) {
        if (str[i] == '.') {
            str.replace(i, 1, "-");
        }
    }
    
    std::cout << str << std::endl;  // Output: 192-168-3-4
    
    return 0;
}
```

But be careful—you still need to verify whether ChatGPT is making things up >.O!

> p.s. Actually, if you look at ChatGPT’s example, it is very inefficient. You could just do `str[i] = '-'` 😂
