---
title: 還在用 const std::string &? 試試 std::string_view 吧！
date: 2023-06-07 00:01:00
tags: [c++, string_view, ]
des: "本文介紹 C++17 的 std::string_view 特性以及提出範例"
lang: zh
translation_key: stringview
---

![COVER IMAGE](https://github.com/tigercosmos/blog/assets/18013815/e93fadda-2edc-4ba8-87fd-4df5569939e4)


## 簡介

`std::string` 作為 C++ 程式入門最基本的函示庫相信大家都用的很熟了，說到如何把字串傳入其他函數，一般來說常見作法就是 `const std::string &` 亦或是 `const char *`，也就是傳址輸入的概念，傳入參考（reference）或是指標（pointer）所以不需要做記憶體的複製。

舉例來說，以下程式法就非常常見：

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

而從 C++17 標準之後，我們有新的選擇——`std::string_view`，View（視圖）在軟體工程上他的概念是「開發人員以不同的視角或方式來查看和存取相同數據陣列（Array）」，意思是 View 提供了一種輕量的抽象介面，讓開發者以不同的方式操作和處理陣列數據，而無需複製或重新排列數據。這種機制可以在不佔用額外記憶體的情況下，對陣列進行切片、重塑、重新排序、重映射等操作，這樣我們可以避免不必要的物件建立或是記憶體操作。所以不管是 Array View 或是 String View（底層還是陣列），概念其實就是用輕量的操作去存取裡面內部的元素，舉例來說我們可以在 Python 的 NumPy 庫、C++ 的 Eigen 庫中看見這個概念。

現在我們稍微改一下前面的程式碼：

```c++
#include <string>
#include <iostream>
#include <string_view> // 記得要引入函示庫

void print(std::string_view input) {
    std::cout << input << std::endl;
}

int main() {
    std::string message("hello world");
    print(std::string_view(message));
}
```

接著來解釋為啥這樣做會比較好。

## std::string_view 優點

使用 `std::string_view` 有以下好處：

- **輕量、低開銷**：使用 `string_view` 可以對原始字串做任意建立、複製、傳遞操作，都不會對原始的記憶體資料作任何拷貝，反之如果用一般的 `string` 做複製的話成本就會很高
- **相容** `std::string` 的函數：我們平常對 `string` 做的函數操作幾乎都支持，比方說 Iterator、`cout` 輸出、`substr`、`find` 等等。
- **更加安全**：`string_view` 永遠不會有所有權，當你使用 `string_view` 的時候可以大膽的刪除他。
- **更高的彈性**：使用 `string_view` 的時候對不同型態的字串物件會有更高的相容性，例如你可以傳入 `std::wstring` 或 `winrt::hstring` 而不會導致錯誤發生，但 `const std::string &` 你可能型別上就直接編譯不過。
- **更快速的字串操作**：我們鮮少對 `std::string` 做 `substr` 因為這樣會產生新的 `string` 物件，成本極高。但用 `std::string_view` 我們可以用 `substr` 且速度依舊很快。根據我的實驗兩者速度可以差到 17 倍之多。另外 `string_view` 可以做前綴後綴的操作，當傳入 `const std::string &` 的時候我們則無法辦到。
- **更現代**：擺脫使用 `const std::string &` 或 `const char *`，雖然舊方法的效能和新方法一樣，但我們已經提出了不少好處了

## std::string_view 範例

### substr 子字串

前面提到，`std::string::substr` 和 `std::string_view::substr` 兩者效能差很大，實驗 `string` 呼叫 `substr` 會慢 `string_view` 17 倍，理由也簡單，`string` 的 `substr` 會再產生一個新的 `string`。

> 注意 `substr` 用法是 `substr(start, length)`，不是給頭跟尾喔！

那使用 `string` 的時候不想使用 `substr` 來做子字串比對怎麼辦呢？你可以自己操作 `string` 底層的記憶體，只是就不是很方便。


```c++
void print1(const std::string &input) {
    if(input.substr(0,3) == "123") { // 這個情況下子字串是 std::string
      // ...
    }
}

void print2(std::string_view input) {
    if(input.substr(0,3) == "123") { // 這個情況下子字串是 std::string_view
      // ...
    }
}
```

### 移除前綴＆後綴的操作

前綴（prefix）與後綴（suffix）的操作也很常見，例如我們想要移除一個檔案的前面的路徑，以及後面的副檔名。

方法一：
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

方法二：
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

這邊給了兩個範例，上面使用 `remove_prefix` 和 `remove_suffix` 做範例，注意使用這兩個函數會對「原本」的 `string_view` 物件做修改。下面範例示範了如何用 `substr` 達到一樣的效果。

從頭到尾我們沒用到任何字串複製，全程就是只有 `string_view`，並且這樣的操作又非常的「高階」，在沒有 `string_view` 的幫助下，我們想要不建立物件或拷貝記憶體來讓 `cout` 印出東西就不可能這麼優雅。

## 結論

總的來說，使用 `std::string_view` 主要是為了提高效能和節省資源。由於它僅僅是持有指向現有字串的指標和長度，不需要進行新的記憶體配置或字串複製，因此建立速度快且佔用的記憶體較少。同時，`std::string_view` 提供了一個輕量級的、唯讀 View，方便對大型字串進行高效的存取。這使得它成為處理字串引用或在函數參數傳遞時的理想選擇。

然而，使用 `const std::string &` 也有其獨特的用途，特別是當需要修改字串或使用 `std::string` 特有的功能時。在這種情況下，`std::string_view` 是無法滿足需求的，因為它僅提供了只讀的視圖。另外如果要把字串傳入其他函示庫的 API 的時候，如果 API 指定要傳入 `const std::string &`，我們也無法傳入 `string_view`。

當我們要做字串操作的時候，除非我們要對 `std::string` 進行操作，或是需要產生新的所有權，不然我們應該盡可能使用 `std::string_view`。

## 參考資料

- [std::string_view: The Duct Tape of String Types](https://devblogs.microsoft.com/cppblog/stdstring_view-the-duct-tape-of-string-types/)
- [class std::string_view in C++17](https://www.geeksforgeeks.org/class-stdstring_view-in-cpp-17/)
- [std::basic_string_view (from cppreference)](https://en.cppreference.com/w/cpp/string/basic_string_view)
