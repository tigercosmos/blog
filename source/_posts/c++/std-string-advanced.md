---
title: 看完這一篇就夠了，讓你融會貫通使用 C++ 的 std::string 進階篇！
date: 2024-11-21 02:01:00
tags: [c++, string, ]
des: "本文深入淺出介紹各種 string 的進階使用方法，包含進階的 std::string、std::stringview、std::stringstream、std::regex、std::format、std::wstring、短字串最佳化，以及 <charconv> 函式庫。"
---

字串作為程式語言最重要的功能之一，本篇文章我們會學習 C++ 中進階的字串技巧，包含 `std::string` 的進階 API，介紹如何用 `std::stringview` 更有效率操作字串，如何用 `std::stringstream` 來處理資料流，如何使用 `std::regex` 處理正規表示式，如何利用 `std::format` 來格式化字串，如何操作 `std::wstring` 處理寬字元，瞭解什麼是短字串最佳化，以及學習如何使用 `<charconv>` 函式庫。

前篇：[看完這一篇就夠了，讓你融會貫通使用 C++ 的 std::string 初級篇！](/post/2023/06/c++/std-string-beginner/)
後篇：[看完這一篇就夠了，讓你融會貫通使用 C++ 的 std::string 進階篇！](/post/2024/11/c++/std-string-advanced/)

![封面照片](https://github.com/user-attachments/assets/0e911498-715e-4e9c-84fb-2c1db0e92573)
（2024 攝於九寨溝）

## 1. string 容器概念

在 C++ 中，`std::string` 不只是單純的字串，它是一個功能強大的容器（container），擁有靈活的大小調整能力。理解 `std::string` 的容量管理相關操作，可以更有效率地處理字串資料。這一節將介紹 `capacity`、`size`、`resize`、`reserve` 以及 `push_back` 和 `pop_back` 這些和容器相關的重要操作。

### 1.1 capacity 與 size

在操作 `std::string` 時，你會經常遇到 `size()` 和 `capacity()` 這兩個函數。`size()` 表示目前字串的實際長度，也就是你存入的字元數量，為實際數量。而 `capacity()` 則是字串目前分配的記憶體大小，也就是能夠容納的字元數量，也就是容量。數量永遠會小於等於容量。

例如，以下範例展示了 `size()` 和 `capacity()` 的差異：

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

`std::string` 通常會分配比實際使用更多的記憶體，以減少多次重新分配的成本，例如此範例中 `capacity` 是 15，而 `size` 為 5，接下來如果還要增加新字元時並不會重新分配記憶體，要一直到容量用磬，`std::string` 才會再次分配記憶體。注意到重新分配記憶體是很貴的操作，因為背後的原理是新建一個兩倍長的記憶體空間，然後將原本的記憶體資料複製到新的區間。

### 1.2 resize

`resize` 函數允許你手動調整字串的長度。如果新長度比原來長，`std::string` 會自動填充額外的空間（預設用 `\0`）。如果新長度比原來短，則會截斷字串。

```cpp
std::string str = "Hello";
str.resize(10); // str 變成 "Hello\0\0\0\0\0"
std::cout << "New Size: " << str.size() << std::endl; // 10
str.resize(3); // str 變成 "Hel"
std::cout << "New Size: " << str.size() << std::endl; // 3
```

注意到 `resize` 在改變字串的長度時，做的是「填滿」該字串，所以假設你 `str.resize(10)`，則此時 `size` 和 `capacity` 都會是 10。

### 1.3 reserve

當 `std::string` 容量不足時要重新分配記憶體的操作非常貴，所以如果我們事先就知道要用多少記憶體時，就可以先一次分配好，這就是 `reserve` 函數的功用。`reserve` 只會影響 `capacity`，不會改變 `size`。

```cpp
std::string str;
str.reserve(100); // 預先分配 100 字元的空間
std::cout << "Capacity after reserve: " << str.capacity() << std::endl; // 至少是 100
std::cout << "Size after reserve: " << str.size() << std::endl; // 0
```

預留好空間之後，你可以使用 `str += str2` 或是 `str.push_back(c)` 來向 `str` 字串添加新內容，此時因為記憶體空間已經事先預留好，因此不會有任何記憶體分配產生，直到我們將 `str` 的空間用磬，才會再次發生記憶體分配，因此當我們用 `reserve` 預留空間時，要注意之後避免用超過事先安排的空間大小。

使用 `reserve` 可以在處理大量資料時減少記憶體重新配置的次數而提高效能，如果你已經知道要進行大量的字串操作，則此時先將記憶體分配好就可以避免之後不必要的記憶體重新分配。

> 小提示：使用 `size()` 時因為字串已經被填滿，所以通常會使用 `str[i]` 來改變字串內容；而使用 `reserve()` 的時候，由於只是預留空間，此時你可以使用 `str += str2` 或是 `str.push_back(c)` 來向原本的字串後端添加新字元。

### 1.4 push_back 和 pop_back

`push_back` 函數允許你在 `std::string` 的末尾追加單一字元。如果空間不足的話，則會導致重新分配記憶體。

```cpp
std::string str = "Hello";
str.push_back('!'); // 等同 str += '!'
std::cout << str << std::endl; // Hello!
```

`pop_back` 函數允許你從 `std::string` 的末尾丟掉一個字元。

```cpp
std::string str = "Hello";
str.pop_back();
std::cout << str << std::endl; // Hell
```

單一操作字元有時候在使用字串時會非常方便，像是我們有時候要用 for 迴圈逐一處理每個字元，並且搭配 `reserve` 使用時基本上可以確保我們一直都在容量內。

> 注意到本小節介紹的函數和 `std::vector` 用法是一模一樣的，因為其實 `std::string` 可以理解成 `std::vector<char>`。

## 2. string 與 C-style string

在 C++ 中，`std::string` 是一個強大的容器，但傳統的 C-style 字串（`char*`）在某些場景下依然不可避免，例如和 C 語言的函式庫進行互動時，就只能使用 C-style 字串。

### 2.1 C-style 字串轉換

`std::string` 要轉成 C-style 字串很簡單，只需使用 `.c_str()` 就能取得 `const char*` 字串。

```cpp
std::string filename = "data.txt";
const char* c_filename = str.c_str();
FILE* file = fopen(c_filename, "w"); // POSIX  API 只接受 C-style 字串
```

如果需要非 `const` 的 `char*`，可以使用 `std::vector<char>` 或是 `std::array<char>` 建立新的記憶體空間，並將字串複製過去。

### 2.2 C-style 字串與 std::string 的轉換

從 C-style 字串轉換到 `std::string` 非常簡單，直接使用 `std::string` 的建構子即可：

```cpp
const char* c_str = "Hello";
std::string str(c_str);
std::cout << str << std::endl; // 輸出 "Hello"
```

> 在現代 C++ 開發中，我們應該優先選擇 `std::string` 來處理字串。

## 3. string_view

`std::string_view` 是 C++17 引入的新型別，用來提供輕量級且不擁有資料的字串視圖（string view）。它類似於指標，可以參考現有的字串而不需要複製資料，因此減少不必要的記憶體開銷，此外 `std::string_view` 非常適合處理字串的局部操作，例如去頭去尾、取得子字串等，傳統上以 `std::string` 操作的話一定會有額外記憶體開銷。

> 延伸閱讀：[還在用 `const std::string &`? 試試 `std::string_view` 吧！](/post/2023/06/c++/stringview/)

### 3.1 string_view 的基本用法

`std::string_view` 可以從 `std::string` 或 C-style 字串初始化，並可以像 `std::string` 一樣使用許多操作函數，但不同的地方是，使用 `std::string_view` 可以避免不必要的字串複製。

```cpp
#include <string_view>

void print_string(std::string_view sv) {
    std::cout << sv.substr(0,3) << std::endl; // 列印子字串
}

int main() {
    std::string str = "Hello, world!";
    print_string(str); // 以 std::string 傳入
    print_string("Temporary C-string"); // 以 C-style 字串傳入
    return 0;
}
```

在上面範例中我們使用了 `substr()` 函式來得到子字串，如果是 `std::string::substr` 的話此時已經經過一次記憶體拷貝了，但幸運的是 `std::string_view` 是一個視圖，它是去參考原始字串的記憶體，因此此時我們用 `std::string_view::substr` 得到的子字串也會是一個 `std::string_view`，並且過程中沒有任何記憶體複製！


> `std::string_view` 的生存週期要小心管理，因為它不擁有底層資料，因此底層資料的生存週期必須比 `std::string_view` 長。使用 `std::string_view` 時要避免指向暫時性資料（例如局部變數的字串）。


## 4. stringstream

`std::stringstream` 是字串的資料流，概念跟 `std::cin` 與 `std::cout` 很像，只是前者是將資料導入或導出「字串流」，後者將資料輸入輸出到 I/O 中。`std::stringstream` 在處理格式化、輸入輸出、資料轉換時非常有用。

### 4.1 基本用法

我們可以用 `std::stringstream` 單純去編輯字串，例如常見的用法：

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

`std::stringstream` 讓我們可以像用 `std::cout` 一樣用 `<<` 去將資料輸入進字串流中，結束之後我們可以再用 `ss.str()` 去將字串流轉換成 `std::string`。


另外，`std::stringstream` 可以像 `std::cout` 和 `std::cin` 一樣進行資料讀取和寫入，舉例來說可以將數字轉換為字串或從字串解析數字：

```cpp
#include <sstream>
#include <iostream>
#include <string>
int main() {
    std::stringstream ss;
    ss << 123 << " " << 234; // 將資料輸入 stringstream
    std::string result = ss.str(); // 轉換成 std::string
    std::cout << result << std::endl; // 輸出 「123 234」
    
    // ss 裡面還保有「123 234」
    int number;
    ss >> number; // 從 stringstream 中讀取數字，從左到右開始
    std::cout << number << std::endl; // 輸出 「123」
    return 0;
}
```



### 4.2 stringstream 的應用

例如，在處理 CSV 或其他結構化資料時，我們可以用 `std::stringstream` 來當作資料流來源。

```cpp
#include <sstream>
#include <string>
#include <vector>
#include <iostream>

int main() {
    std::string line = "apple,banana,orange";
    std::stringstream ss(line); // 資料流，可以來自網路或檔案
    
    std::string item;
    std::vector<std::string> items;

    while (std::getline(ss, item, ',')) {
        items.push_back(item);
    }

    return 0;
}
```


## 5. regex

regex（regular expression，正規表示式）常用來處理字串的模式匹配與查找，包含檢查格式、搜尋特定模式、或是進行字串取代，而 C++ 的標準庫提供了 `<regex>` 函式庫來處理 regex。

> 沒用過 regex 的讀者可以先閱讀 [MDN 的基本語法教學](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Regular_expressions)。

### 5.1 基本用法

使用 `regex` 最常見的用法是使用 `std::regex` 物件與 `std::regex_match` 或 `std::regex_search` 來進行比對：

```cpp
#include <iostream>
#include <regex>
#include <string>

int main() {
    std::string input = "hello123";
    std::regex pattern("[a-z]+\\d+"); // 定義正規表示式的模式
    if (std::regex_match(input, pattern)) {
        std::cout << "Input matches the pattern!" << std::endl;
    }
    return 0;
}
```

在上面的例子中，我們定義了一個正規表示式模式 `[a-z]+\\d+`，這個模式表示「至少一個小寫字母後接著一個或多個數字」。透過 `std::regex_match`，我們可以檢查整個字串是否符合這個模式。

### 5.2 字串搜尋與取代

除了 `std::regex_match`，我們也可以使用 `std::regex_search` 來在字串中搜尋符合的模式，或使用 `std::regex_replace` 來進行字串替換：

```cpp
#include <iostream>
#include <regex>
#include <string>

int main() {
    std::string input = "abc123def456";
    std::regex pattern("\\d+");

    // 搜尋
    std::smatch match;
    if (std::regex_search(input, match, pattern)) {
        std::cout << "Found number: " << match.str() << std::endl;
    }

    // 替換
    std::string replaced = std::regex_replace(input, pattern, "#");
    std::cout << "Replaced string: " << replaced << std::endl;

    return 0;
}
```

這裡我們使用 `std::regex_search` 來尋找字串中第一個符合 `\\d+`（一個或多個數字）的部分，然後利用 `std::regex_replace` 將所有數字取代為 `#`。

> 雖然 `regex` 功能強大，但其效能非常低，其實也很直覺，模式匹配其實就是反覆的去驗證字串，複雜度絕對不會低。所以如果有更好的作法可以去檢查字串的話，我們要盡量避免使用 `regex`。

## 6. format

很多語言都有內建的字串格式化工具，例如 Python、Rust、Golang，終於 C++20 引入了 `std::format`，這是一個新的字串格式化工具，比傳統的 `std::sprintf` 或 `std::ostringstream` 更加現代且安全。`std::format` 允許我們使用簡潔且可讀性高的方式來格式化字串。

注意要 C++20 才支援 `std::format`，所以至少要 `g++-13` 以上（Ubuntu24 預設）才支援。如果只能使用 C++17 的話，也可以考慮 [libfmt](https://github.com/fmtlib/fmt)，基本上提供一樣的功能，只不過就不是 std 的一部分，在設定專案上會有點小麻煩。

### 6.1 基本用法

我們先來看看最基本的用法：

```cpp
#include <iostream>
#include <format>

int main() {
    int number = 42;
    std::string name = "Alice";

    // 使用 std::format 進行格式化
    std::string result = std::format("Hello, {}! Your number is {}.", name, number);
    std::cout << result << std::endl;

    return 0;
}
```

在這個例子中，我們使用 `std::format` 來替換 `{}` 內的參數。它支援多種型別的輸入，不需要擔心格式錯誤的風險，這也是它比傳統的 `sprintf` 更安全的原因。

### 6.2 格式化參數

`std::format` 支援不同的格式化參數，例如設定數字的進位、寬度、填充字元等：

```cpp
#include <iostream>
#include <format>

int main() {
    int number = 255;

    std::cout << std::format("Hexadecimal: {:#x}\n", number); // 16 進位表示
    std::cout << std::format("Padded number: {:08}\n", number); // 以 0 填充至 8 位數
    std::cout << std::format("Scientific notation: {:.2e}\n", 12345.6789); // 科學記號格式

    return 0;
}
```

輸出結果：

```
Hexadecimal: 0xff
Padded number: 00000255
Scientific notation: 1.23e+04
```


基本上使用起來跟 `boost::format` 或 `std::cout` 差不多，一樣有很多格式彈性。

## 7. wstring

一般 string 的字元只佔用一個位元，其數值最多到 256，也就是可以表達 256 種字元，但顯然要支援 Unicode 字元遠遠不夠用（多達上萬個字元），根據不同語言，其字元佔用的長度也不一樣，像是一般的 CJK（中日韓）字元屬於多位元組字元（multibyte character），會用到 2（UTF-16）至 4 （UTF-32）個位元來表達。這時候就需要使用 `std::wstring`，它是 `std::string` 的寬字元（wide character）版本，使用 `wchar_t` 來儲存每一個字元，適合處理 Unicode 或其他多位元組編碼的字元。

### 7.1 基本用法

使用 `std::wstring` 和 `std::string` 非常相似，但要注意的是它需要搭配 `L""` 字面值來進行初始化：

```cpp
#include <iostream>
#include <string>

int main() {
    std::ios::sync_with_stdio(false);
    std::wcout.imbue(std::locale("en_US.utf8"));

    std::wstring ws = L"你好，世界"; // 使用 L"" 字面值初始化
    std::wcout << ws << std::endl;
    std::wcout << "Length: " << ws.size() << std::endl; // 5

    return 0;
}
```

這裡使用 `std::wcout` 來輸出寬字元字串，通常在處理寬字元時需要注意輸出流的支援情況。

> 注意我們並不知道編譯器底層實際是使用哪種編碼方式，有可能是 UTF-16、UTF-32 或者其它。

> 我們這邊使用 `sync_with_stdio(false)`，這是因為 C++ 預設是跟 C 語言相容，若是沒有把相容性關掉的話，寬字元依舊會被用一般的窄字元來去解讀，導致印出來的結果錯誤，細節可以看[這邊](https://stackoverflow.com/a/31577195/6798649)。

雖然我們也可以使用一般的 string 來儲存寬字元，但寬字元就會被一個一個 `char` 的形式被儲存，如果我們要對每個寬字元處理時，使用 string 就會不好用。

例如你可以跑看看下面範例：

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

你會發現使用寬字元時可以確保每個字都是正確被印出來，反之用 string 的窄字元則會都是亂碼。

### 7.2 使用 codecvt 進行編碼轉換

在處理 `std::string` 和 `std::wstring` 之間的轉換時，`<codecvt>` 函式庫提供了字元編碼的轉換功能，包括窄字元與寬字元的轉換。


雖然 `<codecvt>` 在 C++17 已被棄用，但它仍然在許多現有的專案中被使用。我們可以用 `std::wstring_convert` 和 `std::codecvt_utf8` 來進行 UTF-8 和寬字元的轉換：

```cpp
#include <iostream>
#include <string>
#include <locale>
#include <codecvt>

int main() {
    std::wstring wide_string = L"こんにちは";
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;

    // 寬字元轉換成 UTF-8
    std::string utf8_string = converter.to_bytes(wide_string);
    std::cout << "UTF-8: " << utf8_string << std::endl;

    // UTF-8 轉換成寬字元
    std::wstring converted_back = converter.from_bytes(utf8_string);
    std::wcout << L"Wide: " << converted_back << std::endl;

    return 0;
}
```

雖然 `<codecvt>` 可以用，但由於已經被棄用，因此在 C++20 之後，推薦使用其他的 Unicode 處理庫，如 Boost.Locale 來處理字串編碼轉換。

> 在新的標準被制訂出來之前，`<codecvt>` 函式都不會被移除，因此目前 C++20 版本都還能繼續用。

## 8. small string

C++ 短字串最佳化（short string optimization, SSO）是一種針對短字串的優化技術，通常標準的 `std::string` 在內部會為短字串保留一小塊固定的緩衝區，避免動態記憶體分配。如果字串的長度小於某個閾值（通常是 15 或 23 個字元，依不同的編譯器與實現而異），那麼它會直接在 stack 上儲存字串資料，而非進行 heap 記憶體分配。

範例：

```cpp
#include <iostream>
#include <string>

int main() {
    std::string small_string = "short"; // 可能使用 SSO
    std::string large_string = "this is a very long string that might not fit in SSO";

    // Small string: size: 5, capacity: 15
    std::cout << "Small string: size: " << small_string.size() << ", capacity: " << small_string.capacity() << std::endl;

    // Large string: size: 52, capacity: 52
    std::cout << "Large string: size: " << large_string.size() << ", capacity: " << large_string.capacity() << std::endl;

    return 0;
}
```

在這個例子中，`small_string` 使用 SSO 優化，因為它的長度足夠短，所以我們看到雖然只有 5 個字元，但容量確有 15；而 `large_string` 則會進行堆積記憶體分配。

SSO 技術能夠顯著減少短字串操作的記憶體分配成本，從而提高效能。雖然大多數時候我們不需要特別關注短字串的最佳化問題，但在效能比較敏感的情境中，我們仍須注意 SSO 帶來的幫助和影響。

> 延伸閱讀： [C++ 短字串最佳化（Short String Optimization）](/post/2022/06/c++/sso/)

## 9. to_chars & from_chars

C++17 引入了新的數值字串轉換函式庫 `<charconv>`，其中 `std::to_chars` 和 `std::from_chars` 提供高效的字串與數字互轉的功能。`<charconv>` 是 header-only，所以非常輕量，並且使用最先進的演算法，使其效率非常高。

> 延伸學習：演講 [Stephan T. Lavavej “Floating-Point ＜charconv＞: Making Your Code 10x Faster With C++17's Final Boss”](https://www.youtube.com/watch?v=4P_kbF0EbZM)，影片 45 分左右位置，你會看到 `<charconv>` 比原本作法還要快好幾倍的實驗數據。

### 9.1 基本用法

#### 9.1.1 std::to_chars
 
我們可以從數字轉換成字串，`std::to_chars` 範例：

```cpp
#include <iostream>
#include <charconv>
#include <array>

int main() {
    int number = 12345;
    std::array<char, 20> buffer; // 提前分配的緩衝區

    // 使用 std::to_chars 進行轉換
    auto [ptr, ec] = std::to_chars(buffer.data(), buffer.data() + buffer.size(), number);

    if (ec == std::errc()) { // 檢查是否成功
        std::cout << "Converted number: " << std::string(buffer.data(), ptr) << std::endl;
    } else {
        std::cout << "Conversion failed." << std::endl;
    }

    return 0;
}
```

在這個例子中，我們定義了一個 `buffer`，然後使用 `std::to_chars` 將整數轉換成字串。`std::to_chars` 返回一個包含結果指標（`ptr`）和錯誤代碼（`ec`）的 `std::to_chars_result` 結構。

`ptr` 指向的是被完整處理好的指標位置，也就是說，如果成功轉換，`ptr` 就會是成功轉換好的字串的尾端，因此我們可以使用 `std::string(buffer.data(), ptr)` 來得到轉換好的字串。

#### 9.1.2 std::from_chars

我們也可以從字串轉換成數字，`std::from_chars` 範例：

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

如果成功轉換，`resultInt` 就會是正確結果，反之我們可以用 `ptr` 去檢查最後被處理到的指標位置，代表從那個位置開始，原始的 `intStr` 並不是能被處理的數值。

例如如果將 `intStr` 給入 `12345 abc`，會得到成功訊息， `ptr` 會指向 ` abc`，因為後面部分是無法被解讀的。反之，如果給入 `abc123`，則會直接得到轉換錯誤的訊息，並且 `ptr` 指向 `abc123` 的最一開始。

### 9.2 進階用法

`std::to_chars` 也支援不同的進位制（例如十六進位）和浮點數的轉換：

```cpp
#include <iostream>
#include <charconv>
#include <array>

int main() {
    double value = 3.14159;
    std::array<char, 20> buffer;

    // 浮點數轉換
    auto [ptr, ec] = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);

    if (ec == std::errc()) {
        std::cout << "Converted float: " << std::string(buffer.data(), ptr) << std::endl;
    } else {
        std::cout << "Conversion failed." << std::endl;
    }

    int value = 8;
    std::array<char, 20> buffer;

    // 不同進位
    [ptr, ec] = std::to_chars(buffer.data(), buffer.data() + buffer.size(), 8 /* 八進位 */);

    if (ec == std::errc()) {
        std::cout << "Converted int: " << std::string(buffer.data(), ptr) << std::endl;
    } else {
        std::cout << "Conversion failed." << std::endl;
    }

    return 0;
}
```

解果會分別得到字串的 `3.14159` 與字串的 `10` （十進位的 8 等於八進位的 10）

> 儘管 `std::to_chars` 在效率上比傳統的字串轉換方法更好，但它目前僅支援基本的數字型別（例如整數和浮點數），不支援複雜的格式化操作。如果需要格式化字串的靈活性，可以考慮使用 `std::format` 來輔助。
