---
title: "C++ std::string 進階：この一篇で応用まで融会貫通！"
date: 2024-11-21 02:01:00
tags: [c++, string, ]
des: "本記事では string の進階的な使い方を分かりやすく紹介します。進階の std::string、std::stringview、std::stringstream、std::regex、std::format、std::wstring、短字串最佳化、そして <charconv> を扱います。"
lang: jp
translation_key: std-string-advanced
---

文字列はプログラミング言語における最重要機能の 1 つです。本記事では C++ における進階的な文字列テクニックを学びます。`std::string` の進階 API、`std::stringview` を用いた効率的な文字列操作、`std::stringstream` によるデータストリーム処理、`std::regex` を用いた正規表現、`std::format` による文字列フォーマット、`std::wstring` によるワイド文字の扱い、短字串最佳化（SSO）とは何か、そして `<charconv>` ライブラリの使い方を紹介します。

前篇：[入門看完這一篇就夠了，讓你融會貫通使用 C++ 的 std::string 初級篇！](/post/2023/06/c++/std-string-beginner/)
後篇：[看完這一篇就夠了，讓你融會貫通使用 C++ 的 std::string 進階篇！](/post/2024/11/c++/std-string-advanced/)

![封面照片](https://github.com/user-attachments/assets/0e911498-715e-4e9c-84fb-2c1db0e92573)
（2024 攝於九寨溝）

## 1. string のコンテナとしての考え方

C++ では `std::string` は単なる「文字列」ではなく、柔軟にサイズ変更できる強力なコンテナ（container）でもあります。`std::string` の容量管理を理解すると、文字列データをより効率的に扱えます。本節では、`capacity`、`size`、`resize`、`reserve`、そしてコンテナ操作として重要な `push_back` / `pop_back` を紹介します。

### 1.1 capacity と size

`std::string` を操作するとき、`size()` と `capacity()` に頻繁に遭遇します。`size()` は現在の文字列の実長（格納されている文字数）であり、`capacity()` は現在確保されているメモリ量（格納可能な文字数）、つまり容量です。常に `size <= capacity` です。

次の例は `size()` と `capacity()` の違いを示します：

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

`std::string` は再確保（reallocation）のコストを減らすため、実際に必要な量より多くのメモリを確保することが一般的です。上の例では `capacity` が 15、`size` が 5 です。次に文字を追加してもすぐには再確保されず、容量が尽きるまで再確保は起きません。再確保は高コストで、一般的には「より大きい（多くは 2 倍）領域を新規確保し、元のデータを新領域へコピーする」という処理になるためです。

### 1.2 resize

`resize` は文字列の長さを手動で変更します。新しい長さが元より長い場合、`std::string` は追加領域を自動で埋めます（デフォルトは `\0`）。短い場合は切り詰めます。

```cpp
std::string str = "Hello";
str.resize(10); // str 變成 "Hello\0\0\0\0\0"
std::cout << "New Size: " << str.size() << std::endl; // 10
str.resize(3); // str 變成 "Hel"
std::cout << "New Size: " << str.size() << std::endl; // 3
```

`resize` は「その長さまで埋める」動きなので、たとえば `str.resize(10)` をすると、その時点で `size` と `capacity` がともに 10 になります。

### 1.3 reserve

`std::string` の容量不足による再確保は非常に高コストです。そのため、必要なサイズが事前に分かっている場合は `reserve` で一括確保できます。`reserve` は `capacity` のみを変更し、`size` は変えません。

```cpp
std::string str;
str.reserve(100); // 預先分配 100 字元的空間
std::cout << "Capacity after reserve: " << str.capacity() << std::endl; // 至少是 100
std::cout << "Size after reserve: " << str.size() << std::endl; // 0
```

領域を予約した後は `str += str2` や `str.push_back(c)` で文字列の末尾に追加できます。領域が事前に確保されているので、容量を使い切るまではアロケーションは発生しません。そのため `reserve` を使う場合は、後続操作で予約容量を超えないように注意する必要があります。

`reserve` を使うことで、膨大な文字列操作の際に再確保回数を減らし、性能を向上できます。大量操作が分かっているなら、先に確保しておくと無駄な再確保を避けられます。

> 小提示：`size()` はすでに文字列が埋まっているので `str[i]` で内容を書き換えることが多いです。一方 `reserve()` は領域を予約するだけなので、`str += str2` や `str.push_back(c)` で末尾に追加するのが自然です。

### 1.4 push_back と pop_back

`push_back` は `std::string` の末尾に 1 文字追加します。容量が不足している場合は再確保が発生します。

```cpp
std::string str = "Hello";
str.push_back('!'); // 等同 str += '!'
std::cout << str << std::endl; // Hello!
```

`pop_back` は末尾から 1 文字削除します。

```cpp
std::string str = "Hello";
str.pop_back();
std::cout << str << std::endl; // Hell
```

1 文字単位の操作は便利なことが多く、たとえば for ループで 1 文字ずつ処理し、`reserve` と組み合わせれば容量内に収めやすくなります。

> 本節の関数は `std::vector` と全く同じ使い方だと気付くはずです。`std::string` は `std::vector<char>` と考えられ、本質的にはほぼ同じだからです。

## 2. string と C-style string

C++ では `std::string` が強力なコンテナですが、従来の C-style 文字列（`char*`）も一部の場面では避けられません。たとえば C 言語ライブラリと連携する場合、C-style しか受け付けないことがあります。

### 2.1 C-style 文字列への変換

`std::string` を C-style に変換するのは簡単で、`.c_str()` を呼ぶだけで `const char*` を取得できます。

```cpp
std::string filename = "data.txt";
const char* c_filename = str.c_str();
FILE* file = fopen(c_filename, "w"); // POSIX  API 只接受 C-style 字串
```

`const` ではない `char*` が必要なら、`std::vector<char>` や `std::array<char>` で新しいメモリ領域を作り、`memcpy` などでコピーします。

### 2.2 C-style 文字列から std::string への変換

C-style 文字列から `std::string` への変換も簡単で、`std::string` のコンストラクタを使うだけです：

```cpp
const char* c_str = "Hello";
std::string str(c_str);
std::cout << str << std::endl; // 輸出 "Hello"
```

> 現代の C++ 開発では、文字列処理は基本的に `std::string` を優先するべきです。

## 3. string_view

`std::string_view` は C++17 で導入された新しい型で、軽量でデータの所有権を持たない文字列ビュー（string view）です。ポインタに似ており、既存の文字列を参照してデータをコピーせずに扱えるため、不要なメモリコストを削減できます。また、`std::string_view` は部分操作（先頭や末尾の削除、部分文字列の取得など）に非常に向いており、従来 `std::string` で同様の操作をすると追加のメモリコストが発生しがちです。

> 延伸閱讀：[還在用 `const std::string &`? 試試 `std::string_view` 吧！](/post/2023/06/c++/stringview/)

### 3.1 string_view の基本用法

`std::string_view` は `std::string` や C-style 文字列から初期化でき、`std::string` に似た操作関数を多く使えます。ただし `std::string_view` を使うことで不要な文字列コピーを避けられます。

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

上の例では `substr()` で部分文字列を取得しています。もし `std::string::substr` ならメモリコピーが発生します。しかし `std::string_view` はビューであり、元の文字列メモリを参照します。そのため `std::string_view::substr` の結果も `std::string_view` になり、過程でメモリコピーは発生しません！

> `std::string_view` は生存期間の管理に注意が必要です。データを所有しないため、参照先データの寿命が `std::string_view` より長くなければいけません。`std::string_view` が一時データ（例：ローカル変数の文字列）を指すことを避けてください。

## 4. stringstream

`std::stringstream` は文字列のデータストリームです。概念は `std::cin` / `std::cout` と似ていますが、前者は「文字列ストリーム」へ入出力し、後者は I/O に入出力します。`std::stringstream` はフォーマット、入出力、データ変換などに非常に便利です。

### 4.1 基本用法

`std::stringstream` は文字列を編集する用途でよく使えます。例えば次のような書き方です：

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

上の例は配列を文字列へ出力しています。`std::stringstream` に対しても `std::cout` と同様に `<<` でデータを流し込めます。最後に `ss.str()` を呼ぶと `std::string` に変換できます。

また `std::stringstream` は `std::cout` / `std::cin` のように出力・入力もできます。たとえば数値を文字列に変換したり、文字列から数値を解析したりできます：

```cpp
#include <sstream>
#include <iostream>
#include <string>
int main() {
    std::stringstream ss;
    ss << 123 << " " << 234; // 將資料輸入 stringstream
    std::string result = ss.str(); // 轉換成 std::string
    std::cout << result << std::endl; // stdout 輸出 「123 234」
    
    // ss 裡面還保有「123 234」
    int number;
    ss >> number; // 從 stringstream 中讀取數字，從左到右開始
    std::cout << number << std::endl; // stdout 輸出 「123」
    return 0;
}
```

### 4.2 stringstream の応用

たとえば CSV やその他の構造化データを処理するとき、`std::stringstream` をデータストリームとして使えます。

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

`std::getline` の定義は `istream& getline(istream& input_stream, string& output, char delim);` です。第 1 引数は `std::istream` を受け取ります。`std::stringstream` は `std::istream` を継承しているため、そのまま `getline` を使えます。同様に、`std::getline(std::cin, ...)` を使ったことがあるなら、`std::cin` も `std::istream` を継承しています。

## 5. regex

regex（regular expression、正規表現）は文字列のパターンマッチングや検索によく使われます。フォーマットチェック、特定パターンの検索、文字列置換などができます。C++ の標準ライブラリは `<regex>` を提供しています。

> regex を使ったことがない場合は、まず [MDN の基本文法ガイド](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Regular_expressions) を読むと良いです。

### 5.1 基本用法

よくある使い方は `std::regex` と `std::regex_match` / `std::regex_search` の組み合わせです：

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

この例では正規表現パターン `[a-z]+\\d+` を定義しています。意味は「1 文字以上の小文字に続いて 1 個以上の数字」です。`std::regex_match` で文字列全体がこのパターンに一致するかを確認できます。

### 5.2 文字列検索と置換

`std::regex_match` 以外に、`std::regex_search` で文字列中の一致部分を探したり、`std::regex_replace` で置換したりできます：

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

ここでは `std::regex_search` を使って `\\d+`（1 個以上の数字）に一致する最初の部分を探し、その後 `std::regex_replace` で数字全体を `#` に置換しています。

> `regex` は強力ですが性能は非常に低いです。直感的にも、パターンマッチングは文字列検証を繰り返すため、計算量が低くなることはありません。より良い方法があるなら `regex` は避けましょう。

## 6. format

多くの言語（Python、Rust、Golang など）には文字列フォーマット機能が組み込まれています。C++20 でついに `std::format` が導入されました。これは従来の `std::sprintf` や `std::ostringstream` よりモダンで安全な文字列フォーマット機能です。`std::format` により簡潔で読みやすい形で文字列を整形できます。

`std::format` は C++20 が必要です。実際には `g++-13` 以上（Ubuntu24 のデフォルト）でないと使えません。C++17 しか使えない場合は、[libfmt](https://github.com/fmtlib/fmt) を検討できます。機能はほぼ同等ですが std の一部ではないため、プロジェクト設定が少し面倒になります。

### 6.1 基本用法

まずは基本：

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

`std::format` は `{}` の部分に引数を埋め込みます。Python の `str.format` や Rust の `format!` に似ています。引数は自動的に文字列へ変換され、埋め込まれます。

### 6.2 フォーマット指定子

`std::format` は進数、幅、埋め文字など様々な指定子をサポートします：

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

出力：

```
Hexadecimal: 0xff
Padded number: 00000255
Scientific notation: 1.23e+04
```

使い勝手は `boost::format` や `std::cout` と似た雰囲気で、多くの指定が可能です。

## 7. wstring

通常の string の `char` は 1 バイトで、最大 256 種類の値しか表現できません。Unicode を扱うには不十分です（Unicode は何万もの文字があります）。言語によって 1 文字あたりのバイト数も異なり、CJK（中日韓）の文字はマルチバイト（multibyte）で 2（UTF-16）〜 4（UTF-32）バイトを必要とします。そこで `std::wstring` を使います。これは `std::string` のワイド文字（wide character）版で、各文字を `wchar_t` として保持し、Unicode などの扱いに適します。

### 7.1 基本用法

`std::wstring` の使い方は `std::string` と非常に似ていますが、初期化に `L""` リテラルが必要です：

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

ここでは `std::wcout` を用いてワイド文字列を出力しています。ワイド文字の出力では、出力ストリームの対応状況やロケール設定に注意が必要です。

> コンパイラ内部の実際のエンコーディング方式が UTF-16 / UTF-32 / その他のどれかは分かりません。

> ここで `sync_with_stdio(false)` を使うのは、C++ がデフォルトで C と互換であり、互換性を切らないとワイド文字が狭文字として解釈されて出力が壊れることがあるためです。細部は[こちら](https://stackoverflow.com/a/31577195/6798649)を参照してください。

ワイド文字を通常の string に入れることもできますが、その場合はワイド文字が `char` の列として格納されます。ワイド文字を 1 文字ずつ処理したい場合は string では扱いにくくなります。

次の例を実行してみてください：

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

ワイド文字列なら各文字が正しく出力されますが、狭文字列では文字化けするはずです。

### 7.2 codecvt によるエンコーディング変換

`std::string` と `std::wstring` の変換には `<codecvt>` が提供する文字コード変換機能（狭文字 ↔ ワイド文字）を使えます。

`<codecvt>` は C++17 で非推奨になりましたが、既存プロジェクトではまだ使われています。`std::wstring_convert` と `std::codecvt_utf8` を使えば UTF-8 とワイド文字を変換できます：

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

`<codecvt>` は動きますが非推奨であるため、C++20 以降では Boost.Locale のような別の Unicode ライブラリを使うことが推奨されます。

> 新しい標準が策定されるまで `<codecvt>` は削除されないため、現状 C++20 でも利用できます。

## 8. small string

C++ の短字串最佳化（short string optimization, SSO）は短い文字列を最適化する技術です。`std::string` は内部に短文字列用の固定バッファを持ち、短い場合は動的メモリ確保を避けます。文字列長が閾値（実装により 15 や 23 など）より小さい場合、ヒープではなくスタック上に文字列データを保持します。

例：

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

この例では `small_string` が短いので SSO が効いている可能性があり、文字数は 5 なのに capacity が 15 になっているのが見えます。一方で `large_string` はヒープ確保が発生します。

SSO は短文字列操作におけるアロケーションコストを大きく減らし、性能向上に寄与します。多くの場合、SSO を特別に意識する必要はありませんが、性能に敏感な場面では SSO の影響を理解しておくと役立ちます。

> 延伸閱讀： [C++ 短字串最佳化（Short String Optimization）](/post/2022/06/c++/sso/)

## 9. to_chars & from_chars

C++17 では数値と文字列の変換のための新しいライブラリ `<charconv>` が導入されました。`std::to_chars` と `std::from_chars` は高性能な数値↔文字列変換を提供します。`<charconv>` は header-only で軽量であり、先進的なアルゴリズムで非常に高性能です。

> 延伸學習：講演 [Stephan T. Lavavej “Floating-Point ＜charconv＞: Making Your Code 10x Faster With C++17's Final Boss”](https://www.youtube.com/watch?v=4P_kbF0EbZM)。45 分あたりで `<charconv>` が従来手法より何倍も速いベンチマーク結果が見られます。

### 9.1 基本用法

#### 9.1.1 std::to_chars

数値から文字列に変換できます。`std::to_chars` の例：

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

この例では `buffer` を用意し、`std::to_chars` で整数を文字列に変換しています。`std::to_chars` は結果ポインタ `ptr` とエラーコード `ec` を含む `std::to_chars_result` を返します。

`ptr` は処理済みの末尾位置を指します。変換が成功した場合、`ptr` は変換結果文字列の末尾なので、`std::string(buffer.data(), ptr)` で結果文字列を作れます。

#### 9.1.2 std::from_chars

文字列から数値への変換もできます。`std::from_chars` の例：

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

変換が成功すれば `resultInt` に結果が入ります。失敗した場合は `ptr` で最後に処理できた位置を確認でき、その位置以降の `intStr` が数値として解釈できないことを意味します。

たとえば `12345 abc` を与えると成功し、`ptr` は ` abc` を指します（後半が解釈不能だからです）。一方 `abc123` を与えると変換エラーになり、`ptr` は先頭 `abc123` を指します。

### 9.2 進階用法

`std::to_chars` は異なる進数（例：16 進）や浮動小数点変換もサポートします：

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

解果はそれぞれ文字列の `3.14159` と文字列の `10`（10 進の 8 は 8 進で 10）になります。

> `<charconv>` はシンプルな変換に向いており性能も高い一方で、複雑なフォーマットはできません。フォーマットが必要なら `std::format` を検討してください。
