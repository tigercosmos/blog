---
title:  入門看完這一篇就夠了，讓你融會貫通使用 C++ 的 std::string 初級篇！
date: 2023-06-21 00:01:00
tags: [c++, string, ]
des: "本文深入淺出介紹 std::string，完整介紹各種 string 的使用方法"
---

## 1. string 簡介

不管你是以 C++ 為第一個語言程式初學者，或者從 C 或 Python 來的開發者，想必你都意識到「字串」在一個程式語言中是多麼的重要，在 C++ 中字串們會使用 `std::string` 函式庫（本文之後皆以 string 代稱），第一次聽到嗎？別擔心，這篇文章會詳細解說所有關於 string 你該知道的知識和細節。

這篇文章宗旨是讓稍微有寫程式經驗的初學者得以融會貫通如何使用 string，建議初學者至少把隨便一本 C++ 教學書前半部大致翻過（至少要知道什麼是變數，迴圈是什麼），過程中會出現許多專業的術語，我盡量用簡單的話去解釋，但畢竟受限版面不能一一解釋，最好的辦法是看不懂得術語，直接去問一下 ChatGPT；而對於比較進階的讀者，可能比較簡單的部分都知道了，但應該會有一些學習的盲點，可以透過本文補齊。

為什麼需要 string 呢？

想一下不靠 string 的話我們要怎麼去表達一個字串？

```cpp
const char *a = "abcdfe";
char b[] = "123345";
```

這樣不是不可以，畢竟我們在 C 語言基本上都這樣幹的。但是就沒有很方便，比方說我想知道長度是多少，想要搜尋子字串，想要想加字串，都沒有很方便！

string 的話就讓世界變簡單了！

```cpp
string s = "aabbcc";
s.size(); // return 6
s.find("bb"); // return 2
s += "123"; // s = aabbcc123
```

在我們開始學習之前，記得之後的程式要使用 `std::string` 時都必須引入 `<string>` 函式庫喔！

另外 string 是標準函式庫，一般來說都是要用 `std::string` 來操作，為求精簡，本文皆假設已經宣靠 `using std::string;`，這樣不需要一直打 `std::`。不了解這邊的同學可以查一下 `using` 關鍵字，還有 `namespace` 關鍵字。

所以預設範例程式碼都包含前提：

```c++
#include <string>
using std::string;
```

本篇文章是「**基本篇**」，會帶你認識和了解 string 的基本操作和用法，接下來讓我們來學習 string 吧！

![cover image](https://github.com/tigercosmos/blog/assets/18013815/3fdf2165-cb58-4a01-8698-bca0fec93acb)

##  2. string 的宣告初始化

讓我們來看怎麼樣去宣告 string：

```c++
string s1; // 預設初始化，為空字串
string s2 = s1; // s2 是 s1 的拷貝
string s3 = "hello world"; // s3 是字串值的拷貝（先轉型成 string）
string s4("hello world"); // s4 以給予的字串來初始化
```

接著要談一下直接初始化（direct initialization）、拷貝初始化（copy initialization）、移動初始化（move initialization）

```c++
string s5("abc"); // 直接初始化
string s6 = "abc"; // 拷貝初始化
string s7 = string("abc"); // 拷貝初始化
string s8(s7); // 拷貝初始化
string s9 = s8; // 拷貝初始化
string s10(std::move(s9)); // 移動初始化
```

先介紹 s5，非常直觀就是我們直接告訴 string 要用什麼值去做初始化。

s6 是很多新手會用的寫法，事實上 s6 跟 s7 是完全等價的，s6 例子中右值其實會做隱含轉型（implicit），轉型後就等同 s7。

s6 或 s7 例子是拷貝轉換，意思是我們用一個 string 物件（右值的 string）去新建一個 string 物件（左值的 string），等於你需要生成 string 兩次，所以很多新手不知道的是，用這種語法做初始化其實非常沒效率，但幸運的是大多數現代編譯器都會幫你做最佳化，實際上編譯完其實可能也沒有差，但我們仍然需要了解在沒有編譯器最佳化幫助下這些初始化的差異。

s8、s9 都是拷貝初始化，不過通常用這種語法的時候是明確知道我們需要做字串複製，所以沒什麼大問題。

s10 是移動初始化，白話解釋就是這邊把 s9 的 string 裡面的資料直接「讓給」s10，此時 s9 就不能用，而 s10 因為直接拿了 s9 的資料，所以初始化比較有效率。這邊牽扯到 `std::move` 以及右值（rvalue）的概念，這邊先提個頭，有興趣的讀者可以根據關鍵字去做延伸學習。

完整的初始化方式可以參考 string 的[建構子](https://en.cppreference.com/w/cpp/string/basic_string/basic_string)列表，不過大多數的建構方法都比較進階了，我們可以簡單記住以上幾種 string 初始化方式，如果你知道怎麼使用 `std::vector`，vector 的建構子方法也適用 string。

最後介紹一個特別的初始化方法：

```cpp
using namespace std::literals;
string s3_2 = "hello world"s; // s3 的另一種寫法，""s 會直接讓右值是 string
```

透過 `""s` 運算子，我們可以直接讓字元字串字面值（character string literals）直接宣告成 string。不過要記住必須加入 `using namespace std::literals;`。

## 3. string 的基本運算操作

以下是 string 可以進行的運算操作，先列出常用的操作：

```c++
os << s // 輸出 s 
is >> s // 輸入 s
s.empty() // 檢查 s 是否為空
s.size() // s 目前長度
s[n] // 直接取得 s 的第 n 個元素
s1 + s2 // 把 s1 加 s2 取得新的字串
s1.append(s2) // 把 s2 加到 s1 後面
s1 = s2 //　拷貝複製 s2
s1 != s2 // 比較 s1 和 s2 是否相同
<, <=, ==, >=, > // 做大小比較，以字典排序
```

### 3.1 string 的輸入和輸出

我們可以簡單寫一個程式來做程式輸入和輸出：

```c++
#include <iostream>
#include <string>
int main() {
    std::string input;
    while(std::cin >> input) { // 不斷讀取資料，直到遇到 EOF（檔案終止符號）
        std::cout << input << std::endl; // 輸出剛剛得到的 input
    }
    return 0;
}
```

> 不知道 `std::cin` 和 `std::cout` 的同學可以查一下 `iostream` 用法。`std::endl` 代表換行符號。

你也可以讀取一整行

```c++
#include <iostream>
#include <string>
int main() {
    std::string line;
    while(std::getline(std::cin, line)) { // 不斷讀取資料，一次一行（以 \n 分行），直到遇到 EOF（檔案終止符號）
        std::cout << line << std::endl; // 輸出剛剛得到的 input
    }
    return 0;
}
```

當然這邊不一定要是 `ostream`（`std::cin`, `std::cout`），例如之後我們會介紹 `stringstream` 也適用 `>>`、`<<`，只要是 stream 基本上都是用，甚至你還可以客製化 `>>`、`<<` 運算子讓 string 可以輸入輸出自定義的 C++ 物件，但這邊就超過討論範圍了。

### 3.2 `empty()` 和 `size()`

`empty()` 和 `size()` 是用來檢查 string 常用的函數。

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

例如剛剛 `getline` 的例子，如果我們想跳過空的行，就可以這樣寫：

```c++
    while(std::getline(std::cin, line)) {
        if(!line.empty()) { // 確認不是空字串
            std::cout << line << std::endl;
        }
    }
```

注意雖然 `!s.empty()` 意思等價 `s.size() > 0`，但以 `!s.empty()` 來表示不為空字串可以視為更簡潔的寫法。


另外 `size()`  的數值型別為 `string::size_type`，實際型別視標準函式庫的實作決定，一般來說就是 `size_t`（非負整數）。所以 `s.size()` 的值不是 `int`！

所以當我們要遍歷一個 string 時，以下寫法是錯的：

```c++
// 錯誤！
for(int i = 0; i < s.size(); ++1) {
    std::cout << s[i];
}
```

正確的話應該寫成：

```c++
// 正確
for(std::string::size_type i = 0; i < s.size(); ++1) {
    std::cout << s[i];
}
```

> 別擔心，馬上就會解釋 `for` 了！

你也可以用 `size_t`，一般來說都是對的，或是你很懶的話你就直接用 `auto`。

在這種情況你寫 `int` 當然也可以，因為 `int` 跟 `size_t`（`s.size()`）仍然可以做比較，但是有些情況會導致錯誤，例如去比較 `s.size() < n`，且當 `n` 為 `int` 負數時，這時候 `n` 會被轉型成 `size_t`，而變成一個無限大的正數，因此就永遠會是 `true`。

### 3.3 字串的存取

我們如何去存取 string 裡面的字元呢？

最簡單的語法有兩個 `s[]` 和 `s.at()`

```c++
string s("0123456789");
 
s[2] = 'a'; // s = "01a3456789"
std::cout << s[9]; // 9

s.at(3) = '6'; "01a6456789"
std::cout << s.at(3); // 6
```

看起來 `[]` 和 `at()` 好像一樣對吧？

其實他們差在有沒有做邊界檢查：
```c++
std::cout << s[100]; // 故意存取超出邊界
// 未定義行為，可能是亂碼，也可能 Segmentation Fault

std::cout << s.at(100); // 故意存取超出邊界
// terminate called after throwing an instance of 'std::out_of_range'
//   what():  basic_string::at: __n (which is 100) >= this->size() (which is 10)
// Aborted
```

可以看到 `at()` 幫我們做了邊界檢查，明確告訴你這程式碼有問題，這時你還可以搭配 try-catch 語法，來做錯誤處理。反之，直接用 `[]` 存取邊界以外則是未定義行為，多數時候你會得到 segmentation fault。不過這代表 `[]` 不好嗎？其實不然，邊界檢查是有效能代價的（畢竟多做了檢查），用 `[]` 比較直觀且會比較有效率，只是開發者自己必須很小心去處理存取的邊界問題。

我們可以這樣做檢查：

```c++
string s("abcd");
size_t index = /* any number */;
if(index >=0 && index < s.size()) {
    std::cout << s[index];
}

另外有兩個函數也滿常使用的，`s.front() 和 `s.back()`。顧名思義，存取最前面跟存取最後面。

```c++
string s("abc");
std::cout << s.front(); // a
std::cout << s.back(); //c
```

你當然也可以用 `s[0]` 和 `s[s.size() - 1]`，只是比較不直觀且比較醜。

### 3.4 string 的相加

#### 3.4.1 string 彼此相加
把兩個字串相加，讓 `"abc"` 和 `"defg"` 變成 `"abcdefg"`。

一個簡單的作法就是用 `+` 運算子，例如 `s1 + s2`。另一個作法是 `s1.append(s2)`。

先來看個範例：

```c++
string s1("aaa");
string s2("bbb");

string s3 = s1 + s2; // s3 = "aaabbb"

s1 += s2; // s1 = s1 + s2
s1.append(s2); // 結果等同上式，但比較有效率
```

`s1 + s2` 運算可以讓兩個 string 去產生一個新個 string，而 `s1.append(s2)` 則是把 `s2` 加到 `s1` 後面。兩者差別在於前者會先拷貝 `s1` 再拷貝 `s2` 產生出一個新的 string。後者則是只會拷貝 `s2` 並放到 `s1` 後面。可以發現 `+` 會使 `s1` 也拷貝，並多產生一個 string 物件。

如果你不想去動到 `s1` 和 `s2`，那麼用 `+` 正好。

如果你不在乎 `s1` 是否被修改，那就用 `append`，可以減少不必要的拷貝。

上面例子中，`s1 += s2` 其實就是 `s1 + s2` 產生一個新的 string，再拷貝給 `s1` 使其取代。那麼還不如一開始就直接用 `s1.append(s2)`，還比較有效率！

#### 3.4.2 string 與字面值相加

string 也可以跟字元字面值（character literals）和字元字串字面值（character string literals）互相相加。概念也很簡單，其實就是他會自己把字面值做轉型。

不過當我們混合 string 與字面值相加時，`+` 左右必須有一個是 string。

```c++
string s1 = "hello";
string s2 = "world";
string s3 = s1 + ' ' + s2 + "!\n"; // OK
string s4 = "123" + "567"; // 錯誤，不能直接相加兩個字面值
string s5 = "123"s + "567"s; // OK，等同 string 相加
string s6 = s1 + "aaa" + "bbb"; // OK，等同 s1 + "aaa" 產生一個新 string，新 string 與 "bbb" 相加
string s7 = "aaa" + "bbb" + s1; // 錯誤，"aaa" + "bbb" 會先運算，兩個字面值不能相加
string s8 = s1 + "aaa"; // OK
string s9 = "aaa" + s1; // OK
```

### 3.5 兩個 string 的比較

接下來介紹 `<, <=, ==, !=, >=, >`。

`==` 很簡單理解，兩個字串一樣長，內容都一樣，`s1 == s2` 就為真。

而當我們需要比大小時，有兩條規則：
1. s1 和 s2 不同長度，但從其面數來內容都一樣，這時比較長的字串比較大
2. s1 和 s2 不同長度、不同內容，這時從前面數來，第一個不同值的字元誰大，那個字串就是比較大

直接上例子：

```c++
// 以下皆為 true
"aaa" == "aaa" // 相同
"aaa" != "bbb" // 不相同
"abcd" < "abcde" // 規則 1
"abcd" > "abcc" // 規則 2，d > c
"abcd" > "abcceeeeee" // 規則 2，d > c，即使右邊比較長
```

比較常見的比較是 `==`、`!=`，個人認為直接用「大小」去比較兩字串不太直觀，實務上也很少見到這樣的操作。

## 4. string 的單字元的操作

在處理自串的時候，依序處理每一個字元也是很長見的操作。例如有一個字串 `"abcdefg"`，我們想檢查裡面有沒有包含 `'f'`，或是我們想要把每個字元都做平移，變成 `"bcdefgh"`，也有可能想要檢查一個字串裡面有沒有特殊符號。不論如何，你勢必要遍歷過整個字串來做檢查。

說到遍歷，那自然是 `for` 了，這邊介紹最主要你會需要的兩種。

第一種，指定範圍的遍歷：
```c++
string s("aaabbbccc");
for(size_t i = 3 ; i < s.size() ; i++) {
    std::cout << s[i];
}
// 印出 bbbccc
```
這邊我們可以自己決定 `i` 的起始和終點，例如這邊讓 `i` 從 3 開始。


第二種，遍歷全部字元，這邊我們可以使用迭代（Iterator）語法 `for(declaration : expression)`，冒號左邊是對字元的宣告，右邊是來源的字串。
```c++
for(char c : s) {
    std::cout << c;
}
// aaabbbccc
```
這邊我們去掃過一個一個字元，並複製存到 `char c` 裡面，所以原本的 `s` 是不可更改的。

如果我們想要去改變 `s` 的話，可以變成 `char &c`，這樣就會去存取每一個 `s` 的字元的參考。

```c++
for(char &c : s) {
    c += 1;
}
// s = bbbcccddd
```

## 5. string 的一些操作

在使用 string 一個很長見的情景就是要去查詢一個字串是否包含一個子字串，或是要對字串做切割插入等動作。

以下提供一些常使用的 API 供參考：

```c++
s.find(sub_string); // 查詢，回傳第一個發現的子字串的位置
s.replace(pos, length, new_string); // 取代，從 pos 位置，取代 length 長度，換成新的字串 new_string
s.substr(pos, length); // 擷取子字串，從 pos 位置，擷取 length 長度
s.insert(pos, new_string); // 插入，從 pos 位置，插入一個 new_string
s.contains(sub_string); // 包含，檢查有沒有包含子字串 sub_string（注意，c++23 之後才支援）
```

以下為簡單的範例：

```cpp
std::string http_url = "http://tigercosmos.xyz/about/";

// 從位置 4 地方插入，也就是 p 的下一個位置，得到 https://tigercosmos.xyz/about/
http_url.insert(4, "s"); 
// 檢查位置 0 到 5 的子字串
assert(http_url.substr(0, 5) == "https");
// 檢查字串是否包含 "about"
assert(http_url.contains("about") == true);

// 找到 "xyz" 子字串的開頭位置
size_t pos = http_url.find("xyz");
// 從剛剛找到的 pos 位置，取代三個字元，換成 "co.jp"，於是得到 https://tigercosmos.co.jp/about/
http_url.replace(pos, 3, "co.jp");
```

std::string 還提供很多 API 可以使用，而且每一個 API 還包含很多重載（overloads），白話一點就是你可以有多種使用方法，例如 insert 就可以是插入 string 或是 char。基本上就是你發現你需要什麼功能，就檢查一下 [string 標準函式庫](https://en.cppreference.com/w/cpp/string/basic_string)有沒有提供，沒提供的話就自己土砲自幹！（有時很我很懶，就會直接想拿現成 API，有時候也很享受自己造輪子的樂趣！）

 ## 6. string 與數字的型別轉換

想要把字串變成數字嗎？

你需要以下這些函數（都在 `<string>` 函式庫裡）

```c++
std::stoi // 轉 int
std::stol // 轉 long int
std::stoll // 轉 long long int
std::stoul // 轉 unsigned long int
std::stoull // 轉 unsigned long long int
std::stof // 轉 float
std::stod // 轉 double
std::stold // 轉 long double
```

```c++
int a = std::stoi(string("5"));
double b = std::stod(string("5.5555"));
```

注意喔！沒有 `stou`，這是個[神秘的迷團](https://stackoverflow.com/questions/8715213/why-is-there-no-stdstou)。

如果你想要把數字變成 string，則可以用 `std::to_string()`。

以下為簡單範例：
```c++
int a = 5;
std::string s = "a: " + std::to_string(a);
// s = "a: 5"
```

如果你只是想要讀取一串 string 型別的數字，比方說 `"123456"`，並操作每一個數字，那你其實不需要 `stoi`，可以用以下的小技巧：

```cpp
string s("123456");
for(size_t i = 0; i < s.size(); i++) {
    int a = s[i] - '0';
}
```

string 的字元為 char，所以每一個字元其實都是 [ASCII 編碼](https://zh.wikipedia.org/wiki/ASCII)的 char，例如「A」的編碼是 65，「0」的編碼是 48。

所以如何取得一個 string 裡面的的數字？比方說上面的例子，`s[2]` 是 `'3'`，我們讓 `'3'` 的 ASCII 編碼（51）去扣掉 `'0'` 的編碼（48），其差值正是 3。

上面的例子只是想說明，很多時候你其實不需要把字串直接換成數字，或是數字直接換成字串，反之我們可以一個一個字元處理，這時候你可能會有一些新的演算法的點子。

## 結論

這篇文章我們詳細介紹 std::string 的各種基本用法，如果有不太會用的地方，可以查一下 string 的 API，不過更常見也更簡單的作法，不是 Google，也不是 Stack Overflow，讓我們問一下萬能的 ChatGPT 吧！

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

不過小心喔，還是得自己檢查一下 ChatGPT 有沒有在胡說 >.O！

> p.s. 事實上你看一下 ChatGPT 給的範例，其實非常沒效率，直接 `str[i] = '-'` 不就得了 😂
 