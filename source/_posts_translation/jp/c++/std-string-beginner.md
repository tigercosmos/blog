---
title: "C++ std::string 入門：この一篇で基礎を融会貫通！"
date: 2023-06-21 00:01:00
tags: [c++, string, ]
des: "本記事では std::string を分かりやすく解説し、string の初級使用方法を網羅的に紹介します。"
lang: jp
translation_key: std-string-beginner
---

## 1. string の概要

C++ を最初の言語として学ぶ初心者でも、C や Python から来た開発者でも、「文字列」がプログラミング言語においてどれほど重要かはすでに実感していると思います。C++ では文字列に `std::string` ライブラリ（本記事では以降 `string` と呼びます）を使います。初めて聞きましたか？大丈夫です。この一篇で、string に関して知っておくべき知識と細部をすべて丁寧に解説します。

この記事の目的は、少しでもプログラミング経験がある初学者が string を「融会貫通」できるようになることです。初学者は、少なくとも適当な C++ 教科書の前半くらいはざっと目を通しておくと良いでしょう（変数とは何か、ループとは何か、程度は分かっている状態）。途中で専門用語がたくさん出てきます。できるだけ簡単な言葉で説明しますが、紙面の都合上すべての用語を一つ一つ説明することはできません。分からない用語が出たら、ChatGPT に聞くのが一番です。上級者にとっては、簡単な部分はすでに知っているかもしれませんが、学習の盲点があるはずなので、本文で補完できると思います。

なぜ string が必要なのでしょうか？

string なしで文字列を表現しようとすると、どうなりますか？

```cpp
const char *a = "abcdfe";
char b[] = "123345";
```

もちろん不可能ではありません。C 言語では基本こうやってやります。しかし不便です。たとえば長さを知りたい、部分文字列を検索したい、文字列を連結したい……どれも面倒です！

string を使えば世界はシンプルになります：

```cpp
string s = "aabbcc";
s.size(); // return 6
s.find("bb"); // return 2
s += "123"; // s = aabbcc123
```

学習を始める前に覚えておいてください。`std::string` を使うには必ず `<string>` を include する必要があります！

また、string は標準ライブラリなので、通常は `std::string` と書きます。ただし簡潔さのため、本記事では `using std::string;` をすでに宣言した前提で説明します。これにより `std::` を毎回書かなくて済みます。ここが分からない人は、`using` キーワードと `namespace` キーワードを調べてください。

したがって、本文中のサンプルコードは次の前提を含みます：

```c++
#include <string>
using std::string;
```

本記事は「**基本篇**」です。string の基本操作と使い方を紹介します。それでは string を学びましょう！

前篇：[入門看完這一篇就夠了，讓你融會貫通使用 C++ 的 std::string 初級篇！](/post/2023/06/c++/std-string-beginner/)
後篇：[看完這一篇就夠了，讓你融會貫通使用 C++ 的 std::string 進階篇！](/post/2024/11/c++/std-string-advanced/)

![cover image](https://github.com/tigercosmos/blog/assets/18013815/3fdf2165-cb58-4a01-8698-bca0fec93acb)

## 2. string の宣言と初期化

string をどう宣言するか見てみましょう：

```c++
string s1; // デフォルト初期化。空文字列
string s2 = s1; // s2 は s1 のコピー
string s3 = "hello world"; // s3 は文字列値のコピー（いったん string に暗黙変換）
string s4("hello world"); // s4 は与えられた文字列で初期化
```

次に、direct initialization（直接初期化）、copy initialization（コピー初期化）、move initialization（ムーブ初期化）、copy assignment（コピー代入）、move assignment（ムーブ代入）について触れます。あるいは [Rule of Three](https://en.cppreference.com/w/cpp/language/rule_of_three) としてまとめて語られることもあります。

> ここからの説明は少し複雑です。初心者は、ひとまず以下の s5〜s9 のような方法で string を作れることだけ覚えておけば OK です。

```c++
string s5("abc"); // 直接初期化
string s6 = "abc"; // コピー代入
string s7 = string("abc"); // コピー代入
string s8(s7); // コピー初期化
string s9 = s8; // コピー初期化
string s10(std::move(s9)); // ムーブ初期化
string s11 = std::move(s10); // ムーブ代入
```

まず s5。これは直感的で、string にどんな値で初期化するかを直接指定しています。

s6 は非常によくある宣言方法で、実は s6 と s7 は完全に等価です。s6 の右辺値は暗黙変換（implicit）され、結果として s7 と同じになります。

s6 / s7 は「コピー変換」です。つまり、右辺の string（右辺値の string）から左辺の string（左辺値の string）を新しく作るので、概念的には string を 2 回作ることになります。理論上は非効率な書き方ですが、幸運なことに多くの現代コンパイラは最適化してくれるため、実際には差がないこともあります。ただし、最適化がない状況を前提に、初期化方法の違いを理解しておく価値はあります。

s8 / s9 はどちらもコピーです。ただし、この書き方をするときは「意図的にコピーが必要」な場合が多いので、問題になることは少ないです。

s10 は平易に言えば「s9 の内部データをそのまま s10 に譲る」動きです。このとき s9 は使えなくなり、s10 はデータをそのまま受け取るので初期化が効率的になります。s11 も同様です。ここには `std::move` や rvalue（右辺値）といった概念が関係しますが、ここでは軽く触れるだけにします。興味がある人はキーワードで調べてみてください。

初期化方法を網羅的に知りたい場合は、string の[コンストラクタ](https://en.cppreference.com/w/cpp/string/basic_string/basic_string)一覧を参照してください。ただし多くのコンストラクタは上級向けです。まずは上のような初期化方法を覚えておけば十分です。`std::vector` が使えるなら、vector のコンストラクタの考え方は string にも当てはまります。

最後に、少し特殊な初期化方法も紹介します：

```cpp
using namespace std::literals;
string s3_2 = "hello world"s; // s3 の別表現。""s により右辺値が string になる
```

`""s` 演算子を使うと、文字列リテラル（character string literals）をそのまま string として宣言できます。ただし `using namespace std::literals;` が必要です。

## 3. string の基本演算

まずはよく使う操作を列挙します：

```c++
os << s // s を出力
is >> s // s を入力
s.empty() // s が空かチェック
s.size() // s の現在の長さ
s[n] // s の n 番目の要素にアクセス
s1 + s2 // s1 と s2 を連結して新しい文字列を得る
s1.append(s2) // s1 の末尾に s2 を追加
s1 = s2 // s2 をコピー
s1 != s2 // s1 と s2 が同じか比較
<, <=, ==, >=, > // 辞書順で大小比較
```

### 3.1 入力と出力

簡単な入力と出力の例：

```c++
#include <iostream>
#include <string>
int main() {
    std::string input;
    while(std::cin >> input) { // EOF（ファイル終端）まで読み続ける
        std::cout << input << std::endl; // 読んだ input を出力
    }
    return 0;
}
```

> `std::cin` / `std::cout` が分からない人は `iostream` の使い方を調べてください。`std::endl` は改行です。

1 行ずつ読みたい場合：

```c++
#include <iostream>
#include <string>
int main() {
    std::string line;
    while(std::getline(std::cin, line)) { // \n で区切って 1 行ずつ読み、EOF まで繰り返す
        std::cout << line << std::endl; // 読んだ line を出力
    }
    return 0;
}
```

ここで使うストリームは `std::cin` / `std::cout` に限りません。たとえば後で紹介する `stringstream` でも `>>` と `<<` が使えます。ストリームであれば基本的に同じ操作が可能です。さらに `>>` / `<<` を自分でオーバーロードして、string を用いて独自の C++ オブジェクトを入出力することもできますが、ここでは範囲外です。

### 3.2 `empty()` と `size()`

`empty()` と `size()` は string の状態を確認するときによく使います。

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

たとえば先ほどの `getline` の例で空行を飛ばしたい場合はこう書けます：

```c++
    while(std::getline(std::cin, line)) {
        if(!line.empty()) { // 空文字列ではないことを確認
            std::cout << line << std::endl;
        }
    }
```

`!s.empty()` は `s.size() > 0` と同じ意味ですが、`!s.empty()` のほうが簡潔な書き方だと見なされることが多いです。

また、`size()` の戻り値型は `string::size_type` です。標準ライブラリ実装によって異なりますが、一般的には `size_t`（非負整数）です。つまり `s.size()` は `int` ではありません！

したがって、string を走査するときに次の書き方は誤りです：

```c++
// 錯誤！
for(int i = 0; i < s.size(); ++1) {
    std::cout << s[i];
}
```

正しくは次のように書くべきです：

```c++
// 正確
for(std::string::size_type i = 0; i < s.size(); ++1) {
    std::cout << s[i];
}
```

> 心配しないでください。すぐに `for` を説明します！

`size_t` を使っても基本的には OK です。あるいは面倒なら `auto` でも構いません。

多くのケースでは `int` を使っても比較自体はできるのですが、バグの原因になります。たとえば `s.size() < n` を比較し、`n` が負の `int` の場合、`n` が `size_t` に変換されて巨大な正の数になってしまい、比較が常に `true` になることがあります。

### 3.3 文字のアクセス

string の中の文字をどうアクセスするか？

もっとも簡単な文法は `s[]` と `s.at()` です。

```c++
string s("0123456789");
 
s[2] = 'a'; // s = "01a3456789"
std::cout << s[9]; // 9

s.at(3) = '6'; "01a6456789"
std::cout << s.at(3); // 6
```

`[]` と `at()` は同じに見えますよね？

実は、境界チェックをするかどうかが違います：

```c++
std::cout << s[100]; // わざと範囲外アクセス
// 未定義動作。ゴミが出るかもしれないし、Segmentation Fault かもしれない

std::cout << s.at(100); // わざと範囲外アクセス
// terminate called after throwing an instance of 'std::out_of_range'
//   what():  basic_string::at: __n (which is 100) >= this->size() (which is 10)
// Aborted
```

`at()` は境界チェックをして「このコードはおかしい」と明確に教えてくれます。その上で try-catch を使ったエラー処理もできます。一方、`[]` は範囲外アクセスが未定義動作で、多くの場合 segmentation fault になります。では `[]` はダメなのでしょうか？そうでもありません。境界チェックには性能コストがあるためです（チェック分だけ余計な処理が増える）。`[]` のほうが直感的で高速になりやすいですが、その代わり開発者が境界を慎重に管理する必要があります。

境界チェックは次のようにできます：

```c++
string s("abcd");
size_t index = /* any number */;
if(index >=0 && index < s.size()) {
    std::cout << s[index];
}
```

また、`s.front()` と `s.back()` もよく使います。名前の通り、先頭と末尾の文字を取得します。

```c++
string s("abc");
std::cout << s.front(); // a
std::cout << s.back(); //c
```

もちろん `s[0]` と `s[s.size() - 1]` でも同じことができますが、直感的ではなく、見た目も微妙です。

### 3.4 string の連結

#### 3.4.1 string 同士の連結

`"abc"` と `"defg"` を連結して `"abcdefg"` にする。

最も簡単なのは `+` 演算子（`s1 + s2`）です。もう 1 つの方法は `s1.append(s2)` です。

例を見てみましょう：

```c++
string s1("aaa");
string s2("bbb");

string s3 = s1 + s2; // s3 = "aaabbb"

s1 = s1 + s2; // case 1 沒效率
s1 += s2; // case 2 有效率
s1.append(s2); // case 3 有效率
```

case 1 は `s1 + s2` により新しい string を作り、それを s1 にコピーするので非効率です。case 2 と case 3 はどちらも `s2` を `s1` の末尾に追加する概念なので、実行効率は同じです。case 1 と case 2（あるいは case 3）の差は、前者が `s1` をコピーしてから `s2` もコピーし、新しい string を作るのに対し、後者は `s2` を 1 回コピーして `s1` の末尾に追加するだけだという点です。

つまり case 1 は `s1` もコピーされ、さらに string オブジェクトが 1 つ余計に生成されます。`s1` と `s2` の値を変更したくないなら `string new_str = s1 + s2` がちょうど良いです。

`s1` が変更されてもよいなら `append` や `+=` を使い、不要なコピーを減らしましょう。

上の例の `s1 = s1 + s2` は非常に非効率なので、やめてください！

#### 3.4.2 string とリテラルの連結

string は文字リテラル（character literals）や文字列リテラル（character string literals）とも連結できます。概念としては、リテラル側が自動で型変換されるだけです。

ただし string とリテラルを混ぜて `+` する場合、左右どちらかは string である必要があります。

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

### 3.5 2 つの string の比較

次に `<, <=, ==, !=, >=, >` を紹介します。

`==` は簡単で、2 つの文字列が同じ長さで、内容も同じなら `s1 == s2` は真になります。

大小比較が必要な場合、string は「辞書順」に従います。ルールは 2 つあります：

1. `s1` と `s2` の長さが違っても、先頭から内容が同じなら、長いほうが大きい。
2. 長さや内容が異なる場合、先頭から見て最初に異なる文字の大小で決まる（大きい文字を持つほうが大きい）。

例：

```c++
// 以下皆為 true
"aaa" == "aaa" // 相同
"aaa" != "bbb" // 不相同
"abcd" < "abcde" // 規則 1
"abcd" > "abcc" // 規則 2，d > c
"abcd" > "abcceeeeee" // 規則 2，d > c，即使右邊比較長
```

よく使うのは `==` と `!=` による等価比較です。大小比較は辞書順が必要なときに使えば OK です。

辞書順の活用例：

```c++
std::vector<std::string> words; // "aaa", "abc", "bbb", ... のような文字列が多数入っている

std::sort(words.begin(), words.end(), [](auto& s1, auto& s2){
    return s1 > s2; // 大きい文字→小さい文字の順でソート
});
```

## 4. string の単一文字操作

文字列を扱うとき、1 文字ずつ処理するのも非常によくある操作です。たとえば `"abcdefg"` に `'f'` が含まれるかチェックしたい、各文字を 1 つずつずらして `"bcdefgh"` にしたい、特殊記号が含まれるか調べたい……などなど。どの場合でも、結局は文字列全体を走査する必要があります。

走査といえば `for` です。ここでは特に重要な 2 つの形を紹介します。

1 つ目：範囲を指定して走査する：

```c++
string s("aaabbbccc");
for(size_t i = 3 ; i < s.size() ; i++) {
    std::cout << s[i];
}
// 印出 bbbccc
```

ここでは `i` の開始と終了を自分で決められます。この例では `i` を 3 から始めています。

2 つ目：全要素を走査する。ここでは `for(declaration : expression)` の範囲 for（range-based for）構文を使えます。コロンの左側は要素の宣言で、右側は走査元の文字列です。

```c++
for(char c : s) {
    std::cout << c;
}
// aaabbbccc
```

この場合、各文字が `char c` にコピーされるため、元の `s` は変更できません。

`s` を変更したいなら `char &c` にします。これにより `s` の各文字への参照を扱います：

```c++
for(char &c : s) {
    c += 1;
}
// s = bbbcccddd
```

## 5. string の便利 API

string を使う場面で非常によくあるのは、部分文字列の有無を調べたり、文字列を切り出したり挿入したりする操作です。

参考としてよく使う API をいくつか挙げます：

```c++
s.find(sub_string); // 検索。最初に見つかった位置を返す
s.replace(pos, length, new_string); // 置換。pos から length 分を new_string に置き換える
s.substr(pos, length); // 部分文字列。pos から length 分を切り出す
s.insert(pos, new_string); // 挿入。pos に new_string を挿入する
s.contains(sub_string); // 包含。sub_string を含むか（注意：C++23 以降）
```

簡単な例：

```cpp
std::string http_url = "http://tigercosmos.xyz/about/";

// 位置 4（p の次）に挿入して https://tigercosmos.xyz/about/ を得る
http_url.insert(4, "s"); 
// 位置 0 から長さ 5 の部分文字列を確認
assert(http_url.substr(0, 5) == "https");
// "about" を含むか確認
assert(http_url.contains("about") == true);

// "xyz" 部分文字列の開始位置を探す
size_t pos = http_url.find("xyz");
// pos から 3 文字を "co.jp" に置換して https://tigercosmos.co.jp/about/ を得る
http_url.replace(pos, 3, "co.jp");
```

`std::string` には他にも多くの API があり、各 API にも多くの overload（オーバーロード）があります。平たく言うと「使い方が複数ある」ということです。たとえば `insert` は string を挿入することも、char を挿入することもできます。基本的に「この機能が欲しい」と思ったら、[string の標準ライブラリ](https://en.cppreference.com/w/cpp/string/basic_string)にあるかを確認し、なければ自分で実装しましょう！（既存 API を使いたいときもあれば、自作して遊びたいときもあります！）

## 6. string と数値の型変換

文字列を数値に変換したいですか？

次の関数を使います（すべて `<string>` にあります）：

```c++
std::stoi // int に変換
std::stol // long int に変換
std::stoll // long long int に変換
std::stoul // unsigned long int に変換
std::stoull // unsigned long long int に変換
std::stof // float に変換
std::stod // double に変換
std::stold // long double に変換
```

```c++
int a = std::stoi(string("5"));
double b = std::stod(string("5.5555"));
```

注意！`stou` はありません。これは[謎の迷団](https://stackoverflow.com/questions/8715213/why-is-there-no-stdstou)です。

数値を string に変換したい場合は `std::to_string()` を使えます。

簡単な例：

```c++
int a = 5;
std::string s = "a: " + std::to_string(a);
// s = "a: 5"
```

もし `"123456"` のような数字文字列を読み、各桁を処理したいだけなら、`stoi` は不要です。次の小技が使えます：

```cpp
string s("123456");
for(size_t i = 0; i < s.size(); i++) {
    int a = s[i] - '0';
}
```

string の各文字は `char` で、各文字は [ASCII コード](https://zh.wikipedia.org/wiki/ASCII)です。たとえば「A」は 65、「0」は 48 です。

では string 中の数字をどう取り出すか？上の例で `s[2]` は `'3'` です。`'3'` の ASCII（51）から `'0'` の ASCII（48）を引けば、差は 3 になります。

この例で言いたいことは、「文字列全体を数値に変換する必要がない場合も多い」ということです。1 文字ずつ処理することで、新しいアルゴリズムの発想が出てくることもあります。

## 結論

この記事では `std::string` の基本的な使い方を詳しく紹介しました。分からない点があれば string の API を調べても良いですが、より一般的で簡単な方法は Google でも Stack Overflow でもなく、万能の ChatGPT に聞くことです！

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

ただし注意！ChatGPT が適当なことを言っていないか、自分で必ず確認しましょう >.O！

> p.s. ちなみに、ChatGPT の例は実はかなり非効率です。`str[i] = '-'` で十分じゃん 😂
