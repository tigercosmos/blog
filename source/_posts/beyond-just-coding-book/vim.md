---
title: "每個工程師都應該懂的 Vim 基本技巧：《程式設計原來不只有寫 CODE！》之〈Vim〉章節"
date: 2024-12-26 01:00:00
tags: [程式設計原來不只有寫 CODE！, 資工基本素養, vim, 編輯器]
des: "本文摘自《程式設計原來不只有寫 CODE！》之〈Vim〉章節，介紹 Vim 編輯器的基本使用技巧，讓你在 Vim 操作環境下成功生存，並提升程式碼修改的效率。"
---

<style>
    .button-like {
        border: 1px solid rgb(0, 0, 0);
        border-radius:5px;
        font-size: 0.8em;
        padding: 0 10px;
    }
</style>


<blockquote style="border: 5px solid #ee9c6b;border-radius:30px;">

本文摘自[**《程式設計原來不只有寫 CODE！銜接學校與職場的五堂軟體開發實習課》**](/books/beyond-just-coding-book.html)一書的〈Vim〉章節（原書章節 1.2.3）。本文為精簡版，敬請[支持原書](/books/beyond-just-coding-book.html)來觀看完整內容以及精美排版。

</blockquote>

## 前言

根據 Stack Overflow 所得到的統計（圖 1），Vim 獨佔*終端機介面編輯器*的龍頭寶座。Vim 是已逝荷蘭工程師 Bram Moolenaar 在1991 年改良 Vi 編輯器後發布，這也是 Vim 其名由來，亦即改良版的 Vi（**V**i **IM**proved），自此逐漸演變成開發者中熱門的編輯器之一。有意思的是，電腦科學發展史上曾經上演[編輯器大戰（Editor War）](https://en.wikipedia.org/wiki/Editor_war) ，從 1980 年代開始人們爭論著 Vi 和 Emacs 孰優孰劣，兩個編輯器在當時都是開發者熱門選擇，有意思的是兩者正好都是 1976 年正式發布，自是少不了瑜亮情節，時至今日雖然許多系統預設都會將兩個編輯器都裝上，但圖 1 的結果似乎已經宣告著 Vi 陣營的勝利。今日，當你在終端機輸入 `vi` 命令的時候，預設都會直接執行 Vim 了（等同輸入 `vim` 命令）。

> Vi 編輯器 1976 年由 Bill Joy所發明，後來誕生許多衍生版，Vim 就是其中之一。
> 
> Emacs 最初由 David A. Moon 和 Guy L. Steele Jr. 於 1976 年所發明，後續也有多個衍生版，其中最知名的是Richard Stallman 開發的 GNU Emacs。

<img width="80%" alt="Stack Overflow 2023 問卷調查 種不同 IDE 的使用比例" src="https://github.com/user-attachments/assets/cbde77cd-dc28-45cd-bca2-8af457acb6e7" />

（圖 1： [Stack Overflow 2023 問卷](https://survey.stackoverflow.co/2023/)調查不同 IDE 的使用比例）

學習 Vim 不像使用 VSCode 直觀，VSCode 我相信沒用過的人三分鐘也能理解要怎樣操作，至於 Vim 嘛，光是理解大概就要三個小時，要能上手可能要花個三個禮拜。你可能會很意外（或者早已略有所聞），原來 Vim 並不是那麼好入門，甚至可以說是入門即放棄。

圖 2 展示了 VSCode 和 Vim 的學習曲線：VSCode 非常容易入門，到了中期你可能為了要增加開發效率必須學會很多快捷鍵、設定檔、擴充套件等等，讓學習難度增加，但之後都熟練了之後就會開始用得很順了；反之，Vim 是一開始就非常難，光是不能用滑鼠就非常違反現代電腦操作哲學，你需要花大把精力去熟習Vim的操作邏輯，但慢慢的你會發現 Vim 越用越快，中間你可能會想要開始做一些 Vim 的設定，但那些相對於初期學習的痛苦都不算什麼，最後你甚至開始感受到不需要使用滑鼠高效率開發的美妙之處。

<img src="https://github.com/user-attachments/assets/6905ca27-cf7d-4bd3-bcff-64dd6d9ff47e" width="70%" alt="VSCode 與 Vim 的學習曲線"  />


（圖 2： VSCode 與 Vim 的學習曲線）

以筆者來說，用 Vim 的最大時機是使用 SSH 連線到 Linux 伺服器，這時你只有終端機可以改程式碼，雖然 Nano 也堪用，但還是不太方便，此時我就會使用 Vim 來做文字編輯，特別是修改 Python 程式碼、Bash 腳本、或是一些專案設定檔的時候特別實用。筆者大學的時候曾經以 Vim 作為主要 IDE 來做開發，但後來還是覺得受不了，老老實實地回歸 VSCode 懷抱。所以除了一些頂尖高手以外，筆者並不建議把 Vim 當作開發主力，但我們仍要知道基本操作，日常處理小事情的時候相當方便。

## 安裝Vim

首先我們得要有 Vim 可以操作，以 Ubuntu 來說，我們可以直接從 APT 安裝：

```
sudo apt install vim
```

macOS 的話也很簡單：

```
brew install vim
```

或是如果想要最新版本的話，我們可以從原始碼來安裝 ：

```
git clone https://github.com/vim/vim.git
cd vim/src
make
sudo make install
```

[Vim 官網](https://www.vim.org/download.php)上還提供很多種下載方法，其他的平台可以直接上其官網參考。

## Vim之生存技巧

Vim 令人聞風喪膽的其中一點是，一旦進入該程式，你甚至不知道怎麼離開！在這小節中你會學到如何開啟 Vim 後還能退出，諸如此類簡單的「生存方法」。

首先我們得開啟 Vim ，請直接在終端機輸入 `vim` 命令，就會直接進入 Vim 的畫面了，如圖 3 所示，作者還很貼心，跟你說要怎麼離開（輸入 `:q<Enter>`），確實光是怎麼離開就很困惑人了。筆者當初一開始「不小心」開啟此軟體時，光是要怎麼離開就夠困擾的了，因為大多數情況我們可以輸入 `Ctrl+C` 使程式觸發中斷（interrupt）而離開程式，但是 Vim 不讓你這樣做。當然，沒辦法這樣做是有原因的，讓我們接著學習如何進行基本操作。

<img width="800px" alt="Vim 初始畫面" src="https://github.com/user-attachments/assets/f9cf1763-8e0b-47b1-ae12-dd7897a0566a" />

（圖 3：Vim 初始畫面）


首先我們要了解到 Vim 有分**一般模式**（normal mode）和**插入模式**（insert mode）、**記錄模式**（record mode）以及其他模式。前兩個模式最重要，也是我們生存的第一步。

當我們進入 Vim 之後，一開始會在*一般模式*，在一般模式中任何按鍵都可以是功能鍵（不管是英文按鍵、數字按鍵等），我們可以用功能鍵進行一些操作，例如下指令進行存檔、搜尋字串、瀏覽檔案、複製和刪改等等；而*插入模式*則是讓你可以進行文字輸入，在此模式中你按的任何按鍵都會被視為文字插進文本中。

現在請在圖 3 的介面中以鍵盤輸入「`:q`」 （有正確輸入的話，畫面左下方會顯示你輸入的文字）並按下 <span class="button-like">Enter</span>，你就會離開 Vim 的畫面了。

> `:q` 即為 quit 的意思，有時候指令是英文字母的縮寫，例如 `:d`（delete）、`:w`（write），有時候則沒有特別原因只能硬背，例如 `:x`（exit if modified, otherwise just quit）。

在一般模式中，下達指令的方式都是「:」或「/」開頭，例如剛剛的「`:q`」。接下來為了方便書寫和閱讀，本書在一般模式下指令時，會以「`:q<Enter>`」的方式進行標註，其中「`<Enter>`」代表要按下 <span class="button-like">Enter</span>。如果要你按下 <span class="button-like">Ctrl</span> + <span class="button-like">λ</span>，將會直接標註 <span class="button-like">C-λ</span>，如果要連續按幾個按鍵，例如按兩下 <span class="button-like">d</span>，則會直接標註成 <span class="button-like">dd</span>，此外 Vim 在一般模式中大小寫的英文字母會是不同功能，大寫的 J 按鍵會直接標註成 <span class="button-like">J</span> 反之小寫的 j 會標示成 <span class="button-like">j</span>（用 <span class="button-like">CapsLock</span> 來進行大小寫切換），所以請留意字母的大小寫之分。

最後， Vim 的游標位置會以「|」來表示，注意：根據不同平台，一般模式的游標可能是細的或粗的（包住字元並反白），以 `ti|ger` 為範例，當游標是粗的情況下，字母 g 會被粗的游標框住並反白，本書中一般模式將一律以細的游標進行講解，之後不會特別再針對粗游標做說明。

### 開啟範例

現在請打開「ch1/vim-practice/」資料夾，以 Vim 開啟「[random_conf.conf](https://github.com/tigercosmos/beyond-just-coding-book/blob/master/ch1/vim-practice/randome_conf.conf)」檔案。

> 網路讀者請先到[本書 GitHub 專案](https://github.com/tigercosmos/beyond-just-coding-book)下載範例程式碼：
> `git clone https://github.com/tigercosmos/beyond-just-coding-book.git`

```
$ vim random_conf.conf
```

現在你會看到「random_conf.conf」檔案的內容，為了方便讀者閱讀，這邊將該文件前面的部分印出來，同時記住你目前是處在一般模式。

```
# Random Configuration File

[General]
app_name = MyApp
version = 2.0
debug_mode = false

[Database]
db_host = localhost # 第 9 行
db_port = 3306 # 第 10 行
db_name = my_database
db_user = admin
db_password = my_secure_password

[Logging]
log_level = info
log_file = /var/log/myapp.log

[Server]
server_host = 127.0.0.1
server_port = 8080
max_connections = 100
timeout = 60
```

### 操控方向

還記得我們在先前學過盲打嗎？我們的右手食指一般是放 j 上面，在 Vim 中我們操控游標（cursor）使用右手按 <span class="button-like">h</span>、<span class="button-like">j</span>、<span class="button-like">k</span>、<span class="button-like">l</span>（正好在同一排，之後我會以英文方向按鍵來代稱），功能分別等於方向鍵 <span class="button-like">←</span>、<span class="button-like">↓</span>、<span class="button-like">↑</span>、<span class="button-like">→</span>，你可以選擇用英文方向按鍵或方向鍵來操控游標，但一般來說會使用前者，因為手不需要位移，注意，我提到「位移」，這也是 Vim 的精髓所在，手大部分時間都固定在同個位置，不需要移動手掌去按方向鍵，當然也不需要移動右手去操控滑鼠，這是為什麼使用 Vim 進行編輯會比較快（前提是你夠熟練）。

### 插入模式進行文字編輯

現在我想要把第 10 行的數字從 `3306` 改成 `3307`。我們先在一般模式使用英文方向按鍵移動到第 10 行數字的位置，然後游標移動到 `3306` 的「6」後方，接著按 <span class="button-like">i</span> 進入插入模式，插入模式的操作跟你平常如何使用記事本或 Word 一樣，所以我們現在可以用 <span class="button-like">Backspace</span> 來刪掉「6」並且插入「7」，就完成更動，最後只需要按 <span class="button-like">ESC</span> 來離開插入模式即可。

### 存檔並離開

現在我們回到一般模式了，我們可以輸入 `:w<Enter>` 來進行存檔，然後再輸入 `:q<Enter>` 來離開Vim。不過要打兩次有點麻煩，所以 Vim 可以讓你直接打 `:wq<Enter>` 來儲存後離開，是不是方便多了！注意 `:wq` 的順序不能反，畢竟邏輯是要能先存檔，然後才能離開。

恭喜你，到目前為止就是 Vim 最最最基本的操作了，你已經能夠用 Vim 做簡單的文本修改了，如果是只要改改設定檔的數字，相信已經很夠用了。但是 Vim 的功能遠遠不只如此，當我們想要編輯的程式碼更多更複雜的時候，只靠上面的幾個功能顯得不太夠力。

## Vim 進階操作技巧

接著讓我們來學習稍微進階的 Vim 使用技巧吧，礙於篇幅這邊會以條列式進行說明，讀者可以跟著進行操作，實際看功能或指令的效果。

### 進入插入模式

- <span class="button-like">i</span> ：在游標位置處進入插入模式。
- <span class="button-like">a</span> ：在游標位置後進入插入模式，例如原本一般模式游標在 `ti|ger`，進入插入模式後游標會變成 `tig|er`。
- <span class="button-like">o</span> （小o）：在游標位置下方插入空白一行，並進入插入模式。
- <span class="button-like">O</span> （大O）：在游標位置上方插入空白一行，並進入插入模式。

記住都是用 <span class="button-like">ESC</span> 來離開插入模式。

### 一般模式編輯

- <span class="button-like">x</span>：可以刪除指標位置的字元，例如 `ti|ger` 就會變成 `tier`。
- <span class="button-like">r</span>：可以更改游標位置的字元，例如 `ti|ger`，此時我們按一下<span class="button-like">r</span>，接著按「a」，原本「g」就會被替換成「a」，該字串變成 `tiaer`。
- <span class="button-like">dd</span>：可以刪除游標位置的那一行。要注意的是，Vim 的一般模式中，凡是刪除的文字的指令（<span class="button-like">dd</span>、<span class="button-like">x</span> 等都是）都會自動進入剪貼簿，所以該指令的語意等同「刪除且複製到剪貼簿」。
- <span class="button-like">p</span>：可以把複製的內容貼上，例如剛剛 <span class="button-like">dd</span> 的那一行。
- <span class="button-like">yy</span>：可以複製游標位置的一行，複製完可以用 <span class="button-like">p</span> 貼上。

### 游標移動

-  <span class="button-like">0</span> ：移動到游標該行的最前面。
-  <span class="button-like">^</span> ：移動到游標該行非空白的第一個字元
-  <span class="button-like">$</span> ：移動到游標該行的最後一個字元。
-  <span class="button-like">w</span> 、 <span class="button-like">W</span> ：移動到游標所在位置的下一個單詞（word）的開頭，其中大W移動時包含標點符號。舉例來說，`f|or(int i = 0; i < 10; i++)` 這段文字，按下小 <span class="button-like">w</span> 會到 `for|(int i = 0; i < 10; i++)`，而按大 <span class="button-like">W</span> 則會到 `for(int |i = 0; i < 10; i++)`。
-  <span class="button-like">e</span> 、 <span class="button-like">E</span> ：移動到游標所在單詞的最後，其中大 <span class="button-like">E</span> 移動時包含標點符號。舉例來說，`f|or(int i = 0; i < 10; i++)`，按小 <span class="button-like">e</span> 會到 `fo|r(int i = 0; i < 10; i++)`，按大 <span class="button-like">E</span> 會到 `for(in|t i = 0; i < 10; i++)`。

### 回復＆重做

- <span class="button-like">u</span>：回復。等於大多數編輯軟體的 <span class="button-like">Ctrl</span>+ <span class="button-like">Z</span>。
- <span class="button-like">C-r</span>：重做。等於大多數編輯軟體的 <span class="button-like">Ctrl</span> + <span class="button-like">Shift</span> + <span class="button-like">Z</span>。

### 搜尋＆取代

- `/搜尋字串`：例如我們搜尋用 `/host` 來搜尋「random_conf.conf」，第一個會出現第 9 行 `db_|host`，使用 <span class="button-like">n</span> 可以跳到下一個符合的字串，即第 9 行的 `local|host`，再按一次 <span class="button-like">n</span> 會跳到第20行的 `server_|host`。此外，可以使用 <span class="button-like">N</span>（也就是 <span class="button-like">Shift</span> + <span class="button-like">n</span>）回去上一個搜尋到的結果。
- `:%s/搜尋字串/取代字串/參數`：這串指令是搜尋並取代字串，`%s` 代表搜尋整個文件，舉例來說我們可以 `:%s/db_/database_/g`，意思是把 `db_` 全部取代成 `database_`，`g` 的意思是全部符合條件的都要取代，沒有 `g` 的話，每一行只有一第一個符合的字串會被取代（一行可能有多個字串符合條件）。

    執行前文件本來是：
    ```
    db_host = localhost
    db_port = 3306
    db_name = my_database
    db_user = admin
    db_password = my_secure_password
    ```
    執行後變成：
    ```
    database_host = localhost
    database_port = 3306
    database_name = my_database
    database_user = admin
    database_password = my_secure_password
    ```
    我們還可以這樣下指令 `:5,10s/cat/dog/gc`，意思是搜尋第 5 到第 10 行，取代所有 cat 變成 dog，`c`的意思是每個要取代的字串都要執行確認才取代。

## 小節

頭昏眼花了嗎？以上是筆者挑選出比較實用的一些進階語法與功能，這還只是 Vim 的功能的冰山一角，光是要記住上面介紹的功能就要花一些時間和練習了。

> ✨ 小練習
> - 在「random_conf.conf」文件中，使用搜尋找到「127.0.0.1」，並取代成「192.168.0.1」，過程中不能進入插入模式。
> - 呈上，在該行的上方，以一般模式向上插入空白行並進入插入模式，輸入「server_name = myserver」。
> - 呈上，回到一般模式，怎樣把「192.168.0.1」改成「192.168.3.1」，當游標位於「0」的位置時，你應該只需要按兩次按鍵。

<div style="display: flex;justify-content: center;align-items: center;">
<blockquote style="border: 5px solid #ee9c6b;border-radius:30px; width:80%;">

欲閱讀更多內容，請參閱[《程式設計原來不只有寫 CODE！銜接學校與職場的五堂軟體開發實習課》](/books/beyond-just-coding-book.html)介紹頁面。
<a href="/books/beyond-just-coding-book.html" style="display: flex;justify-content: center;align-items: center;"><img src="https://raw.githubusercontent.com/tigercosmos/beyond-just-coding-book/refs/heads/master/book_picture.jpg" width="100%" alt="「程式設計原來不只有寫 CODE！銜接學校與職場的五堂軟體開發實習課」書籍封面" style="max-width:350px"></a>

</blockquote>
</div>