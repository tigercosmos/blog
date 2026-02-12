---
title: Run SPEC ACCEL OpenCL benchmarks on AMD MI50
date: 2020-08-14 00:08:00
tags: [spec accel, note, amd mi50, opencl, benchmark]
des: "本文介紹如何拿 SPEC ACCEL 中的 OpenCL Benchmarks 來測試 AMD MI50，並針對測試的數據簡單做分析，並與 SPEC 官網上提供的 GeForce GTX 1050 跑出來的數據做比較。"
lang: zh
translation_key: spec-accel-opencl-amd-mi50
---
## 簡介

[SPEC ACCEL](https://www.spec.org/accel/) 是 The Standard Performance Evaluation Corporation (SPEC) 組織建立的加速器 (Accelerator) 效能測試指標 (Benchmark)。支援含括 OpenACC、OpenMP、OpenCL，這三個皆為可以讓程式碼分攤工作給 GPU 跑的框架。因為具有極大公信力，許多研究都會以這套指標當作評量測試。

SPEC ACCEL 有免費公開給研究機構使用，不像 SPEC CPU 要花一大筆錢 (雖然我直接拿隔壁實驗室的 😂)，總之免費公開的真香。之前剛好有一台配有 AMD MI50 的裝置要評測效能，於是就拿 SPEC ACCEL 來跑 OpenCL 的部分，看看這張顯卡效能如何。

![MI50](https://user-images.githubusercontent.com/18013815/90196045-88372080-ddfd-11ea-99d1-aa6b24a70ca6.png)


## 安裝執行

安裝過程參照官網「[Install Guide Unix](https://www.spec.org/accel/docs/install-guide-unix.html)」就很清楚了，過程不會遇到甚麼錯誤，就不特別重複過程，請參照文件。

因為是 AMD 的 GPU，所以要裝它的驅動程式 [ROCM](https://github.com/RadeonOpenCompute/ROCm)，裝好之後可以用 `radeontop` 工具檢查看看能不能連到 GPU。

都弄好之後，執行之前要去修改 `.cfg` 裡面的 OpenCL 函式庫路徑，沒意外會在 ROCM 驅動裡面。

接著就可以跑了：

```shell
runspec --config=my.cfg --platform AMD --device GPU opencl
```

跑出來的結果可以透過 SPEC 提供工具轉換成網頁格式，也可以選擇上傳官網供別人參考。

## 測試結果

### SPEC ACCEL OpenCL Benchmarks

SPEC ACCEL 關於 OpenCL 的 Bencmarks 如以下：

![image](https://user-images.githubusercontent.com/18013815/90194424-de09c980-ddf9-11ea-8550-79ac0ef4c458.png)

(source: [SPEC ACCEL: A Standard Application Suite for Measuring Hardware Accelerator Performance](https://link.springer.com/chapter/10.1007/978-3-319-17248-4_3))

### AMD MI50 測試結果

跑出來的結果長這樣，有些測試跑的時候會出錯，但我沒去檢查原因，就留白
![image](https://user-images.githubusercontent.com/18013815/90194468-0265a600-ddfa-11ea-91a0-504888e3e807.png)

化成圖表長這樣：

![image](https://user-images.githubusercontent.com/18013815/90194588-39d45280-ddfa-11ea-9bbf-2a8cbab75a57.png)

### GPU 使用佔比

這邊是別的論文用 NVIDIA Tesla K20 跑出來的 GPU 使用佔比的結果：

![image](https://user-images.githubusercontent.com/18013815/90194641-4d7fb900-ddfa-11ea-9734-f70b0da84ef7.png)

(source: [SPEC ACCEL: A Standard Application Suite for Measuring Hardware Accelerator Performance](https://link.springer.com/chapter/10.1007/978-3-319-17248-4_3))

因為我不知道要怎麼去測量 GPU 使用比例，但是 OpenCL 的 Code 都一樣的情況下，GPU 佔比原則上不會差異太大，所以可以直接拿來當參考。

### 結果分析

我將 AMD MI50 跑出來的結果框在上圖中，其中標註綠色的部分是在 AMD MI50 表現比較優的，紅色則相反。

從上圖可以發現，GPU 使用時間比例跟轉移時間這兩個因子跟效能並沒直接關聯，其實頗令人意外，因為 GPU 使用率高感覺就會因為 GPU 性能比較優而比較快。轉移時間也一樣，感覺花比較多時間在轉移的效能會比較差，但實際上好像也沒關聯。

比較特別是 `kmeans` 的 Base Radio 比 1 還小，意思是這結果比官方標竿機器還慢，但是官方標竿機器原則上應該會比較爛，況且可以看到大部分測試項目都是遠遠超過 1 的。

`cutcp` 在測試的說明頁就有說，它是一個計算量相關的測試，且會直接因為加速器效能有顯著影響，所以在 MI50 表現好也就不易外。

不過其他測試項，不管是特別優秀或特別差的，很多看測試說明也無法直接了解為甚麼會有這結果。

針對比較優的和比較差的 Benchmark，要去理解他們的效能結果可能必須去看 OpenCL Code 是怎麼寫的，以及 MI50 架構設計，但牽扯太複雜了，我沒有進一步去研究。

### MI50 VS GeForce GTX 1050

接著要跟對照組 GeForce GTX 1050 比較。

下圖是跑分網站對兩張顯卡一般性的評分，可以看到 1050 大約只有 MI50 的 87%。
![image](https://user-images.githubusercontent.com/18013815/90195134-4efdb100-ddfb-11ea-828e-0fe5746154e8.png)


然後我參考 SPEC ACCEL 官網上別人貢獻的 GeForce GTX 1050 數據，拿來做比較：

![image](https://user-images.githubusercontent.com/18013815/90195423-e95df480-ddfb-11ea-8885-27b2d738d95b.png)

理論上 MI50 應該都要快個 13% 左右，不過因為架構設計會有所差異也滿合理的。

其中 `nw` 和 `ge` MI50 被 GTX 1050 超越不少，是一個滿特別的結果。其餘 MI50 多多少少都有領先，有符合跑分網站的預期。

至於 MI50 和 GTX 1050 之間的數值差異，則可以用來分析兩者架構在不同情境下的效能優劣。

## 結論

本文介紹如何拿 SPEC ACCEL 中的 OpenCL Benchmarks 來測試 AMD MI50，並針對測試的數據簡單做分析，並與 SPEC 官網上提供的 GeForce GTX 1050 跑出來的數據做比較。
