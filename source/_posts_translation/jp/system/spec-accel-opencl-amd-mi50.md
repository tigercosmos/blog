---
title: "AMD MI50 で SPEC ACCEL の OpenCL ベンチマークを実行する"
date: 2020-08-14 00:08:00
tags: [spec accel, note, amd mi50, opencl, benchmark]
des: "本記事では SPEC ACCEL の OpenCL ベンチマークを AMD MI50 で実行する方法を紹介し、結果を簡単に分析しつつ、SPEC 公式サイトで公開されている GeForce GTX 1050 の結果と比較します。"
lang: jp
translation_key: spec-accel-opencl-amd-mi50
---

## イントロダクション

[SPEC ACCEL](https://www.spec.org/accel/) は The Standard Performance Evaluation Corporation（SPEC）が策定した、アクセラレータ（Accelerator）向けの性能評価指標（Benchmark）です。OpenACC、OpenMP、OpenCL をサポートしており、いずれも GPU に処理をオフロードできるフレームワークです。公信力が高いため、多くの研究で評価指標として利用されています。

SPEC ACCEL は研究機関向けに無償で公開されています（SPEC CPU は大金が必要ですが……私は隣の研究室のものを借りました 😂）。とにかく無料は正義です。ちょうど AMD MI50 を搭載した装置の性能評価が必要だったので、SPEC ACCEL の OpenCL 部分を実行して、この GPU の性能を見てみました。

![MI50](https://user-images.githubusercontent.com/18013815/90196045-88372080-ddfd-11ea-99d1-aa6b24a70ca6.png)

## インストールと実行

インストール手順は公式の「[Install Guide Unix](https://www.spec.org/accel/docs/install-guide-unix.html)」が分かりやすいです。基本的に大きなエラーは起きないはずなので、ここでは繰り返しません。ドキュメントを参照してください。

AMD GPU なので、ドライバとして [ROCM](https://github.com/RadeonOpenCompute/ROCm) をインストールします。インストール後、`radeontop` で GPU に接続できているか確認できます。

準備が整ったら、実行前に `.cfg` の OpenCL ライブラリパスを修正します。通常は ROCm のドライバ内にあります。

その後、次のように実行します：

```shell
runspec --config=my.cfg --platform AMD --device GPU opencl
```

結果は SPEC が提供するツールで Web 形式に変換できます。また、公式サイトにアップロードして他の人が参照できるようにすることも可能です。

## 測定結果

### SPEC ACCEL の OpenCL ベンチマーク

SPEC ACCEL の OpenCL ベンチマークは次の通りです：

![image](https://user-images.githubusercontent.com/18013815/90194424-de09c980-ddf9-11ea-8550-79ac0ef4c458.png)

(source: [SPEC ACCEL: A Standard Application Suite for Measuring Hardware Accelerator Performance](https://link.springer.com/chapter/10.1007/978-3-319-17248-4_3))

### AMD MI50 の結果

実行結果は次のようになります。いくつかのテストは実行中にエラーになりましたが、原因は追っていないので空欄のままです。

![image](https://user-images.githubusercontent.com/18013815/90194468-0265a600-ddfa-11ea-91a0-504888e3e807.png)

グラフにすると次の通りです：

![image](https://user-images.githubusercontent.com/18013815/90194588-39d45280-ddfa-11ea-9bbf-2a8cbab75a57.png)

### GPU 使用割合

次は別論文で NVIDIA Tesla K20 を使って計測された GPU 使用割合の結果です：

![image](https://user-images.githubusercontent.com/18013815/90194641-4d7fb900-ddfa-11ea-9734-f70b0da84ef7.png)

(source: [SPEC ACCEL: A Standard Application Suite for Measuring Hardware Accelerator Performance](https://link.springer.com/chapter/10.1007/978-3-319-17248-4_3))

私は GPU 使用割合をどう計測すべきか分かりませんでした。しかし OpenCL のコードは同じである以上、GPU 使用割合は大きくは変わらないはずなので、参考としてそのまま使います。

### 結果の分析

上図に AMD MI50 の結果を重ねました。緑は AMD MI50 が優れている部分、赤は逆の部分です。

上図から、GPU 使用時間割合と転送時間（transfer time）という 2 つの要因は、性能と直接相関していないことが分かります。これはやや意外でした。直感的には、GPU 使用率が高いほど GPU 性能が効いて速くなりそうですし、転送時間が長いほど性能が悪くなりそうですが、結果を見る限り明確な相関はありません。

特に `kmeans` は Base Ratio が 1 より小さく、公式の基準機より遅いことを意味します。しかし、公式の基準機は原則としてもっと性能が低いはずで、また大半のテスト項目は 1 を大きく上回っています。

`cutcp` は説明ページで「計算量に強く依存し、アクセラレータ性能の影響が大きい」と書かれているため、MI50 が強いのも自然です。

ただし、他のテスト項目については、特に良い／特に悪い理由を、説明だけから直感的に理解するのは難しいものが多いです。

性能差を理解するには、OpenCL コードの実装と MI50 のアーキテクチャ設計を掘る必要がありますが、複雑すぎるのでここでは深掘りしませんでした。

### MI50 と GeForce GTX 1050 の比較

次に、対照として GeForce GTX 1050 と比較します。

次の図はスコアサイトでの一般的な評価で、GTX 1050 は MI50 の約 87% 程度であることが分かります。

![image](https://user-images.githubusercontent.com/18013815/90195134-4efdb100-ddfb-11ea-828e-0fe5746154e8.png)

さらに、SPEC ACCEL 公式サイトで他者が投稿した GeForce GTX 1050 の結果を参照し、比較しました：

![image](https://user-images.githubusercontent.com/18013815/90195423-e95df480-ddfb-11ea-8885-27b2d738d95b.png)

理論上は MI50 が 13% 程度速いはずですが、アーキテクチャ設計の差によって結果が変わるのは十分あり得ます。

特に `nw` と `ge` は、MI50 が GTX 1050 にかなり負けており、興味深い結果です。その他は MI50 が概ね優位で、スコアサイトの予想とも整合しています。

MI50 と GTX 1050 の数値差は、異なる状況で両者のアーキテクチャがどのように性能を発揮するかを分析する材料になります。

## 結論

本記事では SPEC ACCEL の OpenCL ベンチマークを AMD MI50 で実行する方法を紹介し、結果を簡単に分析しつつ、SPEC 公式サイトで公開されている GeForce GTX 1050 の結果と比較しました。
