---
title: "Shopify Pricing 踩雷紀錄：Managed Pricing & Manual Pricing 那邊設定？"
date: 2025-09-17 18:00:00
tags: [shopify, pricing, billing api, managed pricing, manual pricing]
des: "簡單紀錄 Shopify Pricing 踩雷過程，Managed Pricing & Manual Pricing 藏在非常隱密的地方。"
---

如果你要幫你的 Shopify App 接上[付費機制](https://shopify.dev/docs/apps/launch/billing)，那你就會需要選擇 Managed Pricing 或 Manual Pricing，前者是用 Shopify 的公版，後者是用自己的網頁畫面串上 Billing API。

如果你要用 Billing API，必須明確設定 App 要使用 Manual Pricing，問題是官方文件根本沒仔細寫要去哪裡設定，花了很久才找到怎麼樣設定。

流程如下：

1. 進入 App Overview 畫面，選「Manage submissions」
    ![](/img/shopify/pricing-method/app-overview.png)
2. 進入提交審查的畫面（App Store Review），編輯 App listing
    ![](/img/shopify/pricing-method/submit-review.png)
3. 找到 Pricing details 的位置點「Manage」進去
    ![](/img/shopify/pricing-method/app-listing.png)
4. 進到 Pricing 畫面，右上角有個設定按鈕
    ![](/img/shopify/pricing-method/pricing.png)
5. 選擇要 Manged Pricing 或 Manual Pricing
    ![](/img/shopify/pricing-method/final.png)

設定好之後就能正常使用 Shopify Billing API 了！