---
title: 如何利用 PcapPlusPlus 以混雜模式（Promiscuous Mode）進行網路封包分析
date: 2023-05-14 18:20:08
tags: [network, promiscuous mode, network analysis, pcap, PcapPlusPlus]
des: "本文介紹如何用 PcapPlusPlus 以混雜模式（Promiscuous Mode）進行網路封包分析，介紹如何發送、接受、分析網路封包，最後並給一個 ARP 的範例程式。"
---


## 簡介

混雜模式（Promiscuous Mode）是指網路介面卡（NIC）接收和傳輸網路封包的模式。在普通的模式下，網卡只會接收發送給自己或者廣播給所有網卡的封包，而不會接收其他的封包。

而在混雜模式下，網卡可以接收到發送給任何一個網卡的封包，即使這些封包不是發送給自己的。這種模式可以使網卡監聽整個網路，並截取所有傳輸的數據，包括其他主機之間的通信。因此，混雜模式常用於網路故障排除、網路分析、安全監控等方面，但也可能被用於非法監聽和攻擊。

常見的網路封包分析工具，像是 tcpdump 或 Wireshark 背後就是基於混雜模式來進行分析。

在 Linux 上要進行網路封包分析，我們可以使用 `socket` 設定為 `SOCK_RAW` 且監聽所有協定 `ETH_P_ALL`。

並且我們要將 Socket 使用 `setsockopt` 設定為 `PACKET_MR_PROMISC`，也就是混雜模式。（可參閱 [packet(7)](https://man7.org/linux/man-pages/man7/packet.7.html)）

```c
int sock = socket(PF_PACKET, SOCK_RAW, htons(ETH_P_ALL));

struct packet_mreq mr;
memset(&mr, 0, sizeof(mr));
mr.mr_ifindex = if_nametoindex(iface);
mr.mr_type = PACKET_MR_PROMISC;
setsockopt(sock, SOL_PACKET, PACKET_ADD_MEMBERSHIP, &mr, sizeof(mr));
```

另外在使用混雜模式之前，也要記得設定一下網卡：

```sh
sudo ip link set <interface_name> promisc on
```

這樣才能監聽所有經流網卡的網路封包，不然網卡可能就會直接幫你過濾掉那些 MAC 位置不是給你的網路封包。

![COVER IMAGE](https://github.com/tigercosmos/blog/assets/18013815/3100c948-f555-43ad-b58f-71047a0d3a3c)

## PcapPlusPlus 介紹

直接用 System Call 硬幹挺累人的，更高階一點可以使用 libpcap，「pcap」意思就是 Packet Capture（封包捕捉），這原先是由 tcpdump 開發者所開發的 C 函示庫，做了一些封裝方便使用。而 PcapPlusPlus 則是一個基於 libpcap 函示庫的高層次封装，提供了更簡單易用的接口和更多的功能。PcapPlusPlus 支持 Windows、Linux、macOS 跨平台開發，可以輕鬆實現網路數據包的捕捉、解析、修改等操作，同時還支持各種網路協議的分析和解碼，這代表我們不用自幹亂七八糟的協議，可以省很多時間。

以下介紹如何使用 PcapPlusPlus（文中以 PCPP 簡稱），大家也可以多參考其 Github，裡面有完整的[範例程式](https://github.com/seladb/PcapPlusPlus/tree/master/Examples)。

## 安裝 PcapPlusPlus

PcapPlusPlus 官方的[安裝教學](https://pcapplusplus.github.io/docs/install/linux)寫得有點複雜，實際上直接用 CMake 就可以輕鬆安裝了。

```sh
# pre-requirement
sudo apt-get install libpcap-dev

git clone https://github.com/seladb/PcapPlusPlus.git
cd PcapPlusPlus/

mkdir build; cd build; cmake ..; make -j8; sudo make install
```

安裝完之後 Header 和 Library 預設路徑分別是 `/usr/local/include/pcapplusplus/` 和 `/usr/local/lib/`。

## 本文範例程式

安裝完 PcapPlusPlus 後，請下載[本文範例程式](https://github.com/tigercosmos/promiscuous-mode-tutorial)：

```
git clone https://github.com/tigercosmos/promiscuous-mode-tutorial

cd promiscuous-mode-tutorial

mkdir build; cd build; cmake ..; make 
```


## CMake 設置

接下來我們介紹如何設置 CMake

```makefile
cmake_minimum_required(VERSION 3.1)
project(MyPcapPlusPlusProject)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)

# Find PcapPlusPlus library
find_package(PcapPlusPlus REQUIRED)

# Add your project source files
add_executable(main main.cpp)

# Link against PcapPlusPlus library
target_link_libraries(main PcapPlusPlus::Packet++ PcapPlusPlus::Pcap++ PcapPlusPlus::Common++)
```

要引入的函示庫有三個 `Packet++`、`Pcap++`、`Common++`，前面都加上 `PcapPlusPlus::` 比較保險，有時候 CMake 會找不到。

## PcapPlusPlus Hello World

接下來就可以來進行 Hello World 了。請參考 [promiscuous-mode-tutorial/hello_world.cpp](https://github.com/tigercosmos/promiscuous-mode-tutorial/blob/master/hello_world.cpp)。

```cpp
#include <iostream>
#include <pcapplusplus/PcapLiveDevice.h>
#include <pcapplusplus/PcapLiveDeviceList.h>

int main(int argc, char *argv[])
{
    // 可以用 ifconfig 找一下網卡的名字，例如 lo, eth0
    pcpp::PcapLiveDevice *dev = pcpp::PcapLiveDeviceList::getInstance().getPcapLiveDeviceByIpOrName("eth0");

    // 輸出網卡資訊
    std::cout
        << "Interface info:" << std::endl
        << "   Interface name:        " << dev->getName() << std::endl           // get interface name
        << "   Interface description: " << dev->getDesc() << std::endl           // get interface description
        << "   MAC address:           " << dev->getMacAddress() << std::endl     // get interface MAC address
        << "   Default gateway:       " << dev->getDefaultGateway() << std::endl // get default gateway
        << "   Interface MTU:         " << dev->getMtu() << std::endl;           // get interface MTU

    if (dev->getDnsServers().size() > 0)
        std::cout << "   DNS server:            " << dev->getDnsServers().at(0) << std::endl;

    return 0;
}
```

`PcapLiveDevice *dev` 就是網卡介面對應的物件。雖然他是指標，我們在整個程式裡面都不需要去擔心他的生命週期，其實他就是一個 `static` 物件。

稍後我們會用 `dev->open()` 來以混雜模式去開啟對網卡介面的連線。

## 使用 PcapPlusPlus 監聽封包

此小節範例程式請參考 [promiscuous-mode-tutorial/capture.cpp](https://github.com/tigercosmos/promiscuous-mode-tutorial/blob/master/capture.cpp)。

PCPP 提供同步和非同步兩種方式去對封包作處理。同步的話，Main Thread 就會卡住，而非同步就會開啟另一個 Thread 去做監聽，Main Thread 可以繼續做其他事。在處理網路問題時，非同步的操作比較常見，所以以下將以非同步版本作介紹。

要如何開始監聽也很簡單，我們使用 `PcapLiveDevice::startCapture` 即可。這個函數有幾個 Overloads，不過概念都是一樣的，以下取一種做介紹。

```cpp
pcpp::PcapLiveDevice::startCapture(pcpp::OnPacketArrivesCallback onPacketArrives, void *onPacketArrivesUserCookie)
```

以下解釋 `capture.cpp` 範例程式。


`startCapture` 被呼叫之後，PCPP 會開啟一個新的 Thread 去開始監聽封包，收到封包之後可以使用 `pcpp::OnPacketArrivesCallback` Callback 函數去做處理。如果想把東西傳入 Callback 則可以用 `void *onPacketArrivesUserCookie` 把指標傳入，可以稍後再把指標作轉型。
```cpp
dev->startCapture(onPacketArrives, &stats);
```

Callback 的定義是：
```cpp
static void onPacketArrives(pcpp::RawPacket *packet, pcpp::PcapLiveDevice *dev, void *cookie)
```

有三個參數，第一個是 `RawPacket`，基本上就是 Raw Bytes，第二個是來源的裝置，第三個則是剛剛傳入的 Cookie 指標。


```cpp
static void onPacketArrives(pcpp::RawPacket *packet, pcpp::PcapLiveDevice *dev, void *cookie)
{
    // 把傳入的 cookie 做轉型原本的 PacketStats 物件
    PacketStats *stats = (PacketStats *)cookie;

    // 把 RawPacket 變成分析過的 Packet
    pcpp::Packet parsedPacket(packet);

    // 如果封包是 IPv4
    if (parsedPacket.isPacketOfType(pcpp::IPv4))
    {
        // 找出 Source IP 跟 Destination IP
        pcpp::IPv4Address srcIP = parsedPacket.getLayerOfType<pcpp::IPv4Layer>()->getSrcIPv4Address();
        pcpp::IPv4Address destIP = parsedPacket.getLayerOfType<pcpp::IPv4Layer>()->getDstIPv4Address();

        std::cout << "Source IP is '" << srcIP << "'; Dest IP is '" << destIP << "'" << std::endl;
    }

    // 讓 PacketStats 去做統計
    stats->consumePacket(parsedPacket);
}
```

注意到這邊使用 `getLayerOfType<pcpp::IPv4Layer>()` 去取得封包特特定的 Layer，並且解碼都幫我們搞定了，是不是很省事呢！

編譯執行，記住必須使用 `sudo` 去跑，我們可以看到結果：

```bash
$ sudo ./capture
Interface info:
   Interface name:        lo
   Interface description: 
   MAC address:           00:00:00:00:00:00
   Default gateway:       0.0.0.0
   Interface MTU:         65536

Starting async capture...
Source IP is '127.0.0.1'; Dest IP is '127.0.0.1'
Source IP is '127.0.0.1'; Dest IP is '127.0.0.1'
...
...
...
Source IP is '127.0.0.1'; Dest IP is '127.0.0.1'
Source IP is '127.0.0.1'; Dest IP is '127.0.0.1'
Source IP is '127.0.0.1'; Dest IP is '127.0.0.1'
Ethernet packet count: 87
IPv4 packet count:     87
IPv6 packet count:     0
TCP packet count:      85
UDP packet count:      0
DNS packet count:      0
HTTP packet count:     0
SSL packet count:      0
```

因為我是在 WSL 裡面做測試，並且介面設為「lo」，所以封包看起來比較無聊一點。如果本來就是 Linux，可以把裝置設為「eth0」，應該就會有各種來源的 IP。

## 使用 PcapPlusPlus 發送封包

此小節範例程式請參考 [promiscuous-mode-tutorial/create_send.cpp](https://github.com/tigercosmos/promiscuous-mode-tutorial/blob/master/create_send.cpp)。

一個網路封包是由好幾層所組成，我們在發送封包的時候，也是照著把每一層的資訊填上去。PCPP 幫我們做了很好的抽象化，我們只需要把每一層資訊填好，像是依序建立 `pcpp::EthLayer`、`pcpp::IPv4Layer`、`pcpp::UdpLayer` 然後放進 `pcpp::Packet` 裡面即可。

```cpp
    // 建立 Ethernet layer
    pcpp::EthLayer newEthernetLayer(
        pcpp::MacAddress("00:50:43:11:22:33"), // Source MAC
        pcpp::MacAddress("aa:bb:cc:dd:ee"));   // Destination MAC

    // 建立IPv4 layer
    pcpp::IPv4Layer newIPLayer(
        pcpp::IPv4Address("192.168.1.1"), // Source Ip
        pcpp::IPv4Address("10.0.0.1"));   // Destination IP
    newIPLayer.getIPv4Header()->ipId = pcpp::hostToNet16(2000);
    newIPLayer.getIPv4Header()->timeToLive = 64;

    // 建立 UDP layer
    pcpp::UdpLayer newUdpLayer(12345, 53); // 分別為 Source Port 和 Destination Port

    //建立 DNS layer
    pcpp::DnsLayer newDnsLayer;
    newDnsLayer.addQuery(
        "localhost",         // Domain Name 設為 localhost
        pcpp::DNS_TYPE_A,    // Type A 代表 IPv4
        pcpp::DNS_CLASS_IN); // CLASS IN 則是 Internet

    // 建立一個 Packet，預設 Size 是 100，概念跟 std::string 一樣
    pcpp::Packet newPacket(100);

    // 把 Layer 都加上 Packet
    newPacket.addLayer(&newEthernetLayer);
    newPacket.addLayer(&newIPLayer);
    newPacket.addLayer(&newUdpLayer);
    newPacket.addLayer(&newDnsLayer);

    // 去計算要怎麼把 Layer 加上 Packet
    newPacket.computeCalculateFields();

    // 發送 Packet
    dev->sendPacket(&newPacket);
```

可以發現我們可以自由地填 Source IP Address, Source MAC Address，這也是 Promiscuous Mode 的好處，我們可以偽裝自己去送封包，達到很好的測試目的。

我們打開兩個 Shell，一個執行 `./create_send`，我們將會在 lo 上發送發包，而另一個執行 tcmdump 去監測 lo 上的封包。

```sh
(base) tigercosmos@LAPTOP-P7QFA4QB:/mnt/c/Users/tiger$ sudo tcpdump -i lo udp -v -x
tcpdump: listening on lo, link-type EN10MB (Ethernet), capture size 262144 bytes
14:27:50.161985 IP (tos 0x0, ttl 64, id 0, offset 0, flags [none], proto UDP (17), length 55)
    192.168.1.1.12345 > 192.168.1.3.domain: 0 A? localhost. (27)
        0x0000:  4500 0037 0000 0000 4011 f761 c0a8 0101
        0x0010:  c0a8 0103 3039 0035 0023 93c4 0000 0000
        0x0020:  0001 0000 0000 0000 096c 6f63 616c 686f
        0x0030:  7374 0000 0100 01
```

可以發現 tcpdump 顯示封包的資訊跟我們剛剛用程式填的數值一樣，從 `192.168.1.1` 發送到 `192.168.1.3`。

## 使用 PcapPlusPlus 發送和監聽 ARP 

最後我們來統整所學，嘗試發送一個 ARP。

此小節範例程式請參考 [promiscuous-mode-tutorial/arp.cpp](https://github.com/tigercosmos/promiscuous-mode-tutorial/blob/master/arp.cpp)。

接收的部分，我們過濾 ARP 封包，並且只挑出是 ARP Reply 的封包：


```cpp
static void onPacketArrives(pcpp::RawPacket *packet, pcpp::PcapLiveDevice *dev, void *cookie)
{
    // 把 RawPacket 變成分析過的 Packet
    pcpp::Packet parsedPacket(packet);

    // 檢查是否為 ARP 封包
    if (parsedPacket.isPacketOfType(pcpp::ARP))
    {
        pcpp::ArpLayer *arpLayer = parsedPacket.getLayerOfType<pcpp::ArpLayer>();

        // 檢查 APR 是否為 APR Reply
        if (be16toh(arpLayer->getArpHeader()->opcode) == pcpp::ARP_REPLY) // 要做 be16toh 因為網路的編碼是 Big Endian，一般電腦則是 Little Endian
        {
            std::cout << arpLayer->getSenderIpAddr() << ": " << arpLayer->getSenderMacAddress() << std::endl;
        }
    }
}
```

發送部分，我們去更改自己的 Source IP，去符合跟 Target IP 相同的網域，因為有時候一些裝置會設置子網域遮罩，這時我們就必須更改來源 IP 來去欺騙要檢測的裝置。

```
void sendARPRequest(const std::string &dstIpAddr, pcpp::PcapLiveDevice *dev)
{
    // Create an ARP packet and change its fields
    pcpp::Packet arpRequest(500);

    // 把來源 IP 換一個跟目標同域的任一 IP
    std::string srcIPAddr = dstIpAddr;
    while (srcIPAddr.back() != '.')
    {
        srcIPAddr.pop_back();
    }
    srcIPAddr.push_back('2'); // 隨意換一個數字，不要跟目標一樣就好

    pcpp::MacAddress macSrc = dev->getMacAddress();
    pcpp::MacAddress macDst(0xff, 0xff, 0xff, 0xff, 0xff, 0xff);
    pcpp::EthLayer ethLayer(macSrc, macDst, (uint16_t)PCPP_ETHERTYPE_ARP);
    pcpp::ArpLayer arpLayer(pcpp::ARP_REQUEST,
                            macSrc,     // 來源 MAC，不一定要是真的
                            macDst,     // 目標 MAC
                            pcpp::IPv4Address(srcIPAddr),  // 來源 IP，一樣可以造假
                            pcpp::IPv4Address(dstIpAddr)); // 目標 IP

    arpRequest.addLayer(&ethLayer);
    arpRequest.addLayer(&arpLayer);
    arpRequest.computeCalculateFields();

    // 發送 ARP
    dev->sendPacket(&arpRequest);
}
```

我們可以跑跑看程式： 

```sh
$ sudo ./arp eth0 172.22.240.1
172.22.240.1: 00:15:5d:0c:4f:60
```

順利得到答案！

用 tcmdump 也觀察一下這個過程：

```sh
$ sudo tcpdump -i eth0 arp -vv -x
tcpdump: listening on eth0, link-type EN10MB (Ethernet), capture size 262144 bytes
15:45:53.569243 ARP, Ethernet (len 6), IPv4 (len 4), Request who-has LAPTOP-P7QFA4QB.mshome.net tell 172.22.240.2, length 28
        0x0000:  0001 0800 0604 0001 0015 5dc6 5986 ac16
        0x0010:  f002 0000 0000 0000 ac16 f001
15:45:53.569456 ARP, Ethernet (len 6), IPv4 (len 4), Reply LAPTOP-P7QFA4QB.mshome.net is-at 00:15:5d:0c:4f:60 (oui Unknown), length 28
        0x0000:  0001 0800 0604 0002 0015 5d0c 4f60 ac16
        0x0010:  f001 0015 5dc6 5986 ac16 f002
```

這邊的 `LAPTOP-P7QFA4QB.mshome.net` 就是 `172.22.240.1`，我們確實得到正確結果 `00:15:5d:0c:4f:60`！

## 結論

透過 PcapPlusPlus 的幫助，我們可以利用混雜模式（Promiscuous Mode）進行網路封包分析，以收集並分析網路流量中的封包。混雜模式讓網路介面監聽整個網路，不僅能夠收集到發送至本地介面的封包，也可以收集到其他主機發送的封包，唯要注意開啟混雜模式伴隨著網路安全的風險。
