---
title: "PcapPlusPlus を使って混雑モード（Promiscuous Mode）でネットワークパケットを解析する"
date: 2023-05-14 18:20:08
tags: [network, promiscuous mode, network analysis, pcap, PcapPlusPlus]
des: "本記事では PcapPlusPlus を用いて混雑モード（Promiscuous Mode）でネットワークパケット解析を行う方法を紹介します。パケットの送信・受信・解析を説明し、最後に ARP のサンプルプログラムを示します。"
lang: jp
translation_key: promiscuous-mode
---

## イントロダクション

混雑モード（Promiscuous Mode）は、ネットワークインタフェースカード（NIC）がネットワークパケットを受信・送信する際の動作モードの一つです。通常モードでは、NIC は自分宛て、または全 NIC 宛てのブロードキャストパケットのみを受信し、それ以外のパケットは受信しません。

一方、混雑モードでは、NIC は他の NIC 宛てのパケットであっても受信できます。これにより NIC はネットワーク全体を監視し、他ホスト間の通信を含む、ネットワーク上を流れるデータを捕捉できます。そのため混雑モードは、ネットワーク障害の切り分け、ネットワーク解析、セキュリティ監視などでよく使われます。ただし、不正な盗聴や攻撃に悪用される可能性もあります。

tcpdump や Wireshark などの一般的なパケット解析ツールも、背後では混雑モードを前提に解析を行っています。

Linux 上でパケット解析を行うには、`socket` を `SOCK_RAW` に設定し、さらに全プロトコル `ETH_P_ALL` を監視します。

そして Socket に対して `setsockopt` を使い `PACKET_MR_PROMISC`、つまり混雑モードを設定します。（[packet(7)](https://man7.org/linux/man-pages/man7/packet.7.html) を参照）

```c
int sock = socket(PF_PACKET, SOCK_RAW, htons(ETH_P_ALL));

struct packet_mreq mr;
memset(&mr, 0, sizeof(mr));
mr.mr_ifindex = if_nametoindex(iface);
mr.mr_type = PACKET_MR_PROMISC;
setsockopt(sock, SOL_PACKET, PACKET_ADD_MEMBERSHIP, &mr, sizeof(mr));
```

また、混雑モードを使う前に NIC 側の設定も忘れずに行いましょう：

```sh
sudo ip link set <interface_name> promisc on
```

これで、インタフェースを通過するすべてのパケットを監視できます。設定しない場合、NIC が宛先 MAC アドレスが自分ではないパケットをフィルタしてしまうことがあります。

![COVER IMAGE](https://github.com/tigercosmos/blog/assets/18013815/3100c948-f555-43ad-b58f-71047a0d3a3c)

## PcapPlusPlus の紹介

System Call を直接使って実装するのはかなり大変です。もう少し高レベルな方法として libpcap を使えます。「pcap」は Packet Capture（パケットキャプチャ）の略で、もともとは tcpdump の開発者が作った C のライブラリで、使いやすいようにある程度ラップされています。そして PcapPlusPlus は libpcap をベースにした、より高レベルなラッパーです。より簡単で使いやすいインタフェースと、より多くの機能を提供します。PcapPlusPlus は Windows / Linux / macOS のクロスプラットフォーム開発をサポートし、パケットの捕捉・解析・改変などを容易に実現できます。さらに多様なネットワークプロトコルの解析・デコードもサポートしているため、自分で面倒なプロトコル実装をする必要がなく、大幅に時間を節約できます。

以下では PcapPlusPlus（本文中では PCPP と略します）の使い方を紹介します。GitHub には完全な[サンプルコード](https://github.com/seladb/PcapPlusPlus/tree/master/Examples)があるので、そちらも参考になります。

## PcapPlusPlus のインストール

PcapPlusPlus の公式[インストール手順](https://pcapplusplus.github.io/docs/install/linux)は少し複雑に見えますが、実際には CMake で簡単にインストールできます。

```sh
# pre-requirement
sudo apt-get install libpcap-dev

git clone https://github.com/seladb/PcapPlusPlus.git
cd PcapPlusPlus/

mkdir build; cd build; cmake ..; make -j8; sudo make install
```

インストール後、ヘッダとライブラリのデフォルトパスはそれぞれ `/usr/local/include/pcapplusplus/` と `/usr/local/lib/` です。

## 本記事のサンプルコード

PcapPlusPlus をインストールしたら、[本記事のサンプルコード](https://github.com/tigercosmos/promiscuous-mode-tutorial)をダウンロードしてください：

```
git clone https://github.com/tigercosmos/promiscuous-mode-tutorial

cd promiscuous-mode-tutorial

mkdir build; cd build; cmake ..; make 
```

## CMake の設定

次に CMake の設定方法を紹介します。

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

リンクするライブラリは `Packet++`、`Pcap++`、`Common++` の 3 つです。前に `PcapPlusPlus::` を付けておくと安全です（付けないと CMake が見つけられないことがあります）。

## PcapPlusPlus Hello World

次は Hello World です。[promiscuous-mode-tutorial/hello_world.cpp](https://github.com/tigercosmos/promiscuous-mode-tutorial/blob/master/hello_world.cpp) を参照してください。

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

`PcapLiveDevice *dev` は NIC インタフェースに対応するオブジェクトです。ポインタですが、このプログラムではライフサイクルを気にする必要はありません。実質的に `static` なオブジェクトとして扱えます。

このあと `dev->open()` を使って、混雑モードで NIC への接続を開きます。

## PcapPlusPlus でパケットを監視する

この節のサンプルコードは [promiscuous-mode-tutorial/capture.cpp](https://github.com/tigercosmos/promiscuous-mode-tutorial/blob/master/capture.cpp) を参照してください。

PCPP には同期・非同期の 2 種類の処理方法があります。同期の場合はメインスレッドがブロックし、非同期の場合は別スレッドを立てて監視し、メインスレッドは他の処理を続けられます。ネットワーク問題の調査では非同期がよく使われるため、ここでは非同期版を紹介します。

監視の開始は簡単で、`PcapLiveDevice::startCapture` を呼ぶだけです。いくつかオーバーロードがありますが、概念は同じです。ここでは次の形を使います：

```cpp
pcpp::PcapLiveDevice::startCapture(pcpp::OnPacketArrivesCallback onPacketArrives, void *onPacketArrivesUserCookie)
```

`startCapture` を呼ぶと、PCPP は新しいスレッドを作ってパケット監視を開始します。パケットを受信したら `pcpp::OnPacketArrivesCallback` のコールバックで処理できます。コールバックに値を渡したい場合は `void *onPacketArrivesUserCookie` にポインタを渡し、後でキャストして使えます。

```cpp
dev->startCapture(onPacketArrives, &stats);
```

コールバックの定義は次の通りです：

```cpp
static void onPacketArrives(pcpp::RawPacket *packet, pcpp::PcapLiveDevice *dev, void *cookie)
```

引数は 3 つで、1 つ目は `RawPacket`（基本的に生バイト列）、2 つ目は送信元のデバイス、3 つ目は先ほど渡した Cookie のポインタです。

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

ここでは `getLayerOfType<pcpp::IPv4Layer>()` を使って特定のレイヤを取得しています。デコードは PCPP がやってくれるので、とても楽です。

コンパイルして実行します。`sudo` が必要なことに注意してください。次のような結果が出ます：

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

私は WSL でテストし、インタフェースを `lo` にしているため、パケット内容はやや地味です。Linux 環境であればデバイスを `eth0` にすると、多様な送信元 IP のパケットが見えるはずです。

## PcapPlusPlus でパケットを送信する

この節のサンプルコードは [promiscuous-mode-tutorial/create_send.cpp](https://github.com/tigercosmos/promiscuous-mode-tutorial/blob/master/create_send.cpp) を参照してください。

ネットワークパケットは複数のレイヤから構成されます。送信時も各レイヤの情報を順に埋めていきます。PCPP は良い抽象化を提供しており、`pcpp::EthLayer`、`pcpp::IPv4Layer`、`pcpp::UdpLayer` を順に作り、`pcpp::Packet` に入れるだけで済みます。

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

ここでは Source IP Address や Source MAC Address を自由に設定できます。これは Promiscuous Mode を使った解析・テストの強みの一つで、なりすましてパケットを送ることでテストを行えます。

2 つのシェルを開き、1 つ目で `./create_send` を実行して `lo` にパケットを送信し、もう 1 つで tcpdump を実行して `lo` のパケットを監視します。

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

tcpdump の出力が、先ほどプログラムで埋めた値と一致していることが分かります。`192.168.1.1` から `192.168.1.3` に送られています。

## PcapPlusPlus で ARP を送信・監視する

最後に、ここまで学んだ内容をまとめて ARP を送ってみます。

この節のサンプルコードは [promiscuous-mode-tutorial/arp.cpp](https://github.com/tigercosmos/promiscuous-mode-tutorial/blob/master/arp.cpp) を参照してください。

受信側では ARP パケットをフィルタし、さらに ARP Reply だけを取り出します：

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

送信側では、Source IP を変更して Target IP と同じネットワークに属するようにします。デバイスによってはサブネットマスクを設定していることがあり、その場合、送信元 IP を変えて探査対象のデバイスを「騙す」必要があります。

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

実行してみます：

```sh
$ sudo ./arp eth0 172.22.240.1
172.22.240.1: 00:15:5d:0c:4f:60
```

無事に答えが得られました！

tcpdump でこの過程を観察してみます：

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

ここで `LAPTOP-P7QFA4QB.mshome.net` は `172.22.240.1` に対応しており、確かに正しい結果 `00:15:5d:0c:4f:60` が得られています！

## 結論

PcapPlusPlus を使うことで、混雑モード（Promiscuous Mode）でネットワークパケット解析を行い、ネットワークトラフィック中のパケットを収集・分析できます。混雑モードでは、ローカルインタフェース宛てのパケットだけでなく、他ホスト宛てのパケットも収集できるため、ネットワーク全体の監視が可能になります。ただし、混雑モードの有効化にはセキュリティ上のリスクが伴う点に注意してください。
