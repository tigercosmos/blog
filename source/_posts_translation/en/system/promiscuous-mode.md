---
title: "Network Packet Analysis with PcapPlusPlus in Promiscuous Mode"
date: 2023-05-14 18:20:08
tags: [network, promiscuous mode, network analysis, pcap, PcapPlusPlus]
des: "This post introduces how to analyze network packets with PcapPlusPlus in promiscuous mode, including how to send, receive, and parse packets, and finally provides an ARP example program."
lang: en
translation_key: promiscuous-mode
---

## Introduction

Promiscuous mode is a mode of operation for a Network Interface Card (NIC) when receiving and transmitting network packets. In normal mode, a NIC only receives packets sent to itself or broadcast packets sent to all NICs, and it will not receive other packets.

In promiscuous mode, the NIC can receive packets sent to any NIC, even if those packets are not addressed to itself. This mode allows the NIC to monitor the entire network and capture all transmitted data, including communication between other hosts. Therefore, promiscuous mode is commonly used for network troubleshooting, network analysis, and security monitoring—but it can also be abused for illegal sniffing and attacks.

Common packet analysis tools such as tcpdump and Wireshark are based on promiscuous mode under the hood.

On Linux, to perform packet analysis, we can use a `socket` with `SOCK_RAW` and listen to all protocols via `ETH_P_ALL`.

We also need to use `setsockopt` to set the socket to `PACKET_MR_PROMISC`, i.e., promiscuous mode. (See [packet(7)](https://man7.org/linux/man-pages/man7/packet.7.html).)

```c
int sock = socket(PF_PACKET, SOCK_RAW, htons(ETH_P_ALL));

struct packet_mreq mr;
memset(&mr, 0, sizeof(mr));
mr.mr_ifindex = if_nametoindex(iface);
mr.mr_type = PACKET_MR_PROMISC;
setsockopt(sock, SOL_PACKET, PACKET_ADD_MEMBERSHIP, &mr, sizeof(mr));
```

Also, before using promiscuous mode, remember to configure the NIC:

```sh
sudo ip link set <interface_name> promisc on
```

This way you can capture all packets that traverse the interface; otherwise the NIC may filter out packets whose destination MAC address is not yours.

![COVER IMAGE](https://github.com/tigercosmos/blog/assets/18013815/3100c948-f555-43ad-b58f-71047a0d3a3c)

## An Introduction to PcapPlusPlus

Working directly with syscalls is tiring. A more advanced option is libpcap. “pcap” stands for Packet Capture. It was originally developed as a C library by the tcpdump developers, with wrappers to make it easier to use. PcapPlusPlus is a higher-level wrapper built on top of libpcap. It provides a simpler interface and more features. PcapPlusPlus supports cross-platform development on Windows, Linux, and macOS. It makes it easy to capture, parse, and modify network packets. It also supports analysis and decoding for various network protocols. This means we don’t have to implement messy protocol parsing ourselves, which saves a lot of time.

Below I will introduce how to use PcapPlusPlus (abbreviated as PCPP). You can also refer to its GitHub repository, which contains complete [example programs](https://github.com/seladb/PcapPlusPlus/tree/master/Examples).

## Installing PcapPlusPlus

The official PcapPlusPlus [installation guide](https://pcapplusplus.github.io/docs/install/linux) is a bit complicated. In practice, you can install it easily using CMake.

```sh
# pre-requirement
sudo apt-get install libpcap-dev

git clone https://github.com/seladb/PcapPlusPlus.git
cd PcapPlusPlus/

mkdir build; cd build; cmake ..; make -j8; sudo make install
```

After installation, the default include and library paths are `/usr/local/include/pcapplusplus/` and `/usr/local/lib/`.

## Example Code for This Post

After installing PcapPlusPlus, download the [example code for this post](https://github.com/tigercosmos/promiscuous-mode-tutorial):

```
git clone https://github.com/tigercosmos/promiscuous-mode-tutorial

cd promiscuous-mode-tutorial

mkdir build; cd build; cmake ..; make 
```

## CMake Setup

Next, let’s see how to set up CMake.

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

There are three libraries to link: `Packet++`, `Pcap++`, and `Common++`. It is safer to prefix them with `PcapPlusPlus::`, because sometimes CMake may not find them otherwise.

## PcapPlusPlus Hello World

Now let’s do a “Hello World”. See [promiscuous-mode-tutorial/hello_world.cpp](https://github.com/tigercosmos/promiscuous-mode-tutorial/blob/master/hello_world.cpp).

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

`PcapLiveDevice *dev` is the object corresponding to a network interface. Although it is a pointer, we do not need to worry about its lifetime in this program—it is effectively a `static` object.

Later we will call `dev->open()` to open the interface connection in promiscuous mode.

## Capturing Packets with PcapPlusPlus

For this section, see [promiscuous-mode-tutorial/capture.cpp](https://github.com/tigercosmos/promiscuous-mode-tutorial/blob/master/capture.cpp).

PCPP provides both synchronous and asynchronous ways to process packets. In synchronous mode, the main thread blocks. In asynchronous mode, another thread is created to capture packets, while the main thread can continue doing other work. For network troubleshooting, asynchronous capture is more common, so I will use the asynchronous version here.

Starting capture is easy: just use `PcapLiveDevice::startCapture`. There are several overloads, but the idea is the same. Here is one of them:

```cpp
pcpp::PcapLiveDevice::startCapture(pcpp::OnPacketArrivesCallback onPacketArrives, void *onPacketArrivesUserCookie)
```

After `startCapture` is called, PCPP starts a new thread to capture packets. When a packet arrives, the callback `pcpp::OnPacketArrivesCallback` is invoked to process it. If you want to pass something into the callback, you can use `void *onPacketArrivesUserCookie` to pass a pointer and cast it back later.

```cpp
dev->startCapture(onPacketArrives, &stats);
```

The callback signature is:

```cpp
static void onPacketArrives(pcpp::RawPacket *packet, pcpp::PcapLiveDevice *dev, void *cookie)
```

It has three parameters: `RawPacket` (raw bytes), the source device, and the cookie pointer passed above.

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

Note that we use `getLayerOfType<pcpp::IPv4Layer>()` to get a specific layer, and decoding is handled for us—very convenient.

Compile and run. Remember you must use `sudo`. You should see output like:

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

Because I tested this in WSL and set the interface to `lo`, the packets look a bit boring. If you are on Linux, you can set the device to `eth0`, and you should see packets from various source IPs.

## Sending Packets with PcapPlusPlus

For this section, see [promiscuous-mode-tutorial/create_send.cpp](https://github.com/tigercosmos/promiscuous-mode-tutorial/blob/master/create_send.cpp)。

A network packet consists of multiple layers. When sending packets, we fill in each layer’s fields in order. PCPP provides a nice abstraction: we only need to create layers (e.g., `pcpp::EthLayer`, `pcpp::IPv4Layer`, `pcpp::UdpLayer`) and add them into a `pcpp::Packet`.

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

You can see we can freely fill in the source IP address and source MAC address. This is also one of the “benefits” of promiscuous-mode style tooling: we can craft packets and spoof identities to send packets for testing purposes.

Open two shells: run `./create_send` in one (it will send packets on `lo`), and run tcpdump in the other to monitor packets on `lo`.

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

We can see tcpdump shows packet fields matching what we filled in the program: sent from `192.168.1.1` to `192.168.1.3`.

## Sending and Capturing ARP with PcapPlusPlus

Finally, let’s put everything together and send an ARP packet.

For this section, see [promiscuous-mode-tutorial/arp.cpp](https://github.com/tigercosmos/promiscuous-mode-tutorial/blob/master/arp.cpp)。

On the receiving side, we filter ARP packets and only pick ARP Reply packets:

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

On the sending side, we change our source IP to match the target IP’s subnet. This is because some devices configure a subnet mask, and then we must change the source IP to “trick” the device we want to probe.

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

Run the program:

```sh
$ sudo ./arp eth0 172.22.240.1
172.22.240.1: 00:15:5d:0c:4f:60
```

We successfully get the answer!

We can also observe the process with tcpdump:

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

Here, `LAPTOP-P7QFA4QB.mshome.net` corresponds to `172.22.240.1`. We indeed get the correct result `00:15:5d:0c:4f:60`!

## Conclusion

With the help of PcapPlusPlus, we can analyze network packets in promiscuous mode and collect and inspect packets in network traffic. Promiscuous mode enables a network interface to monitor the entire network: it can capture not only packets addressed to the local interface, but also packets sent by other hosts. However, be aware that enabling promiscuous mode comes with security risks.
