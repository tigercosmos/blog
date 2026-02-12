---
title: "How Far Are We From Error? On Measurement Bias in Systems Experiments"
date: 2020-08-30 18:30:00
tags: [system software, measurement, bias, spec cpu]
des: "This paper argues that even if an experiment appears fine, you may silently obtain incorrect data due to the initial setup. On Unix systems, environment variable settings and the link order of object files in C++ can both cause large performance differences."
lang: en
translation_key: measurement-bias
---

## 1. Introduction

For systems researchers, much of the work is about developing or improving a system. To prove the proposed system is effective, we often need experiments to demonstrate that the performance matches expectations—typically by running benchmarks or customized applications. But are the numbers we measure in these experiments actually correct?

This post mainly introduces the paper “[Producing Wrong Data Without Doing Anything Obviously Wrong!](https://users.cs.northwestern.edu/~robby/courses/322-2013-spring/mytkowicz-wrong-data.pdf)” by [Todd Mytkowicz](https://scholar.google.com/citations?user=Z4y_Z3sAAAAJ&hl=en) et al. The paper has been cited more than 300 times. It not only caught my attention, but also genuinely shocked me.

The paper argues that even if an experiment appears to have no problems, we may still silently obtain incorrect results due to the **initial setup**. In Unix systems, both **environment variable settings** and **changes to the object-file link order used by C++ compilers** can lead to very large differences in measured results.

## 2. How Sensitive Programs Can Be

The figure below shows the results of running the code on the left under different environment variable sizes.

![paper figure 1](https://user-images.githubusercontent.com/18013815/91653233-3ff05180-ead1-11ea-9340-743c0b04cb84.png)

We can see that as the environment variable size changes, at the 95% confidence level, the cycle count can vary by around 33%, and in some cases even “spike” to a 300% difference. Environment variables are loaded into memory before the program starts; different environment sizes change memory alignment (Alignment) of subsequent variables, which then causes performance differences.

This simple example tells us that programs are more sensitive than we might think: the initial system environment setup can absolutely affect the observed runtime behavior.

## 3. Experiments

The experiments use a subset of SPEC CPU 2006 benchmarks.

![benchmark](https://user-images.githubusercontent.com/18013815/91654974-e5122680-eadf-11ea-99b7-14aba6c8cbd8.png)

The experimental setups include two CPU platforms (Core2 and Pentium 4) and a simulator (m5-O3CPU), to validate that these bias effects are not tied to a particular device.

![image](https://user-images.githubusercontent.com/18013815/91654988-007d3180-eae0-11ea-8c42-c0c693b53bf8.png)

### 3.1 Bias Caused by Link Order

When compiling, the compiler links multiple `.o` files together. Different link orders lead to different memory layouts.

Taking the perlbench results below as an example, we compare the default link order, alphabetical order, and various random permutations (numbers 3–33). You can see that the ratio of cycles when running GCC compiled with O2 vs O3 differs substantially. If there were no bias, in theory the ratio should be stable.

![figure 2-a](https://user-images.githubusercontent.com/18013815/91655565-1a207800-eae4-11ea-9839-e8f1a3103447.png)

Comparing all benchmarks together:

![figure 2-b](https://user-images.githubusercontent.com/18013815/91655656-ba769c80-eae4-11ea-86d2-7b5cc359c365.png)

We can see that depending on the link order, the O2/O3 ratio can range from 0.95 to 1.10, and most benchmarks show differences of about 0.05. According to the paper’s experiments, this phenomenon is not specific to a device or compiler; similar results occur broadly. This means that for perlbench, since the range reaches 0.15, if an experiment uses this benchmark, a claim like “our system is 10% faster” might actually correspond to being 5% slower.

The paper further investigates why this happens. In the m5 simulator with the O3CPU model, for the bzip2 benchmark, whether a hot loop happens to land on a cache line can have a large performance impact. Because the simulator can access the source, the authors can infer the cause. But on real hardware, vendors disclose too little detail, so the authors say they cannot be certain whether Intel Core2 is affected for the same reason; they can only guess it is likely similar. As of “today” in 2020, it’s unclear whether Intel CPUs provide enough information—who knows how this looks now?

### 3.2 Bias Caused by Environment Variable Size

As mentioned earlier, environment variable size directly affects the stack. Using perlbench as an example, the O2/O3 range also fluctuates, and this bias is irregular—almost impossible to summarize or predict.

![perlbench environment size](https://user-images.githubusercontent.com/18013815/91655921-01659180-eae7-11ea-8c70-de929019bc9b.png)

The violin plot over all benchmarks is shown below.

![all benchmark environment size](https://user-images.githubusercontent.com/18013815/91655945-3376f380-eae7-11ea-823c-c9f639052c63.png)

We can see that most benchmarks have a small range, except for perlbench and lbm. Most are around 1%–4%. Although this is much smaller than the bias caused by link order, it is still enough to mislead experimental analysis. For lbm, you might think you are 12% slower, but in fact you might have a 9% improvement!

If we fix the initial stack position of the program, experiments show that the O2/O3 values basically do not change with environment variable size. In addition, for perlbench, environment variables are copied into the heap at startup, which can cause large differences in later object alignment. Therefore, if we fix the starting heap position, we can also ensure there is no bias.

## 4. The Unpredictability of Bias

You might wonder: since the initial setup can cause large differences, can we tune for an “optimal” value?

Unfortunately, as the data above shows, this bias is completely irregular. At best, we can choose a parameter that performs best on one machine—but does the best value on one machine equal the best value on another machine?

![different device link order](https://user-images.githubusercontent.com/18013815/91656149-f3b10b80-eae8-11ea-905c-e83010ea2aa8.png)

The figure above compares Core 2 and Pentium 4. The x–y axes represent the cycle counts under the same link order on the two machines. We find that these points show no clear pattern. The circled points indicate the best-performing link orders on the two machines, but they correspond to different link permutations. This tells us that link-order bias across machines is also irregular.

![different device env size](https://user-images.githubusercontent.com/18013815/91656220-a08b8880-eae9-11ea-9e9d-96f7c6d496cf.png)

Similarly, the figure above leads to the same conclusion for environment variable size: the environment size that yields the best performance differs across machines.

## 5. How to Avoid Bias

Now that we know experiments can be biased, we should try our best to avoid it. In fact, bias is not a new topic—many scientific fields have to deal with bias.

### 5.1 A Larger Benchmark Suite

We might try to increase the breadth and complexity of the benchmark suite so that the effect of bias “cancels out”.

Unfortunately, the authors tested this hypothesis. They ran all benchmarks under 66 different configurations, averaged results, and plotted the distribution of the averaged benchmark results. If complexity were sufficient to cancel out bias, we would expect a very narrow distribution. But instead, the observed distribution is wide, which means **we cannot rely on benchmark complexity to cancel out the bias introduced by the initial setup.**

![Distribution of speedup due to O3 as we change the experimental setup.](https://user-images.githubusercontent.com/18013815/91656380-e09f3b00-eaea-11ea-813f-eebcdeb1bea0.png)

### 5.2 Randomizing the Experimental Setup

One approach is to use statistics to strengthen our claims. We can use many randomized initial setups, and then use statistical methods such as a T-test to report confidence intervals.

![Using setup randomization to determine the speedup of O3 for perlbench](https://user-images.githubusercontent.com/18013815/91656507-c74abe80-eaeb-11ea-9a81-84d8ac297597.png)

As shown above, the authors used 22 different link orders and 22 different environment variable sizes, for a total of 484 initial setups. They ran each configuration three times and averaged the results. Under a 95% confidence level, the mean O2/O3 ratio is 1.007 ± 0.003. With this, we can confidently say that O3 is indeed better optimized!

### 5.3 Causal Analysis

Finally, to be confident that our conclusions from data are correct, we can adopt causal analysis.

Below is a definition adapted from Wikipedia:

> In general, causality can refer to the relationship between a set of factors (causes) and a phenomenon (effect). Any event that influences an outcome is a factor of that outcome. A direct factor directly affects the result, i.e., without intermediary factors (which are sometimes called mediators). From this perspective, the relationship between cause and effect can also be called a causal nexus. If there are two events, A and B, and without A, B would not occur, then A is the cause of B, and B is the effect of A.

With rigorous causal analysis, it becomes less likely that we draw incorrect conclusions due to experimental bias.

## 6. Conclusion

The authors analyzed 133 papers from venues including ASPLOS, PACT, PLDI, and CGO, and found that none of them mentioned the kind of bias highlighted in this paper. This is quite shocking. How can we be sure those experimental results were not affected by initial values or other sources of bias?

When reading papers, I also found that most papers almost never discuss measurement error. Many do not even use statistical analysis or plot error bars. We should pay more attention to bias in experiments. This paper gave me a huge shock: bias can affect performance by as much as 10%. When we fight over a claimed 5% performance improvement, are we actually being influenced by a 10% bias?
