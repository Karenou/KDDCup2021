# kdd  report
数据工程：
1. smoothing ：
![](kdd%20%20report/1AF52967-80B3-4E8A-9698-68FAD7BF4D12.png)
3. 找周期/选择窗口长度：（通过最大～最小，以及lr = 10%，生成窗口list，依次输出，得出相应的 anomely  进行调参）
![](kdd%20%20report/1AAD5D1C-7A1B-41D2-AC10-F01538E50F50.png)

# model
1. fft ：[KDD Cup2021时间序列异常检测竞赛总结 - 知乎](https://zhuanlan.zhihu.com/p/428260507)
![](kdd%20%20report/82B120D7-9D78-4B00-A102-29B54BB32F8B.png)
2.  sr ：[解读：一种基于谱残差和 CNN 的时间序列异常检测模型（附源码链接） - 敲代码的quant的个人空间 - OSCHINA - 中文开源技术交流社区](https://my.oschina.net/u/4586457/blog/5140236) 
![](kdd%20%20report/7C0E21FB-71E3-47F4-856B-5804D30BC134.png)
3.  rrcf ：[【异常检测第四篇】数据流上的鲁棒随机切割树 - 知乎](https://zhuanlan.zhihu.com/p/347000008)
（来自代码main）
![](kdd%20%20report/6E338085-464E-4590-829B-7B1AA0A63020.png)
4. peak / matrix  profile  等统计方法 参加-日本人youtube和github：[GitHub - josteinbf/rrcforest: Robust random cut forest, based on robust random cut tree package](https://github.com/josteinbf/rrcforest)
函数定义：
![](kdd%20%20report/59E36D62-BC02-494F-998D-A66F7C181C61.png)
main.py调用
![](kdd%20%20report/BDF66D4A-BBAA-4878-A381-FAB2BAC3587A.png)



# performance
针对上面几种方法各
每种方法各做两张输出样图，例如

![](kdd%20%20report/97D270FF-2DB0-4FDB-BA19-EFB644E652A8.png)
分析各种方法在哪个类型的数据里面performance比较好
记录相应的confidence


# reference               
https://github.com/hxer7963/FacialExpressionRecognition         
[GitHub - intellygenta/KDDCup2021: Python code for the KDD Cup 2021: Multi-dataset Time Series Anomaly Detection 5th place solution](https://github.com/intellygenta/KDDCup2021)    
mp等方法ppt截图见下：

![](kdd%20%20report/8547CCE2-5AC1-416E-ADDC-D663A4FA872D.png)

![](kdd%20%20report/54F1526A-A5F3-4C3F-9CA4-C0AB21048209.png)
![](kdd%20%20report/156DC664-1DC3-41D1-B289-6162FCC79AA6.png)

![](kdd%20%20report/66730793-F726-45B8-BEDB-E45DAC1A56D6.png)
![](kdd%20%20report/FB8211F8-C63E-4C90-871C-B7AA2C947B8A.png)
![](kdd%20%20report/C9171DFC-7FE6-4189-A5B2-85E7D2A88E3B.png)
![](kdd%20%20report/E5864F72-5AB2-460C-87AA-7AF2ADAA62C3.png)

![](kdd%20%20report/4C1B3547-9D9B-435D-B9E0-8ED192C9FD4A.png)
![](kdd%20%20report/C9F269E8-80AB-4828-B83A-F62C3E9A32A4.png)
