# LogciMD
**Interpretable Multimodal Misinformation Detection with Logic Reasoning**

原文链接：https://aclanthology.org/2023.findings-acl.620

官方公开代码：https://github.com/less-and-less-bugs/LogicMD

*LogicMD提出了一种基于逻辑的新型多模态错误信息检测神经模型。该模型通过集成可解释的逻辑子句来表达推理过程，利用神经表示对符号逻辑元素进行参数化，从而能够自动生成和评估有意义的**逻辑子句**。此外，为了适应不同的错误信息来源，LogicMD引入了**五个元谓词**，以便根据不同的相关性进行实例化。

# Datasets

论文原始数据集下载地址：https://portland-my.sharepoint.com/:u:/g/personal/liuhui3-c_my_cityu_edu_hk/EYR-45i16q9EivlGM1ZCe9cBrPsuOjr8O9fziihKJPLIoA?e=2KVCFW

本项目使用的数据集为story-based数据集，下载地址：(https://drive.google.com/drive/folders/1rLrh5x5UlYskfbhhVyz523MKgmCDyuX2)，此外本项目为多模态假新闻检测，还需要下载相应图片，下载地址为：(https://drive.google.com/drive/folders/11okt9IRDxXgfTr7Ae1wxl9CHZC1PphhC)。将下载好的数据集放到dataset_our目录下。

