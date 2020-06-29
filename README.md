##Gradcam++

maker：chenhr

gradcam++ imgs 生成流程：forward获取指定卷积层feature_map, backward的时候获取相应feature_map的梯度张量，将两者进过归一化等处理后按照C通道，相乘，累加，最后转换为热力图(没有做最后的relu)。

参考论文：https://arxiv.org/pdf/1710.11063.pdf

###Usage
**shell** : 存放运行的shell文件

**weight** : 存放模型的权重

**model** : 存放模型文件

**loss** : 存放loss文件

**gradcam_imgs** : 存放生成的gradcam文件

**option.py** : 根据你的模型来配置gradcam++，主要有三个方面base Configuration、model Configuration、Loss Configuration

base Configuration：
* nGPU：如果你的模型是多卡训练的，请设置该选项的值大于等于2;
* datadir：存放输入图片的路径，图片具体请按照这样的格式存放：./datadir/一类图片的名字/一类图片，生成的一张图里面就包含所有‘一类图片’;
* load_sel：weight文件的名字，具体存放格式为./weights/weight_name/model/model.pt;
* save:保存生成的gradcam的文件夹名字;
* cam_num：如果你的datadir里面保存过多的图片，你可以只选前cam_num张图片来生成;
* layer_name：根据你模型的文件来选择可视化的层，一般是选择离模型输出最近的一个卷积层（卷积核大小必须大于1x1），或者包含卷积层的一层module；因为是使用register_forward_hook和register_backward_hook函数，所以请选择你的模型文件中init了的层，否则你需要到main_gradcam.py文件中的self.model.get_model()._modules.get(self.args.layer_name).register_forward_hook(p1_farward_hook)和self.model.get_model()._modules.get(self.args.layer_name).register_backward_hook(p1_backward_hook)，将其修改为具体init_layer的第k层，例如self.model.get_model()._modules.get(self.args.layer_name)[k].register_forward_hook(p1_farward_hook);
* layer_name补充说明：例如想可视化resnet50的layer4，在选择好模型文件后，将参设设置为layer4即可；
* height and weight：input的图片resize的大小；

model Configuration：可以根据你的model文件来进行设置参数

Loss Configuration：在gradcam++等论文中，通常都是利用分类得到的结果，对最大概率的那一类和其他类对二分类，然后改造为onehot编码（具体是main_gradcam.py的第157行）， 做反向传播获取梯度，但是实际上根据不同的loss反传获取的梯度不同，最后合成的热力图也会有所不同，因此设置了可以选择不同的loss搭配做出不同的gradcam热力图，格式如1*CrossEntropy+3*Triplet，添加loss请到./loss/__init__.py中设置。ps：程序中现仅对CrossEntropy做onehot处理。

###demo
```shell script
gradcam_pp.sh
```
.
**PS：** 关于model输出的格式，请按照这种这种格式进行存放[[feature_list],[class_list]]。
