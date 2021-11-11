# 分类损失函数
<details>
  <summary>Cross Entropy</summary>
  <h2>1. 损失函数介绍</h2>
    <br />交叉熵（Cross Entropy）是Shannon信息论中一个重要概念，主要用于度量两个概率分布间的差异性信息。
    <br />交叉熵损失函数用于度量实际输出（概率）与期望输出（概率）的距离，也就是交叉熵的值越小，两个概率分布就越接近。
    <br />交叉熵函数常用于分类(classification)。
  <h2>2. 表达式</h2>
    <br />Cross Entropy Loss 定义如下:
    <br /><img src = "figures/CELoss.png" width = "50%">
  <h2>3. 代码实现</h2>
    <br />Cross Entropy Loss的Python代码
    <pre>
    
    def cross_entropy(a, y):
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    # tensorflow version
    loss = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]))

    # numpy version
    loss = np.mean(-np.sum(y_*np.log(y), axis=1))</pre>
</details>

<details>
  <summary>Focal Loss</summary>
  <h2>1. 损失函数介绍</h2>
    <br /> Focal Loss是用于分类问题的带参损失函数, 当前object detection算法：
    <br /> 1. two-stage detector: Faster-RCNN为代表，需要region proposal的算法，由于RPN需要对object进行两次过滤(2-stage)，准确率较高但速度慢
    <br /> 2. one-stage detector: YOLO为代表，速度快准确率不高
    <br /> Focal loss 的目的是让one-stage在维持速度的前提下达到two-stage准确率。作者认为one-stage准确率不佳的核心原因：样本类别不均衡。Focal Loss采用调制因子来减少易分类样本的权重，从而使得模型在训练时更专注于难分类的样本。
  <h2>2. 表达式</h2>
    <br />focal Loss 定义如下:
    <br /><img src = "figures/focal_loss.png" width = "50%">
  <h2>3. 代码实现</h2>
    <br />Focal损失函数的Python代码
    <pre>
       class FocalLoss(nn.Module):
          def __init__(self, gamma=0,alpha=1):
              super(FocalLoss, self).__init__()
              self.gamma = gamma
              self.ce = nn.CrossEntropyLoss()
              self.alpha=alpha
          def forward(self, input, target):
              logp = self.ce(input, target)
              p = torch.exp(-logp)
              loss = (1 - p) ** self.gamma * logp
              loss = self.alpha*loss
              return loss.mean()</pre>
</details>

<details>
  <summary>Hinge Loss</summary>
  <h2>1. 损失函数介绍</h2>
    <br /> 用于2分类问题的不带参损失函数，标签值y的取值+1/-1, 预测值y'∈R, 该二分类问题的目标函数的要求：当y大于等于+1或者小于等于-1时，都是分类器确定的分类结果，此时的损失函数loss为0；而当预测值y'∈(−1,1)时，分类器对分类结果不确定，loss不为0。显然，当y'=0时，loss达到最大值重，从而使得模型在训练时更专注于难分类的样本。
  <h2>2. 表达式</h2>
    <br />Hinge Loss 定义如下:
    <br /><img src = "figures/hinge_loss.png" width = "50%">
  <h2>3. 代码实现</h2>
    <br />Hinge损失函数的Python代码
    <pre>
       loss = max(0, 1-target*prediction)</pre>
</details>

<details>
  <summary>Logistic Loss</summary>
  <h2>1. 损失函数介绍</h2>
    <br /> 用于二分类问题的损失函数
  <h2>2. 表达式</h2>
    <br />Logistic Loss 定义如下:
    <br /><img src = "figures/Logistic_loss.png" width = "50%">
  <h2>3. 代码实现</h2>
    <br />Logistic损失函数的Python代码
    <pre>
       loss = 1 / (1 + torch.exp(-x))</pre>
</details>



# 回归损失函数
<details>
  <summary>Huber loss</summary>
  <h2>1. 损失函数介绍</h2>
    <br /> Huber Loss 是一个用于回归问题的带参损失函数, 优点是能增强平方误差损失函数(MSE, mean square error)对离群点的鲁棒性。
    <br /> 当预测偏差小于 δ 时，它采用平方误差,当预测偏差大于 δ 时，采用的线性误差。
    <br /> 相比于最小二乘的线性回归，HuberLoss降低了对离群点的惩罚程度，所以 HuberLoss 是一种常用的鲁棒的回归损失函数。
  <h2>2. 表达式</h2>
    <br />Huber Loss 定义如下:
    <br /><img src = "figures/Huber Loss.png" width = "50%">
  <h2>3. 代码实现</h2>
    <br />Huber损失函数的Python代码
    <pre># huber 损失
def huber(true, pred, delta):
    loss = np.where(np.abs(true-pred) < delta , 0.5*((true-pred)**2), delta*np.abs(true - pred) - 0.5*(delta**2))
    return np.sum(loss)</pre>
</details>

<details>
  <summary>L1/L2 loss</summary>
  <h2>1. L1-norm loss function</h2>
    <br /><img src = "figures/L1.png" width = "100%">
  <h2>2. L2-norm loss function</h2>
    <br /><img src = "figures/L2.png" width = "100%">
  <h2>3. L1和L2 损失函数区别</h2>
    <br />L2损失函数是最最常用的损失函数，在回归问题中，也就是我们耳熟能详的最小二乘法。并且在满足高斯马尔可夫条件的时候，可以证明使用L2损失函数所得的参数具有无偏性和有效性。
    <br />但是，L1损失函数也有其自己的优点，下面我们对两个损失函数进行比较。
    <br /><img src = "figures/L1L2.png" width = "100%"> 
    <br />稳健性:
    <br />L1损失函数稳健性强是它最大的优点。面对误差较大的观测，L1损失函数不容易受到它的影响。这是因为:L1损失函数增加的只是一个误差，而L2损失函数增加的是误差的平方。当误差较大时，使用L2损失函数，我们需要更大程度的调整模型以适应这个观测，所以L2损失函数没有L1损失函数那么稳定。
    <br />那么，当我们认为模型中可能存在异常值时，使用L1损失函数可能会更好；但是，当我们需要把误差较大的观测也纳入模型中时，使用L2损失函数更好一些。
    <br />解的稳定性:
    <br />首先，从求解效率上来说，L2损失函数处处可导，而L1损失函数在零点位置是不可导的，这就使得使用L2损失函数求解可以得到一个解析解，而L1损失函数则没有；
    <br />其次，当数据有一个微小的变化时，L1损失函数的变化更大，其解更加的不稳定。

  <h2>4. 代码实现</h2>
    <br />L1/L2 loss的Python代码
    <pre>
    import numpy as np
    #定义L1损失函数
    def L1_loss(y_true,y_pre): 
        return np.sum(np.abs(y_true-y_pre))
    #定义L2损失函数
    def L2_loss(y_true,y_pre):
        return np.sum(np.square(y_true-y_pre))</pre>
</details>

# 特定任务损失函数
<details>
  <summary>Verification loss(Re-ID)</summary>
  <h2>1. 损失函数介绍</h2>
  <h2>2. 表达式</h2>
</details>
  
<details>
  <summary>Triplet Loss</summary>
  <h2>1. 损失函数介绍</h2>
    <br /> 回归问题损失函数，用于人脸识别，学习人脸的embedding, 相似的人脸对应的embedding在特征空间内相近，以此距离作人脸识别
  <h2>2. 表达式</h2>
    <br />Triplet Loss 定义如下:
    <br /> (a, p, n) a: anchor, p: positive sample, n: negetive sample
    <br /><img src = "figures/Triplet_loss.png" width = "50%">
  <h2>3. 代码实现</h2>
    <br />Triplet损失函数的Python代码
    <pre>
       triplet_loss = np.maximum(positive_dist - negative_dist + margin, 0.0)</pre>
</details>
  
<details>
  <summary>Contrastive Loss</summary>
  <h2>1. 损失函数介绍</h2>
    <br /> 对比学习的损失函数，使近似样本之间的距离越小越好。不近似样本之间的距离如果小于m，则通过互斥使其距离接近m。
  <h2>2. 表达式</h2>
    <br />Contrastive Loss 定义如下:
    <br /><img src = "figures/contrastive_loss.png" width = "50%">
  <h2>3. 代码实现</h2>
    <br />交叉损失函数的Python代码
    <pre>
  class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label)*torch.pow(euclidean_distance, 2)\
          +(label)*torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))     
        return loss_contrastive</pre>
</details>
