import torch
from torch import nn


def distanceL2(h, t):
    s = h - t
    sum = torch.square(s).sum(-1) # 降维(横向压缩)
    return sum

def dot_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())  # im*s的转置

def l2_sim(im, s):
    b_im = im.shape[0]
    b_s = s.shape[0]
    # unsqueeze(0).repeat(b_s,1,1)在0维处给tensor增加一维，沿着指定的维度对原来的tensor进行数据复制
    return distanceL2(im.unsqueeze(0).repeat(b_s,1,1),s.unsqueeze(1).repeat(1,b_im,1)).transpose(0,1)


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=1.0, measure=False, max_violation=False):
        # max_violation 是否用最难样本
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        if measure == 'l2':
            self.sim = l2_sim
        if measure == 'dot':
            self.sim = dot_sim

        self.max_violation = max_violation

    def forward(self, im, s):
        # compute image-sentence score matrix
        #im,s维度相同，默认将除了配对的都视为负样本
        scores = self.sim(im, s)
        # 二维数组取出对角线上的元素再调整size为size(0)行1列
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores) # 将diagonal扩展成scores一样的维度（重复）
        d2 = diagonal.t().expand_as(scores) 

        # compare every diagonal score to scores in its column 计算scores每一列与其对角线的差值
        # h+r, t-
        # clamp(min=0)和F.relu(x)的区别：x=0时，clamp导数为1，relu为0
        cost_s = (self.margin + scores - d1).clamp(min=0) 
        # compare every diagonal score to scores in its row 计算scores每一行与其对角线的差值
        # (h+r)-, t
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # print(scores)
        # print(scores - d1)
        # print(cost_s)
        # print(cost_im)

        # # clear diagonals
        # mask = torch.eye(scores.size(0)) > .5
        # I = mask
        # if torch.cuda.is_available():
        #     I = I.cuda()
        # cost_s = cost_s.masked_fill_(I, 0)
        # cost_im = cost_im.masked_fill_(I, 0)

        # another mask method 将对角线归0
        mask1 = scores.eq(d1).cuda()
        mask2 = mask1.t()
        cost_s = cost_s.masked_fill_(mask1, 0) #将mask中为1的位置，在cost_S同样位置替换为0
        cost_im = cost_im.masked_fill_(mask2, 0)
        # 以上整个过程得到（self.margin+scores-d1）且对角线为0

        # print(cost_s)
        # print(cost_im)
        # raise Exception(cost_s.sum() + cost_im.sum())

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()
