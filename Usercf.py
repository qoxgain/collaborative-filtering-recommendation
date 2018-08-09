import numpy as np
import scipy.stats
import scipy.spatial


class Usercf():
    def __init__(self):
        self.users = 0  # 数据集中用户数
        self.items = 0  # 数据集中产品数

        self.N = 10  # 推荐列表的长度

        self.data = np.array([])  # 总体数据，用户，产品，评分
        self.trainset = np.array([])  # 训练集
        self.testset = np.array([])  # 测试集

        self.similar_matrix = np.array([])  # 用户之间的相似度矩阵
        self.trainrating = np.array([])  # 训练集评分矩阵
        self.testrating = np.array([])  # 测试集评分矩阵
        self.prerating = np.array([])  # 算法预测的评分矩阵

        self.recomendation = np.array([])  # 算法为每个用户生成的推荐列表

        print('Initialization finished!')

    def load_file(self, filename):  # 读取文件，存储为data数据
        f = open(filename, 'r')
        data = []
        for row in f:
            r = row.split(',')
            e = [int(r[0]), int(r[1]), int(r[2])]
            data.append(e)
        self.data = np.array(data)
        print('Loading data findshed!')

    def get_data(self):  # 划分训练集和测试集，获取训练集评分矩阵和测试集评分矩阵
        train = []
        test = []

        for i in self.data:
            if np.random.random() < 0.8:
                train.append(i)
            else:
                test.append(i)
        self.trainset = np.array(train)  # 划分训练集
        self.testset = np.array(test)  # 划分测试集
        print('Split trainset and testset success!')

        self.users = max(self.data[:, 0])  # 总用户数
        self.items = max(self.data[:, 1])  # 总产品数

        self.trainrating = np.zeros((self.users + 1, self.items + 1))  # 训练集的评分矩阵
        self.testrating = np.zeros((self.users + 1, self.items + 1))  # 测试集的评分矩阵
        for i in self.trainset:
            self.trainrating[i[0]][i[1]] = i[2]
        for j in self.testset:
            self.testrating[j[0]][j[1]] = j[2]

        print(self.users, self.items)

    def similar(self, u1, u2):  # 计算用两个用户之间的相似度
        similar = 0

        user1 = self.trainset[self.trainset[:, 0] == u1]  # 获取训练集中用户的评分数据
        user2 = self.trainset[self.trainset[:, 0] == u2]
        items = list(set(user1[:, 1]).intersection(set(user2[:, 1])))  # 获取两个用户共同评分的产品

        if len(items) > 0:  # 用户对共同评分产品的评分组成的向量，利用两个用户的评分向量计算余弦相似度
            user1_rating = []
            user2_rating = []

            for i in items:
                temp1 = user1[user1[:, 1] == i]  # 用户1的评分数据
                temp2 = user2[user2[:, 1] == i]  # 用户2的评分数据
                user1_rating.append(temp1[:, 2])  # 用户1的评分向量
                user2_rating.append(temp2[:, 2])  # 用户2的评分向量

            similar = 1 - scipy.spatial.distance.cosine(user1_rating, user2_rating)  # 利用余弦计算相似度

        return similar


    def simatrix(self):  # 计算用户相似度矩阵
        print('Computing user similar matrix')
        self.similar_matrix = np.zeros((self.users + 1, self.users + 1))
        for user1 in np.arange(self.users) + 1:
            for user2 in np.arange(self.users) + 1:
                self.similar_matrix[user1][user2] = Usercf.similar(self, user1, user2)  # 调用以上的相似度函数计算用户之间的相似度，循环计算得到用户相似度矩阵，是一个对称矩阵
                self.similar_matrix[user2][user1] = self.similar_matrix[user1][user2]

    def predrating(self, u, v):  # 预测用户u对产品v的评分，对评分产品v的所有用户与用户u之间的相似度进行加权求和
        rated = self.trainset[self.trainset[:, 1] == v]  # 评分产品v的用户数据
        rateduser = rated[:, 0]
        pred = 0
        similar_sum = 0
        for user in rateduser:
            similar = self.similar_matrix[u][user]
            rating = self.trainrating[user][v]
            pred = pred + similar * rating
            similar_sum = similar_sum + similar

        if similar_sum == 0:
            preding = 0
        else:
            preding = pred / similar_sum
        return preding

    def predict(self):  # 计算评分预测矩阵
        print('Predicting ratings')

        self.prerating = np.zeros((self.users + 1, self.items + 1))
        for user in np.arange(self.users) + 1:
              for item in np.arange(self.items) + 1:
                if self.trainrating[user][item] == 0:
                    self.prerating[user][item] = Usercf.predrating(self, user, item)


    def recommend(self):  # 为每个用户生成推荐列表
        print('Generate recommendation')
        self.recomendation = np.zeros((self.users + 1, self.N))
        for user in np.arange(self.users) + 1:
            recommenditems = []
            user_rated = (self.trainset[self.trainset[:, 0] == user])[:, 1]  # 训练集中用户评分的产品
            user_rating = self.prerating[user]  # 用户对所有产品的预测评分
            allitems = np.argsort(user_rating, axis=-1)  # 用户预测评分的所有产品列表
            sort_items = allitems[::-1]  # 对产品的预测评分倒序排序
            for item in sort_items:
                if item not in user_rated:
                    recommenditems.append(item)
                if len(recommenditems) > self.N - 1:
                    break
            self.recomendation[user] = recommenditems  # 每个用户的推荐列表


    def evaluate(self):  # 评测方法，rmse,precision,recall,coverage
        print('evaluating')
        testnonzero = np.nonzero(self.testrating)  # 测试集评分矩阵中非0的数据
        error = self.prerating - self.testrating  # 预测评分矩阵与测试集评分的误差
        error_value = error[testnonzero]
        rmse = np.dot(error_value, error_value) / len(error_value)  # RMSE
        print(rmse)

        hit = 0  # 推荐列表的产品和测试集的产品重合的数量，用来计算推荐的精度
        pre = 0  # 推荐列表的总长度
        rec = 0  # 所有推荐的产品的总数量
        all_rec_movies = set()  # 推荐列表中不重复的所有产品，用来计算覆盖率

        for user in np.arange(self.users) + 1:
            testitems = (self.testset[self.testset[:, 0] == user])[:, 1]  # 测试集中用户评分的产品
            if len(testitems) > 0:
                for item in self.recomendation[user]:  # 推荐列表的产品是否在测试集用户评分的产品中
                    if item in testitems:
                        hit += 1
                    all_rec_movies.add(item)
                pre = pre + self.N
                rec = rec + len(testitems)
        precision = hit / pre  # 准确率
        recall = hit / rec  # 召回率
        coverage = len(all_rec_movies) / self.items  # 覆盖率

        print(precision, recall, coverage)

        value = [rmse, precision, recall, coverage]
        return value


if __name__ == '__main__':
    rating_file = 'G:/Python/Workspace/tensor/ratings.csv'
    userCF = Usercf()
    userCF.load_file(rating_file)
    userCF.get_data()
    userCF.simatrix()
    userCF.predict()
    userCF.recommend()
    userCF.evaluate()