import numpy as np

np.random.seed(12)


class cell:
    def __init__(self,cell_name:str,init_queue=None,level_learn_rate = 1e-3,down_learn_rate = 1e-3):
        self.cell_name = cell_name

        # 激活电位
        self.level = 1
        # 消散速度
        self.down = np.random.rand()

        self.now_val = 0

        if init_queue is None:
            self.queue = [[],[]]
        else:
            self.queue = [[],init_queue.copy()]
        self.neighbors = []

        self.is_fired = False

        self.out_sum = 0

        self.level_learn_rate = level_learn_rate

        self.down_learn_rate = down_learn_rate

    def flash_queue(self):
        self.queue[0],self.queue[1] = self.queue[1],[]

    def update(self):
        self.now_val = max(self.now_val-self.down,0)
        for i in self.queue[0]:
            self.now_val += i

        # self.now_val = np.max(self.now_val-self.down,0)

        self.is_fired = self.now_val >= self.level

        self.level -= self.now_val * self.level_learn_rate

        if self.is_fired: # 若成功点火
            self.fire()
            self.now_val = 0

        print(f"{self.cell_name} - {self.is_fired} -")

        return self.is_fired

    # 该函数用于向所有邻居发送点火信号
    def fire(self):
        for i in self.neighbors:
            i.transmission()

    def set_neighbor(self,neighbor:list):
        self.neighbors = neighbor

    def pain_learn(self,pain_val):
        out_sum = 0

        for i in self.neighbors:
            out_sum += i.freq
            i.pain_update(pain_val)

        if len(self.neighbors) == 0:
            out_sum = pain_val
        else:
            out_sum = out_sum /len(self.neighbors) *pain_val

        self.level +=  out_sum * self.level_learn_rate
        self.down += out_sum *  self.down_learn_rate


'''
to_edge为边类，负责维护边的终点以及边的权重
'''
class to_edge:
    def __init__(self,to:cell,weight=None,freq=0,alpha=0.9,beta=1e-3,weight_learn_rate = 1e-3):
        '''
        :param to:  destination of the edge
        :param weight: the weight of the edge
        :param freq: the powered value of history
        '''
        self.cell_to = to

        if weight is None:
            self.weight = np.random.rand()
        else:
            self.weight = weight

        self.freq = freq
        self.alpha = alpha
        self.beta = beta
        self.weight_learn_rate = weight_learn_rate

    def update_weight(self):
        self.weight += self.beta * self.weight

    def transmission(self):  # 在点火时调用该函数
        val = self.weight

        self.cell_to.queue[1].append(val)

        self.freq = self.alpha*self.freq + (1-self.alpha)*val

        self.update_weight()

    def pain_update(self,pain_value):
        self.weight -= self.weight_learn_rate * pain_value * self.freq


class net():
    def __init__(self):
        self.step_num = None

        self.input_1 = None
        self.input_0 = None

        self.target = None

        self.cell_list = [cell(f"cell_{i}") for i in range(3)]

        self.out_cell = cell("cell_out")  # cell_out

        self.cell_list[0].set_neighbor([to_edge(self.cell_list[1]),to_edge(self.cell_list[2])])

        self.cell_list[1].set_neighbor([to_edge(self.cell_list[0]),to_edge(self.cell_list[2])])

        self.cell_list[2].set_neighbor([to_edge(self.out_cell,weight=1)])

    def income(self,step):
        '''
        该函数用于从呼入端获取信号，并将其传入网络的输入层中
        :param step: 当前时间步
        '''
        if step is None:
            self.cell_list[0].queue[1].append(0)
            self.cell_list[1].queue[1].append(0)
        else:
            self.cell_list[0].queue[1].append(self.input_0[step])
            self.cell_list[1].queue[1].append(self.input_1[step])

    def flash_queue(self):
        '''
        该函数用于重置所有细胞的消息队列
        '''
        self.out_cell.flash_queue()
        for i in self.cell_list:
            i.flash_queue()

    def pain_study(self,learn_direct):
        for i in self.cell_list:
            i.pain_learn(learn_direct)

        self.out_cell.pain_learn(learn_direct)


    def step_count(self,step):

        self.income(step)

        self.flash_queue()

        for i in self.cell_list:
            i.update()

        out = self.out_cell.update()

        print("\n\n")

        return out

    def train(self,input_data,target,later_step=2,epochs=100):
        assert len(input_data[0]) == len(target)

        self.input_0 = input_data[0]
        self.input_1 = input_data[1]
        self.target = target

        self.step_num = len(self.input_0)
        input_step = 0
        target_step = 0

        for i in range(later_step):
            out = self.step_count(input_step)

            if out != 0:
                self.pain_study(out)
            input_step = (input_step + 1) % self.step_num

        for epoch in range(epochs):
            print(f"\n\nepoch : {epoch}")
            for _ in range(self.step_num):

                out = self.step_count(input_step)

                if out is not self.target[target_step]:
                    self.pain_study(out-self.target[target_step])

                input_step = (input_step + 1) % self.step_num
                target_step = (target_step + 1) % self.step_num

            for i in range(later_step):
                out = self.step_count(None)

                if out != self.target[target_step]:
                    self.pain_study(target[target_step])
                target_step = (target_step + 1) % self.step_num









if __name__ == "__main__":
    input_data = [[1,1,0,0,0, 1,1,1,0,0, 1,1,0,0,0, 1,1,1,0,0,],
                  [1,1,1,0,0, 1,1,0,0,0, 1,1,0,0,0, 1,1,1,0,0,]]
    a = net()
    a.train(input_data,target=[1,1,1,0,0, 1,1,1,0,0, 1,1,0,0,0, 1,1,1,0,0 ])



