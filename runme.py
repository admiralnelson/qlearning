
import numpy as np
import matplotlib.pyplot as plt
import random




class Qlearning(object):
    def __init__(self, data, start, goal, episodes, maxMove, alpha, gamma):
        self.data = np.loadtxt(data)
        self.start = start
        self.goal = goal
        self.episodes = episodes
        self.maxMove = maxMove
        self.QTable = np.zeros((len(self.data) * len(self.data), 4))
        self.alpha = alpha
        self.gamma = gamma
        self.actions = ("N","E","S", "W")

        plt.figure(figsize=(20,5))
        plt.axis('off')
        plt.title("Nilai reward pada tabel berikut, close untuk lanjut")
        self.table = plt.table(cellText=self.data,loc='center', fontsize=1)
        self.table.get_celld()[start].set_color('red')
        self.table.get_celld()[goal].set_color('blue')
        plt.show()
        

    def getNextMove(self, position, act):
        if(act == 4):
            return position
        if self.actions[act] == 'N':
            return (position[0]-1, position[1])
        elif self.actions[act] == 'E':
            return (position[0], position[1]+1)
        elif self.actions[act] == 'S':
            return (position[0]+1, position[1])
        elif self.actions[act] == 'W':
            return (position[0], position[1]-1)
        return position

    def getNextAction(self,position):
        if(position == (0, 0)):
            return random.choice([1, 2])
        elif(position == (0, len(self.data)-1)):
            return 4
            #return random.choice([3, 2])
        elif(position == (len(self.data)-1, len(self.data)-1)):
            return random.choice([0, 3])
        elif(position == (len(self.data)-1, 0)):
            return random.choice([0, 1])
        elif(position[1] == 0):
            return random.choice([0, 2, 1])
        elif(position[1] == len(self.data)-1):
            return random.choice([0, 2, 3])
        elif(position[0] ==  0):
            return random.choice([2, 3, 1])
        elif(position[0] == len(self.data)-1):
            return random.choice([0, 3, 1])
        else:
            return random.choice([0, 1, 2, 3])

    def Train(self):
        for episode in range(self.episodes):
            position = (np.random.randint(len(self.data)), np.random.randint(len(self.data)))
        #     position = start
            for move in range(self.maxMove):
                action = self.getNextAction(position)
                if(action == 4):
                    break
                next_move = self.getNextMove(position, action)
                reward = self.data[next_move]
                id = len(self.data) * next_move[1] + next_move[0]
                id_pos = len(self.data) * position[1] + position[0]
                Qmax = max(self.QTable[id])
                self.QTable[id_pos][action] = self.QTable[id_pos][action] + self.alpha * (reward + self.gamma * Qmax - self.QTable[id_pos][action])
                position = next_move


        self.best = []
        rwd = np.zeros((len(self.data), len(self.data)))
        for i, q in enumerate(self.QTable):
            action = self.actions[np.argmax(q)]
            pos = (i//len(self.data), i%len(self.data))
            rwd[pos] = np.argmax(q)
            self.best.append([pos, action])
    
    def Move(self, index, instruction):
        length = len(self.data)    

        if(instruction == "N"):
            return index - length
        if(instruction == "W"):
            return index - 1
        if(instruction == "E"):
            return index + 1
        if(instruction == "S"):
            return index + length
    
    def indexToTwoD(self, pos):
        if(pos < 0): return (0,0)
        return (pos//len(self.data), pos%len(self.data))

    def twoDtoIndex(self, pos):
        y = pos[1] 
        x = pos[0] 
        return (len(self.data) * y) + x


    def DrawMap(self):
        index = self.twoDtoIndex(self.start)
        step = 0
        CUT_OFF = 10000
        
        plt.figure(figsize=(20,5))
        plt.axis('off')
        plt.title("Hijau = pergerakan si agen, close untuk lanjut")
        table = plt.table(cellText=self.data,loc='center', fontsize=1)
        pos = (0,0)
        while (index > 0 and step < CUT_OFF ):
            pos = self.indexToTwoD(index)
            table.get_celld()[pos].set_color('green')
            index = self.Move(index , self.best[index][1])
            step = step + 1
            print("moving to ",  self.best[index][1], "  index ", index, " pos ", pos )


        plt.show()



    def Draw(self):
        plt.figure(figsize=(20,5))
        plt.axis('off')
        plt.title("Vektor setiap cell, close untuk lanjut")
        table = plt.table(cellText=self.data,loc='center', fontsize=1)
        for b in self.best:
            if(b[1] == 'N'):
                table.get_celld()[b[0]].get_text().set_text("⬆")
            elif(b[1] == 'E'):
                table.get_celld()[b[0]].get_text().set_text("➡")
            elif(b[1] == 'S'):
                table.get_celld()[b[0]].get_text().set_text("⬇")
        #         print(b[0])
            elif(b[1] == 'W'):
                table.get_celld()[b[0]].get_text().set_text("⬅")
        table.get_celld()[self.start].set_color('red')
        table.get_celld()[self.goal].set_color('blue')
        plt.show()
        self.DrawMap()


qlearn = Qlearning("DataTugas3ML2019.txt", (0,14), (14,0), 1500, 100, 0.5, 1.0)
qlearn.Train()
qlearn.Draw()