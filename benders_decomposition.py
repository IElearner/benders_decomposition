#https://blog.csdn.net/yuanwangshui/article/details/123310305?spm=1001.2014.3001.5506
import numpy as np
import gurobipy as gp
from gurobipy import GRB

#子问题的对偶问题
class Subproblem:
    def __init__(self, N, M) -> None:
        self.N, self.M = N, M  #分别对应决策变量维数，N对应简单变量z，M对应复杂变量y
        self.m = gp.Model("subproblem")
        #主问题、子问题的约束都是M+N个，因此对偶问题有M+N个变量
        self.u = self.m.addVars(N + M, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='u')
#约束：A（转置）*u>=c
    def add_constrs(self, A, c):
        self.m.addConstrs(
            #shape函数调取维度，shape[0]调行数，shape[1]调列数
            (gp.quicksum(A[j, i] * self.u[j] for j in range(A.shape[0])) >= c[i]) for i in range(A.shape[1]))
#目标函数：min{(b-By)(转置)*u}
    def set_objective(self, B, b, y):
        #np.dot矩阵乘法函数，再reshape转化成一个列表
        self.p = (b - np.dot(B, y)).reshape(self.N + self.M)
        self.m.setObjective(gp.quicksum(self.p[i] * self.u[i] for i in range(self.N + self.M)), sense=GRB.MINIMIZE)

    def solve(self, flag=0):
        self.m.Params.OutputFlag = flag
        self.m.Params.InfUnbdInfo = 1
        self.m.optimize()
#Status模型当前优化状态，UnbodRay即无界射线（无界时得出极射线）
    def get_status(self):
        if self.m.Status == GRB.Status.UNBOUNDED or self.m.Status == GRB.Status.INF_OR_UNBD:
            return np.array([x.getAttr('UnbdRay') for x in self.m.getVars()]), self.m.Status
        elif self.m.Status == GRB.Status.OPTIMAL:
            return self.get_solution(), self.m.Status
        else:
            return None

    def get_solution(self):
        return np.array([self.m.getVars()[i].x for i in range(self.M + self.N)])

    def get_objval(self):
        return self.m.ObjVal if self.m.Status == GRB.Status.OPTIMAL else -np.inf  #np.inf表示正无穷

    def write(self):
        self.m.write("sub_model.lp")


class Master:
    def __init__(self, N, M, d) -> None:
        self.N, self.M, self.d = M, N, d
        self.m = gp.Model("Master")
        self.y = self.m.addVar(lb=0.0, ub=GRB.INFINITY, vtype=GRB.INTEGER, name='y')
        #z=C(zhuzhi)*x
        self.z = self.m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='z')
#benders可行切（u为极射线）
    def add_cut1(self, B, b, u):
        self.p = np.dot(u.T, B)
        self.q = np.dot(u.T, b)
        self.m.addConstr(self.p[0] * self.y <= self.q[0])
#benders最优切（u即极射线）/为啥都用极射线呢，这个不该极点嘛？
    def add_cut2(self, B, b, u):
        self.p = np.dot(u.T, B)
        self.q = np.dot(u.T, b)
        self.m.addConstr(self.z <= self.q[0] - self.p[0] * self.y)
#目标函数即d(zhuanzhi)*y+z
    def set_objective(self):
        self.m.setObjective(self.z + self.d * self.y, sense=GRB.MAXIMIZE)

    def solve(self, flag=0):
        self.m.Params.OutputFlag = flag
        self.m.optimize()

    def get_solution(self):
        return self.m.getVars()[0].x

    def get_objval(self):
        return self.m.ObjVal if self.m.Status == GRB.Status.OPTIMAL else np.inf

    def write(self):
        self.m.write("master_model.lp")


N, M = 10, 1
c, d = np.array([1 + 0.01 * i for i in range(1, N + 1)]), 1.045
A, B = np.vstack((np.ones((1, N)), np.eye(N))), np.array([1 if i == 0 else 0 for i in range(N + 1)]).reshape(N + 1, 1)
b = np.array([1000 if i == 0 else 100 for i in range(N + 1)]).reshape(N + 1, 1)

ub, lb = np.inf, -np.inf
MAX_ITER_TIMES, eps = 10, 0.1 #最大迭代次数与误差

subproblem = Subproblem(N, M)
subproblem.add_constrs(A, c)
masterproblem = Master(N, M, d)
masterproblem.set_objective()
y = 1500   #设置y初始值

for i in range(MAX_ITER_TIMES):
    if ub - lb <= eps:  #这个上下界判别式就是判别对偶值是否约等于原问题简单变量目标函数值
        break
    subproblem.set_objective(B, b, y)
    subproblem.solve()
    subproblem.write()
    rays, solution_status = subproblem.get_status()

    if solution_status == GRB.Status.UNBOUNDED or solution_status == GRB.Status.INF_OR_UNBD:
        masterproblem.add_cut1(B, b, u=rays)
    if solution_status == GRB.Status.OPTIMAL:
        masterproblem.add_cut2(B, b, u=rays)
        lb = max(lb, subproblem.get_objval() + d * y) #更新下界

    masterproblem.solve()
    masterproblem.write()
    y = masterproblem.get_solution()
    ub = masterproblem.get_objval()                  #更新上界
    print("lb: {},  ub: {}, y: {}".format(lb, ub, y))
