// Array of objects containing questions and their corresponding source codes
const questions = [
    {
      question: "Aim:water jug",
      code: `
  import random
e={}
l=[0,1]
c=0
loc=['a','b']
e.update({'a':random.choice(l)})
e.update({'b':random.choice(l)})
i=random.choice(loc)
print(e)
print("vacum is at location",i)
if(i=='a'):
    if(e.get('a')==0):
        print("A is clean")
        print("Move to B")
    else:
        print("A is Dirty")
        e.update({'a':0})
        c=c+1
        print("A is clean")
        print("Move to B")
    if(e.get('b')==0):
            print("B is Clean")
    else:
        print("B is dirty")
        e.update({'b':0})
        c=c+1
        print("B is clean")
else:
    if(e.get('b')==0):
            print("B is Clean")
            print("Move to A")
    else:
        print("B is dirty")
        e.update({'b':0})
        c=c+1
        print("B is Clean")
        print("Move to A")
    if(e.get('a')==0):
        print("A is clean")
    else:
        print("A is dirty")
        e.update({'a':0})
        c=c+1
        print("A is clean")
print("environment is clean")
print(e)
print("performance measure:",c)
	`
    },
    {
      question: "Aim: vacum cleaner",
      code: `
 import random
e={}
l=[0,1]
c=0
loc=['a','b']
e.update({'a':random.choice(l)})
e.update({'b':random.choice(l)})
i=random.choice(loc)
print(e)
print("vacum is at location",i)
if(i=='a'):
    if(e.get('a')==0):
        print("A is clean")
        print("Move to B")
    else:
        print("A is Dirty")
        e.update({'a':0})
        c=c+1
        print("A is clean")
        print("Move to B")
    if(e.get('b')==0):
            print("B is Clean")
    else:
        print("B is dirty")
        e.update({'b':0})
        c=c+1
        print("B is clean")
else:
    if(e.get('b')==0):
            print("B is Clean")
            print("Move to A")
    else:
        print("B is dirty")
        e.update({'b':0})
        c=c+1
        print("B is Clean")
        print("Move to A")
    if(e.get('a')==0):
        print("A is clean")
    else:
        print("A is dirty")
        e.update({'a':0})
        c=c+1
        print("A is clean")
print("environment is clean")
print(e)
print("performance measure:",c)
      `
    },
    {
      question: "Aim:8 PUZZLE BFS",
      code: `
 from collections import deque

# Class to represent the state of the puzzle
class PuzzleState:
    def __init__(self, puzzle, moves=0):
        self.puzzle = puzzle
        self.moves = moves

    def __eq__(self, other):
        return self.puzzle == other.puzzle

    def __hash__(self):
        return hash(str(self.puzzle))

    def __str__(self):
        return '\n'.join([' '.join(map(str, row)) for row in self.puzzle])

    def get_blank_position(self):
        for i in range(3):
            for j in range(3):
                if self.puzzle[i][j] == 0:
                    return i, j

    def get_neighbors(self):
        moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Possible movements: right, down, left, up
        i, j = self.get_blank_position()
        neighbors = []
        for move in moves:
            new_i, new_j = i + move[0], j + move[1]
            if 0 <= new_i < 3 and 0 <= new_j < 3:
                new_puzzle = [row[:] for row in self.puzzle]
                new_puzzle[i][j], new_puzzle[new_i][new_j] = new_puzzle[new_i][new_j], new_puzzle[i][j]
                neighbors.append(PuzzleState(new_puzzle, self.moves + 1))
        return neighbors

# Breadth-First Search
def bfs(initial_state, goal_state):
    visited = set()
    queue = deque([initial_state])

    while queue:
        current_state = queue.popleft()
        if current_state == goal_state:
             return current_state.moves, current_state

        visited.add(current_state)
        for neighbor in current_state.get_neighbors():
            if neighbor not in visited:
                queue.append(neighbor)

    return float('inf'), None


initial_puzzle = [        [1, 2, 3],[8, 0, 4],[7, 6, 5]]
goal_puzzle = [
    [2, 8, 3],
    [1, 6, 4],
    [7, 0, 5]
]
  
"""initial_puzzle = [
        [1, 0, 2],
        [4, 5, 3],
        [7, 8, 6]
    ]
  goal_puzzle = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 0]
    ]"""

initial_state = PuzzleState(initial_puzzle)
goal_state = PuzzleState(goal_puzzle)
moves, solution_state = bfs(initial_state, goal_state)
if solution_state:
    print("Solution found in {} moves:".format(moves))
    print(solution_state)
else:
    print("No solution found.")
      `
    },
    {
      question: "Aim: GREEDY BFS",
      code: `
G_m={
     'arad':366,'Bucharest':0,'Craiova':160,
     'Drobeta':242,'Eforie':161,'Fagaras':176,
     'Giurgiu':77,'Hirsova':151,'Iasi':226,
     'Lugoj':244,'Neamt':234,'Oradea':380,
     'Pitesti':100,'Rimnicu Vilcea':193,'Sibiu':253,
     'Timisoara':329,'Urziceni':80,'Vaslui':199,
     'Zerind':374,'mehadia':241
    }
A_m={
     'arad'          :[['Zerind',75],        ['Timisoara',118],     ['Sibiu',140]],
     'Bucharest'     :[['Pitesti',101],      ['Fagaras',211]],  
     'Craiova'       :[['Pitesti',138],      ['Rimnicu Vilcea',146],['Drobeta',120]],
     'Drobeta'       :[['Craiova',120],      ['mehadia',75]],
     'Fagaras'       :[['Bucharest',211],    ['Sibiu',99]],
     'Lugoj'         :[['mehadia',70],       ['Timisoara',111]],
     'Oradea'        :[['Zerind',71],        ['Sibiu',151]],
     'Pitesti'       :[['Rimnicu Vilcea',97],['Bucharest',101]],
     'Rimnicu Vilcea':[['Pitesti',97],       ['Sibiu',80],          ['Craiova',146]],
     'Sibiu'         :[['arad',140],         ['Fagaras',99],        ['Oradea',151]],
     'Timisoara'     :[['arad',118],         ['Lugoj',111]],
     'Zerind'        :[['arad',75],          ['Oradea',71]],
     'mehadia'       :[['Lugoj',70],         ['Drobeta',75]],
     'Neamt'         :[['Iasi',87]],
     'Iasi'          :[['Neamt',87],         ['Vaslui',92]],
     'Vaslui'        :[['Urziceni',142]],
     'Urziceni'      :[['Vaslui',142],       ['Hirsova',98],        ['Bucharest',85]],
     'Hirsova'       :[['Urziceni',98],      ['Eforie',86]],
     'Eforie'        :[['Hirsova',86]]
    }
def find_next(ps):
    k=A_m[ps]
    m=G_m[k[0][0]]
    ns=0
    for i in range(1,len(k)):
        if(G_m[k[i][0]]<m):
            m=G_m[k[i][0]]
            ns=i
    return ns
def find_path():
    i_s=input("enter :")
    print("initial state:",i_s)
    path=[]
    cost=0
    path.append(i_s)
    j=find_next(i_s)
    k=A_m[i_s]
    cost=cost+k[j][1]
    path.append(k[j][0])
    n=len(path)
    i_s = k[j][0]
    while(path[n-1]!= 'Bucharest' ):
            j=find_next(i_s)
            k=A_m[i_s]
            cost=cost+k[j][1]
            i_s = k[j][0]
            path.append(k[j][0])
            n=len(path)
    print(path)
    print("cost :",cost)
find_path()

      `
    },
    {
      question: "Aim: A* ",
      code: `
  G_m={'arad':366,'Bucharest':0,'Craiova':160,
     'Drobeta':242,'Eforie':161,'Fagaras':176,
     'Giurgiu':77,'Hirsova':151,'Iasi':226,
     'Lugoj':244,'Neamt':234,'Oradea':380,
     'Pitesti':100,'Rimnicu Vilcea':193,'Sibiu':253,
     'Timisoara':329,'Urziceni':80,'Vaslui':199,
     'Zerind':374,'mehadia':241
    }
A_m={
     'arad'          :[['Zerind',75],        ['Timisoara',118],     ['Sibiu',140]],
     'Bucharest'     :[['Pitesti',101],      ['Fagaras',211]],  
     'Craiova'       :[['Pitesti',138],      ['Rimnicu Vilcea',146],['Drobeta',120]],
     'Drobeta'       :[['Craiova',120],      ['mehadia',75]],
     'Fagaras'       :[['Bucharest',211],    ['Sibiu',99]],
     'Lugoj'         :[['mehadia',70],       ['Timisoara',111]],
     'Oradea'        :[['Zerind',71],        ['Sibiu',151]],
     'Pitesti'       :[['Rimnicu Vilcea',97],['Bucharest',101]],
     'Rimnicu Vilcea':[['Pitesti',97],       ['Sibiu',80],          ['Craiova',146]],
     'Sibiu'         :[['arad',140],         ['Fagaras',99],        ['Oradea',151]],
     'Timisoara'     :[['arad',118],         ['Lugoj',111]],
     'Zerind'        :[['arad',75],          ['Oradea',71]],
     'mehadia'       :[['Lugoj',70],         ['Drobeta',75]],
     'Neamt'         :[['Iasi',87]],
     'Iasi'          :[['Neamt',87],         ['Vaslui',92]],
     'Vaslui'        :[['Urziceni',142]],
     'Urziceni'      :[['Vaslui',142],       ['Hirsova',98],        ['Bucharest',85]],
     'Hirsova'       :[['Urziceni',98],      ['Eforie',86]],
     'Eforie'        :[['Hirsova',86]]
    }
def find_next(ps,cost):
    k=A_m[ps]
    m=G_m[k[0][0]]+cost
    ns=0
    for i in range(1,len(k)):
        if(G_m[k[i][0]]+cost<m):
            m=G_m[k[i][0]]+cost
            ns=i
    return ns
def find_path():
    i_s=input("enter :")
    print("initial state:",i_s)
    path=[]
    cost=0
    path.append(i_s)
    j=find_next(i_s,cost)
    k=A_m[i_s]
    cost=cost+k[j][1]
    path.append(k[j][0])
    n=len(path)
    i_s = k[j][0]
    while(path[n-1]!= 'Bucharest' ):
            j=find_next(i_s,cost)
            k=A_m[i_s]
            cost=cost+k[j][1]
            i_s = k[j][0]
            path.append(k[j][0])
            n=len(path)
    print(path)
    print("cost :",cost)
find_path()

      `
    },
    {
      question: "Aim: To implement Python Program to calculate Multi Variate Linear Regression.",
      code: `
  import numpy as np
  
  y = []
  x = []
  n1 = int(input("Enter num. of Dependent Variables(Y's): "))
  n2 = int(input("Enter num. of Independent Variables(X's): "))
  
  for i in range(n1):
      y.append(list(map(float, input("Y{}: ".format(i + 1)).split())))
  
  x.append([1] * len(y[0]))
  
  for i in range(n2):
      x.append(list(map(float, input("X{}: ".format(i + 1)).split())))
  
  x = np.array(x)
  x = np.transpose(x)
  y = np.array(y)
  y = np.transpose(y)
  
  beta = np.dot(np.linalg.inv(np.dot(np.transpose(x), x)), np.dot(np.transpose(x), y))
  beta = np.round(beta, decimals=2)
  shape = beta.shape
  
  y1_expression = "Y1 = "
  y2_expression = "Y2 = "
  
  for i in range(shape[0]):
      y1_expression += " {}X{} + ".format(beta[i][0], i + 1)
      y2_expression += " {}X{} + ".format(beta[i][1], i + 1)
  
  print(y1_expression)
  print(y2_expression)
      `
    },
    {
    question: "Aim: Two way anova.",
      code: `
      import pandas as pd
import scipy.stats as se
h=int(input("Enter no. of Blocks: "))
k=int(input("Enter no. of Treatments: "))
t=[]
for i in range(k):
	t.append(list(map(int,input("Enter treatment(s): ").split())))
Ti=Bj=G=RSS=N=0
alpha=float(input("Enter Level of Significance: "))
for i in t:
	Ti+=sum(i)*sum(i)
	G+=sum(i)
for i in range(len(t[0])):
	position_sum = sum(sublist[i] for sublist in t)
	Bj += position_sum ** 2
print("G =",G)
print("ΣTᵢ² =",Ti)
print("ΣBⱼ² =",Bj)
for i in t:
for j in i:
RSS+=j*j
N+=1
CF=round(G*G/N,2)
SST=round(RSS-CF,2)
SStr=round(Ti/h-CF,2)
SSb=round(Bj/k-CF,2)
SSe=round(SST-SStr-SSb,2)
print("1. RSS =",RSS)
print("2. CF =",CF)
print("3. SST =",SST)
print("4. SStr =",SStr)
print("4. SSb =",SSb)
print("6. SSe =",SSe)
Ftr=round((SStr/(k-1))/(SSe/((k-1)*(h-1))),3)
Fb=round((SSb/(h-1))/(SSe/((k-1)*(h-1))),3)
data={
"Source of Variation":["Treatments","Blocks","Error","Total"],
'Sum of Squares': [SStr,SSb,SSe,SST],
'DOF': [k-1,h-1,(h-1)*(k-1),k*h-1],
'Mean Sum of Squares' :[round(SStr/(k-1),2),round(SSb/(h-1),2),round(SSe/((k-1)*(h-1)),2),"-
"],
'Variance Ratio' :["-",Ftr,Fb,"-"]
}
df=pd.DataFrame(data)
df=df.to_string(index=False)
print(df)
if(Ftr>1):
	Ftabtr=round(se.f.ppf(q=1-alpha,dfn=k-1,dfd=(k-1)*(h-1)),2)
else:
	Ftr=round(1/Ftr,3)
	Ftabtr=round(se.f.ppf(q=1-alpha,dfn=(k-1)*(h-1),dfd=k-1),2)
print("\n---Inference related to Treatments---")
print("F calculated value =",Ftr,"\nF table value =",Ftabtr)
if(Ftr<Ftabtr):
	print("Since Fcal.<Ftable value, We accept H₀(tr).")
else:
	print("Since Fcal.>Ftable value, We reject H₀(tr).")
if(Fb>1):
	Ftabb=round(se.f.ppf(q=1-alpha,dfn=h-1,dfd=(k-1)*(h-1)),2)
else:
	Ftr=round(1/Fb,3)
	Ftabb=round(se.f.ppf(q=1-alpha,dfn=(k-1)*(h-1),dfd=h-1),2)
print("\n---Inference related to Blocks---")
print("F calculated value =",Fb,"\nF table value =",Ftabb)
if(Fb<Ftabb):
	print("Since Fcal.<Ftable value, We accept H₀(b).")
else:
	print("Since Fcal.>Ftable value, We reject H₀(b).")
      `
    },
    // Add more questions and source codes as needed
  ];
  
  // Function to generate question content with buttons to copy source code
  function generateQuestionContent() {
    const questionContainer = document.getElementById('questionContainer');
  
    questions.forEach((item, index) => {
      // Create question heading
      const questionHeading = document.createElement('h3');
      questionHeading.textContent = `Question ${index + 1}: ${item.question}`;
      questionContainer.appendChild(questionHeading);
  
      // Create code button
      const codeButton = document.createElement('button');
      codeButton.textContent = `Copy Source Code`;
      codeButton.addEventListener('click', () => {
        copyCode(item.code);
      });
      questionContainer.appendChild(codeButton);
      questionContainer.appendChild(document.createElement('br')); // Line break for spacing
    });
  }
  
  // Function to copy code to clipboard
  function copyCode(code) {
    navigator.clipboard.writeText(code)
      .then(() => {
      })
      .catch(err => {
        console.error('Unable to copy source code:', err);
      });
  }
  
  // Generate question content when the page loads
  document.addEventListener('DOMContentLoaded', generateQuestionContent);
  
