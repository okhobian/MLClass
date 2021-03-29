import numpy as np

#######################################################
table = """ |1|1.1764|4.2409|0.9750|1|
            |2|1.0400|3.8676|0.4243|1|
            |3|1.0979|1.0227|0.4484|1|
            |4|2.0411|4.7610|0.6668|-1|
            |5|2.0144|4.1217|1.2470|-1|
            |6|2.1454|4.4439|0.3974|-1|  """

lambdas = np.array([1,0.7383,0,0.0411,1,0.6972])
#######################################################

def extract(table):
    lines = table.split('\n')
    l = []
    for line in lines:
        elements = line.split('|')
        elements = [ a for a in elements if a.strip() ]
        l.append(elements)
    
    np_table = np.array(l)
    x = np_table[:,1:-1]
    y = np_table[:,-1]
    return x, y, len(lines)

def calculate_w(k, x, y, lamda):
    x = x.astype(np.float)
    w = []
    for i in range(k):
        temp = float(y[i]) * float(lamda[i])
        w.append(temp * x[i])
    w = np.array(w)
    return w.sum(axis=0)
    
if __name__ == '__main__':
    x, y, k = extract(table)
    w = calculate_w(k, x, y, lambdas)
    print(w)
