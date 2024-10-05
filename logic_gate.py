import numpy as np 


def and_gate(x, y):
    ipt = np.array([x, y])
    w1, w2, theta = 1, 1, 1 
    weights = np.array([w1, w2])
    return 1 if np.sum(ipt*weights) > theta else 0 
    
    
def not_and_gate(x, y):
    return 0 if and_gate(x, y) else 1 
    
    
def or_gate(x, y):
    ipt = np.array([x, y])
    w1, w2, theta = 1, 1, 0
    weights = np.array([w1, w2])
    return 1 if np.sum(ipt*weights) > theta else 0
    
    
def xor_gate(x, y):
    r1 = not_and_gate(x, y)
    r2 = or_gate(x, y)
    return and_gate(r1, r2)
    
    
if __name__ == '__main__':
    print(and_gate(0, 0))
    print(and_gate(1, 0))
    print(and_gate(0, 1))
    print(and_gate(1, 1))
    print(not_and_gate(0, 0))
    print(not_and_gate(1, 0))
    print(not_and_gate(0, 1))
    print(not_and_gate(1, 1))
    print(or_gate(0, 0))
    print(or_gate(1, 0))
    print(or_gate(0, 1))
    print(or_gate(1, 1))
    print(xor_gate(0, 0))
    print(xor_gate(1, 0))
    print(xor_gate(0, 1))
    print(xor_gate(1, 1))