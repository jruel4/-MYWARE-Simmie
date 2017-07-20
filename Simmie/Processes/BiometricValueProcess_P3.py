'''
Asynchronous BV thread should run:
    
    - V* forward prop (predict return of current state)
    
    - T* foward prop (calculate actual reward of new state)
    
    - V* back prop (learn from TD error)
'''