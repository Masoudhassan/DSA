def print_items(n):
    for i in range(n):
     for j in range(n):
        print(i,j)  
    for k in range(n):
       print (k)     
print_items (10)
"""drop non dominants so it will still be O(n^2)"""