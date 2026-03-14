graph={
    'A' :['B','C'],
    'B':['C'],
    'C':['D'],
    'D':[]  
}
heuristic={'A':3,'B':2,'C':1,'D':0} # Estimated cost to reach 'D'
def heuristic_search(start,goal):
    node=start
    while node!=goal:
        print(f"Visting: {node}")
        if not graph[node]:
            break
        node=min(graph[node],key=lambda x:heuristic[x]) #Pick the lowest heuristic value
print("Goal reached!")
heuristic_search('A','D')