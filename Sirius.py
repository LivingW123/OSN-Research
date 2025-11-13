from mathfunc import(
    find_closest_factors_positive
)

def Sirius(nodes):
    delta, port = find_closest_factors_positive(nodes)
    source = {}
    for i in range(nodes):
        for j in range(port):
            source[(nodes, port)] = []
            for k in range(delta):
                source[(nodes, port)].append(j+k)
    return source
            
print(Sirius(4))