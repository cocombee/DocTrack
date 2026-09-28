"""Solve bounded relative translations, fixing unaffected ink first."""
import numpy as np


def infeasible_cycle(bounds, count):
    """Return a real negative constraint cycle for a refused solve."""
    edges = []
    for (a, b), (lower, upper) in bounds.items():
        edges.extend([(a + 1, b + 1, upper), (b + 1, a + 1, -lower)])
    distances = np.zeros(count + 1)
    predecessors = [None] * (count + 1)
    changed = None
    for _ in range(count + 1):
        changed = None
        for a, b, limit in edges:
            if distances[b] > distances[a] + limit + 1e-12:
                distances[b] = distances[a] + limit
                predecessors[b] = (a, b, float(limit))
                changed = b
        if changed is None:
            return []
    node = changed
    for _ in range(count + 1):
        node = predecessors[node][0]
    start = node
    cycle = []
    while True:
        a, b, limit = predecessors[node]
        cycle.append({'from_owner': a - 1, 'to_owner': b - 1,
                      'maximum_relative_shift': limit})
        node = a
        if node == start:
            break
    return list(reversed(cycle))


def ordered_shifts(bounds, count, priority=None):
    # Index zero is the stationary page; variable i uses index i+1.
    distances=np.full((count+1,count+1),np.inf)
    np.fill_diagonal(distances,0)
    for (a,b),(lower,upper) in bounds.items():
        distances[a+1,b+1]=min(distances[a+1,b+1],upper)
        distances[b+1,a+1]=min(distances[b+1,a+1],-lower)
    for k in range(count+1):
        distances=np.minimum(distances,distances[:,k,None]+distances[None,k,:])
    if np.diag(distances).min() < -1e-9:
        return None
    result=np.zeros(count)
    for variable in range(count) if priority is None else priority:
        node=variable+1
        low,high=-distances[node,0],distances[0,node]
        if low > high+1e-9:
            return None
        value=float(np.clip(0,low,max(low,high)))
        result[variable]=value
        # Fix this translation without invalidating the feasible remaining
        # intervals. These two edge updates retain shortest-path closure.
        distances=np.minimum(distances,distances[:,0,None]+value+distances[None,node,:])
        distances=np.minimum(distances,distances[:,node,None]-value+distances[None,0,:])
    for (a,b),(lower,upper) in bounds.items():
        value=(result[b] if b>=0 else 0)-(result[a] if a>=0 else 0)
        if value < lower-1e-8 or value > upper+1e-8:
            return None
    return result
