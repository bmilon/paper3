import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.stats import wilcoxon
from sklearn.metrics import pairwise_distances
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


def solve_vrp_ortools(orders, max_batch_size=4, time_limit=5):
    coords = orders[['x','y']].to_numpy()
    dist = pairwise_distances(coords)
    mgr = pywrapcp.RoutingIndexManager(len(dist), len(dist), 0)
    routing = pywrapcp.RoutingModel(mgr)

    def dist_cb(i,j):
        return int(dist[mgr.IndexToNode(i)][mgr.IndexToNode(j)]*1000)

    cb = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(cb)
    routing.AddDimensionWithVehicleCapacity(cb, 0,
        [max_batch_size*1000]*len(dist), True, 'Cap')

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.seconds = time_limit

    sol = routing.SolveWithParameters(params)
    batches = []
    if sol:
        for v in range(len(dist)):
            idx = routing.Start(v)
            route = []
            while not routing.IsEnd(idx):
                route.append(mgr.IndexToNode(idx))
                idx = sol.Value(routing.NextVar(idx))
            if len(route) > 1:
                batches.append(orders.iloc[route])
    return batches

