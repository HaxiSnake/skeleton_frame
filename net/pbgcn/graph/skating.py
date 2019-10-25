from .graph import Graph

num_node = 25

inward_ori_index = [(9, 2), (2, 1), (1, 16), (1, 17), (16, 18), (17, 19), (2, 3), (2, 6),
                    (3, 4), (4, 5), (6, 7), (7, 8), (9, 10), (10, 11),
                    (11, 12), (12, 25), (12, 23), (23, 24), (9, 13), (13, 14),
                    (14, 15), (15, 22), (15, 20), (20, 21)]
# inward_ori_index = [(8,1), (1, 0), (0, 15), (0, 16), (15, 17), (16, 18), (1, 2), (1, 5),
#                     (2, 3), (3, 4), (5, 6), (6, 7), (8, 9), (9, 10),
#                     (10, 11), (11, 24), (11, 22), (22, 23), (8, 12), (12, 13),
#                     (13, 14), (14, 21), (14, 19), (19, 20)]
inward = [(i-1, j-1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

head = [(0, 1), (1, 2), (1, 5), (0, 15), (0, 16), (15, 17), (16, 18)]
righthand= [(2, 3), (3, 4)]
lefthand= [(5, 6), (6, 7)]
hands = lefthand + righthand
torso = [(1, 2), (1, 5), (1, 8), (8, 9), (8, 12)]
leftleg = [(12, 13), (13, 14), (14, 21), (14, 19), (19, 20)]
rightleg = [(9, 11), (10, 11), (11, 24), (11, 22), (22, 23)]
legs = leftleg + rightleg

class SkatingGraph(Graph):
    def __init__(self,
                 labeling_mode='uniform'):
        super(SkatingGraph, self).__init__(num_node=num_node,
                                      inward=inward,
                                      outward=outward,
                                      parts=[head, hands, torso, legs],
                                      labeling_mode=labeling_mode)
