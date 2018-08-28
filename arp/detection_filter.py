import numpy as np
from arp.line_detection import lane_wid, get_parabola_by_distance
#bind with detect spped, later
SCORE_DEFAULT = 0.2
LINE_MOVE_WEIGHT = 0.3
LINE_SCORE_WEIGHT = 0.9
MAX_CACHE = 50
WID_LANE = lane_wid
#[{x, score, type}]
cache_list = []

def get_predict_list(line_list, frameId):
    global cache_list

    #=========================================================================================
    #remove duplicate line
    #=========================================================================================
    line_list = sorted(line_list, key=lambda k:k['x'])
    print ("source line_list:" + str(line_list))

    if line_list[0]['type'] == 'boundary':
        # line_list = line_list[line_list[1:]['x'] - line_list[0]['x'] > WID_LANE / 2]
        filter_list = [line_list[0]]
        for line in line_list[1:]:
            if line['x'] - line_list[0]['x'] < WID_LANE / 2:
                filter_list[0]['x'] = line['x']
                filter_list[0]['curve_param'][-1] = line['x']
            else:
                filter_list.append(line)
        line_list = filter_list
    if line_list[-1]['type'] == 'boundary':
        # line_list = line_list[line_list[-1]['x'] - line_list[0:-1]['x'] > WID_LANE / 2]
        filter_list = [line_list[-1]]
        for line in line_list[0:-1][::-1]:
            if line_list[-1]['x'] - line['x'] < WID_LANE / 2:
                filter_list[-1]['x'] = line['x']
                filter_list[-1]['curve_param'][-1] = line['x']
            else:
                filter_list.append(line)
        line_list = filter_list

    line_list = sorted(line_list, key=lambda k:k['x'])
    no_near = []
    for index, value in enumerate(line_list):
        if index == 0:
            no_near.append(value)
            continue
        if value['x'] - no_near[-1]['x'] < WID_LANE / 4:
            if len(no_near) > 1:
                distance1 = value['x'] - no_near[-2]['x']
                distance2 = no_near[-1]['x'] - no_near[-2]['x']
                if abs(WID_LANE - distance1) < abs(WID_LANE - distance2):
                    no_near[-1] = value
        else:
            no_near.append(value)
    line_list = no_near

    #=========================================================================================
    #find match line and update score
    #=========================================================================================
    if len(cache_list) == 0:
        cache_list = line_list
    else:
        match_id_array = []
        add_line = []
        for line in line_list:
            match_index = -1
            distance_min = WID_LANE / 4
            move_cache = []
            for index, cache_line in enumerate(cache_list):
                distance = abs(line['x'] - cache_line['x'])
                if distance < distance_min:
                    match_index = index
                    move_cache.append(match_index)
                    # distance_min = distance
            if len(move_cache) > 0:
                match_id_array.extend(move_cache)
                for move_id in move_cache:
                    cache_list[move_id]['x'] = LINE_MOVE_WEIGHT * cache_list[move_id]['x'] + (1 - LINE_MOVE_WEIGHT) * line['x']
                    cache_list[move_id]['score'] = LINE_SCORE_WEIGHT * cache_list[move_id]['score'] + (1 - LINE_SCORE_WEIGHT) * line['score']
                    cache_list[move_id]['curve_param'][2] = LINE_MOVE_WEIGHT * cache_list[move_id]['curve_param'][2] + (1 - LINE_MOVE_WEIGHT) * line['curve_param'][2]
                    cache_list[move_id]['curve_param'][0:2] = line['curve_param'][0:2]
                    if line['score'] > 0.9:
                        cache_list[move_id]['type'] = line['type']
            else:
                line['score'] = SCORE_DEFAULT
                add_line.append(line)
        #=========================================================================================
        #param adjust
        #=========================================================================================
        for id in range(len(cache_list)):
            if not id in match_id_array:
                cache_list[id]['score'] = LINE_SCORE_WEIGHT * cache_list[id]['score'] + (1 - LINE_SCORE_WEIGHT) * (SCORE_DEFAULT /2)

                for index in range(len(cache_list)):
                    if (id + index) in match_id_array:
                        cache_list[id]['curve_param'] = get_parabola_by_distance(cache_list[id + index]['curve_param'],
                                                                                 cache_list[id]['x'] -
                                                                                 cache_list[id + index]['x'])
                        cache_list[id]['x'] = cache_list[id]['curve_param'][2]
                        break
                    elif (id - index) in match_id_array:
                        cache_list[id]['curve_param'] = get_parabola_by_distance(cache_list[id - index]['curve_param'],
                                                                                 cache_list[id]['x'] -
                                                                                 cache_list[id - index]['x'])
                        cache_list[id]['x'] = cache_list[id]['curve_param'][2]
                        break

        cache_list.extend(add_line)

    cache_list = sorted(cache_list, key=lambda k: k['x'])
    cache_list = np.array(cache_list)

    #=========================================================================================
    #filter by prob trigger
    #=========================================================================================
    filter_list = []
    for line in cache_list:
        if line['score'] > 0.11:
            filter_list.append(line)
    cache_list = filter_list

    #log
    score_list = []
    x_list = []
    type_list = []
    #=========================================================================================
    #merge prob by close line
    #=========================================================================================
    filter_pos = []
    for line in cache_list:
        if (len(filter_pos) > 0):
            if abs(line['x'] - filter_pos[-1]['x']) < WID_LANE / 4:
                filter_pos[-1]['type'] = line['type'] if line['score'] > filter_pos[-1]['score'] else filter_pos[-1]['type']
                percent = line['score'] / (line['score'] + filter_pos[-1]['score'])
                adjust_x = line['x'] * percent + filter_pos[-1]['x']*(1-percent)
                filter_pos[-1]['x'] = adjust_x
                filter_pos[-1]['curve_param'][2] = adjust_x
                score = line['score'] + filter_pos[-1]['score']
                filter_pos[-1]['score'] = score if score < LINE_SCORE_WEIGHT else LINE_SCORE_WEIGHT
            else:
                filter_pos.append(line)
        else:
            filter_pos.append(line)

    #     score_list.append("%.2f" % line['score'])
    #     x_list.append("%.2f" % line['x'])
    #     type_list.append(line['type'])
    # print ("x_list:" + str(x_list))
    # print ("score_list:" + str(score_list))
    # print ("type_list:" + str(type_list))

    cache_list = filter_pos

    filter_pro = []
    distance_log = []
    type_log = []
    x_log = []
    pre_line = None
    for line in cache_list:
        if line['score'] > 0.6:
            filter_pro.append(line)
            type_log.append(line['type'])
            x_log.append(line['x'])
            if not pre_line is None:
                distance_log.append(int(line['x'] - pre_line['x']))
            pre_line = line
    print ("distance_log:" + str(distance_log))
    print ("type_log:" + str(type_log))
    print ("x_log:" + str(x_log))
    for l in cache_list:
        if abs(l['curve_param'][2] - l['x']) > 1:
            print ("please check frame :" + str(frameId))
    return filter_pro, cache_list
