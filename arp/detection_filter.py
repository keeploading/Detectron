import numpy as np
from detectron.utils.logging import setup_logging
from arp.line_detection import lane_wid, get_parabola_by_distance
#bind with detect spped, later
SCORE_DEFAULT = 0.2
LINE_MOVE_WEIGHT = 0.9
LINE_SCORE_WEIGHT = 0.7
MAX_CACHE = 50
WID_LANE = lane_wid
#[{x, score, type}]
cache_list = []
logger = setup_logging(__name__)
def get_predict_list(line_list, frameId):
    global cache_list
    if frameId == 40:
        pass
    #=========================================================================================
    #remove duplicate line
    #=========================================================================================
    line_list = sorted(line_list, key=lambda k:k['x'])
    print ("source line_list:" + str(line_list))

    x_log = []
    for line in line_list:
        if line['score'] > 0.11:
            x_log.append(int(line['x']))
    logger.info (str(frameId) + " x_log input:" + str(x_log))

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
        if value['x'] - no_near[-1]['x'] < WID_LANE / 3:
            if len(no_near) > 1:
                distance1 = value['x'] - no_near[-2]['x']
                distance2 = no_near[-1]['x'] - no_near[-2]['x']
                if abs(WID_LANE - distance1) < abs(WID_LANE - distance2):
                    no_near[-1] = value
        else:
            no_near.append(value)
    line_list = no_near

    x_log = []
    for line in line_list:
        if line['score'] > 0.11:
            x_log.append(str(int(line['x'])) + "(" + str(int(line['middle'])) + ")")

    x_log_cache = []
    for line in cache_list:
        if line['score'] > 0.11:
            x_log_cache.append(str(int(line['x'])) + "(" + str(int(line['middle'])) + ")")
    logger.info ("x_log_cache:" + str(x_log_cache))
    #=========================================================================================
    #find match line and update score
    #=========================================================================================
    avg_move_list = []
    if len(cache_list) == 0:
        cache_list = line_list
    else:
        match_id_array = []
        add_line = []
        for line in line_list:
            match_index = -1
            distance_min = WID_LANE / 3
            move_cache = []
            for index, cache_line in enumerate(cache_list):
                #use middle instead of x during diff frame compare
                distance = abs(line['middle'] - cache_line['middle'])
                if distance < distance_min:
                    match_index = index
                    move_cache.append(match_index)
                    # distance_min = distance
            if len(move_cache) > 0:
                match_id_array.extend(move_cache)
                for move_id in move_cache:
                    print ("x_log" + str(cache_list[move_id]['x']) + " --> " + str(line['x']))
                    # cache_list[move_id]['x'] += LINE_MOVE_WEIGHT * (line['middle'] - cache_list[move_id]['middle'])
                    x_change = LINE_MOVE_WEIGHT * (line['x'] - cache_list[move_id]['x'])
                    avg_move_list.append(x_change)
                    cache_list[move_id]['x'] += x_change
                    cache_list[move_id]['middle'] += LINE_MOVE_WEIGHT * (line['middle'] - cache_list[move_id]['middle'])
                    cache_list[move_id]['score'] = LINE_SCORE_WEIGHT * line['score'] + (1 - LINE_SCORE_WEIGHT) * cache_list[move_id]['score']
                    # cache_list[move_id]['curve_param'][2] = LINE_MOVE_WEIGHT * line['curve_param'][2] + (1 - LINE_MOVE_WEIGHT) * cache_list[move_id]['curve_param'][2]
                    cache_list[move_id]['curve_param'][2] = cache_list[move_id]['x']
                    cache_list[move_id]['curve_param'][0:2] = line['curve_param'][0:2]
                    if line['score'] > 0.9:
                        cache_list[move_id]['type'] = line['type']
            else:
                line['score'] = SCORE_DEFAULT
                add_line.append(line)
        print ("x_log add_line:" + str(add_line))
        avg_move = np.array(avg_move_list).mean()
        #=========================================================================================
        #param adjust
        #=========================================================================================
        for id in range(len(cache_list)):
            if not id in match_id_array:
                cache_list[id]['score'] = 0.9 * cache_list[id]['score']#(1 - LINE_SCORE_WEIGHT) * cache_list[id]['score']

                for index in range(len(cache_list)):
                    if (id + index) in match_id_array:
                        cache_list[id]['curve_param'] = get_parabola_by_distance(cache_list[id + index]['curve_param'],
                                                                                 cache_list[id]['x'] -
                                                                                 cache_list[id + index]['x'] + avg_move)
                        dalta = cache_list[id]['curve_param'][2] - cache_list[id]['x']
                        cache_list[id]['x'] += dalta
                        cache_list[id]['middle'] += dalta
                        break
                    elif (id - index) in match_id_array:
                        cache_list[id]['curve_param'] = get_parabola_by_distance(cache_list[id - index]['curve_param'],
                                                                                 cache_list[id]['x'] -
                                                                                 cache_list[id - index]['x'])
                        dalta = cache_list[id]['curve_param'][2] - cache_list[id]['x']
                        cache_list[id]['x'] += dalta
                        cache_list[id]['middle'] += dalta
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
    predict_x_log = []
    pre_line = None
    for line in cache_list:
        if line['score'] > 0.6:
            filter_pro.append(line)
            type_log.append(line['type'])
            predict_x_log.append(str(int(line['x'])))
            if not pre_line is None:
                distance_log.append(int(line['x'] - pre_line['x']))
            pre_line = line
    print ("distance_log:" + str(distance_log))
    print ("type_log:" + str(type_log))

    x_log = str(x_log)
    logger.info ("x_log:" + str(x_log) + ((80 - len(x_log)) * " ") + "----> " + str(predict_x_log))
    for l in cache_list:
        if abs(l['curve_param'][2] - l['x']) > 1:
            print ("please check frame :" + str(frameId))
    return filter_pro, cache_list
