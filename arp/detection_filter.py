import numpy as np
from detectron.utils.vis import lane_wid, get_parabola_by_distance
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

    #remove duplicate line
    # line_list = np.array(line_list)
    line_list = sorted(line_list, key=lambda k:k['x'])
    print ("source line_list:" + str(line_list))

    if line_list[0]['type'] == 'boundary':
        # line_list = line_list[line_list[1:]['x'] - line_list[0]['x'] > WID_LANE / 2]
        filter_list = [line_list[0]]
        for line in line_list[1:]:
            if line['x'] - line_list[0]['x'] < WID_LANE / 2:
                filter_list[0]['x'] = line['x']
            else:
                filter_list.append(line)
        line_list = filter_list
    if line_list[-1]['type'] == 'boundary':
        # line_list = line_list[line_list[-1]['x'] - line_list[0:-1]['x'] > WID_LANE / 2]
        filter_list = [line_list[-1]]
        for line in line_list[0:-1][::-1]:
            if line_list[-1]['x'] - line['x'] < WID_LANE / 2:
                filter_list[-1]['x'] = line['x']
            else:
                filter_list.append(line)
        line_list = filter_list

    line_list = sorted(line_list, key=lambda k:k['x'])
    no_nest = []
    for index, value in enumerate(line_list):
        if index == 0:
            no_nest.append(value)
            continue
        if value['x'] - no_nest[-1]['x'] < WID_LANE / 4:
            if len(no_nest) > 1:
                distance1 = value['x'] - no_nest[-2]['x']
                distance2 = no_nest[-1]['x'] - no_nest[-2]['x']
                if abs(WID_LANE - distance1) < abs(WID_LANE - distance2):
                    no_nest[-1] = value
        else:
            no_nest.append(value)


    if len(cache_list) == 0:
        cache_list = no_nest
    else:
        line_list = no_nest
        match_id_array = []
        add_line = []
        for line in line_list:
            match_index = -1
            distance_min = WID_LANE / 4
            for index, cache_line in enumerate(cache_list):
                distance = abs(line['x'] - cache_line['x'])
                if distance < distance_min:
                    match_index = index
                    distance_min = distance
            if match_index >= 0:
                match_id_array.append(match_index)
                cache_list[match_index]['x'] = LINE_MOVE_WEIGHT * cache_list[match_index]['x'] + (1 - LINE_MOVE_WEIGHT) * line['x']
                cache_list[match_index]['score'] = LINE_SCORE_WEIGHT * cache_list[match_index]['score'] + (1 - LINE_SCORE_WEIGHT) * line['score']
                cache_list[match_index]['curve_param'] = line['curve_param']
            else:
                line['score'] = SCORE_DEFAULT
                add_line.append(line)
        for id in range(len(cache_list)):
            if not id in match_id_array:
                cache_list[id]['score'] = LINE_SCORE_WEIGHT * cache_list[id]['score'] + (1 - LINE_SCORE_WEIGHT) * (SCORE_DEFAULT /2)
                if frameId == 4840:
                    pass
                for index in range(len(cache_list)):
                    if (id + index) in match_id_array:
                        cache_list[id]['curve_param'] = get_parabola_by_distance(cache_list[id + index]['curve_param'],
                                                                                 cache_list[id]['x'] -
                                                                                 cache_list[id + index]['x'])
                        break
                    elif (id - index) in match_id_array:
                        cache_list[id]['curve_param'] = get_parabola_by_distance(cache_list[id - index]['curve_param'],
                                                                                 cache_list[id]['x'] -
                                                                                 cache_list[id - index]['x'])
                        break

        cache_list.extend(add_line)

    cache_list = sorted(cache_list, key=lambda k: k['x'])
    cache_list = np.array(cache_list)
    # cache_list = cache_list[cache_list[:]['score'] > 0.1]
    filter_list = []

    score_list = []
    x_list = []
    type_list = []

    #filter by prob trigger
    for line in cache_list:
        if line['score'] > 0.11:
            filter_list.append(line)
        score_list.append("%.2f" % line['score'])
        x_list.append("%.2f" % line['x'])
        type_list.append(line['type'])
    print ("x_list:" + str(x_list))
    print ("score_list:" + str(score_list))
    print ("type_list:" + str(type_list))

    #merge prob by close line
    filter_pos = []
    for line in filter_list:
        if (len(filter_pos) > 0):
            if abs(line['x'] - filter_pos[-1]['x']) < WID_LANE / 4:
                filter_pos[-1]['type'] = line['type'] if line['score'] > filter_pos[-1]['score'] else filter_pos[-1]['type']
                percent = line['score'] / (line['score'] + filter_pos[-1]['score'])
                filter_pos[-1]['x'] = line['x'] * percent + filter_pos[-1]['x']*(1-percent)
                score = line['score'] + filter_pos[-1]['score']
                filter_pos[-1]['score'] = score if score < 1 else 1
            else:
                filter_pos.append(line)
        else:
            filter_pos.append(line)


    filter_pro = []
    distance_log = []
    pre_line = None
    for line in filter_pos:
        if line['score'] > 0.7:
            filter_pro.append(line)
            if not pre_line is None:
                distance_log.append(int(line['x'] - pre_line['x']))
            pre_line = line
    print ("distance_log:" + str(distance_log))
    cache_list = filter_list
    return filter_pro, cache_list
