import numpy as np
bitrate = 0
def student_entrypoint(Measured_Bandwidth, Previous_Throughput, Buffer_Occupancy, Available_Bitrates, Video_Time, Chunk, Rebuffering_Time, Preferred_Bitrate ):
    #student can do whatever they want from here going forward
    global bitrate
    R_i = list(Available_Bitrates.items())
    R_i.sort(key=lambda tup: tup[1] , reverse=False) # Sort chunk size in ascending order
    bitrate = BOLA(prev_bitrate=bitrate, buffer_occupancy=Buffer_Occupancy, 
                   available_bitrates_sorted=R_i, chunk_args=Chunk, 
                   video_time=Video_Time, prev_band=Previous_Throughput)
    # bitrate = BOLA_BASIC(Buffer_Occupancy, R_i, Chunk)
    
    # print(Buffer_Occupancy)
    return bitrate

# helper function: adopt logarithmic utility function to calculate utility for the input bitrate
def calculate_utility(target_chunk, min_chunk):
    v = np.log(target_chunk/min_chunk)
    return v
# helper function: get bitrate of the previous chunk
def get_prev_bitrate(value, list_of_list): 
    for e in list_of_list:
        if value == int(e[0]):
            return e
    value = max(int(i[0]) for i in list_of_list)
    for e in list_of_list:
        if value == int(e[0]):
            return e
# BOLA-FINITE algorithm: an adaptation of BOLA-BASIC
def BOLA(prev_bitrate, buffer_occupancy, available_bitrates_sorted, chunk_args, video_time, prev_band, V_flag = 0):
    p = chunk_args["time"] # chunk time (sec)
    # calculate remaining video time
    remaining_time = buffer_occupancy["time"] + chunk_args["left"] * p
    t = min(video_time, remaining_time)
    t_prime = max(t/2, 3*p)
    # calculate average chunk size in the current manifest (Byte)
    average_chunk_size = sum(tup[1] for tup in available_bitrates_sorted)/len(available_bitrates_sorted)
    # calculate the approximated maximum buffer size
    Q_max = p*buffer_occupancy["size"]/average_chunk_size
    # calculate the dynamic buffer size 
    Q_max_D = min(Q_max, t_prime/p) 
    gamma = 5.0/p # set weight parameter γ for prioritizing utility versus smoothness


    min_chunk = available_bitrates_sorted[0][1]
    S = [] # list of chunk size (bit)
    v = [] # list of utility

    # calculate chunk sizes and utility
    for tup in available_bitrates_sorted:
        S.append(tup[1] * 8)
        v_m = calculate_utility(tup[1], min_chunk)
        v.append(v_m)

    # calculate the dynamic parameter V based on Q_max_D, v_max, γ and p
    V = (Q_max_D - 1) * 1.75/ (v[-1] + gamma * p)
    if V_flag:
        V = 1
    # current buffer occupancy (sec)
    Q = buffer_occupancy["time"]
    objectives = [] # list of values of objective functions corresponding to index m
    for m in range(0, len(S)):
        obj = (V * v[m] + V * gamma * p - Q) / S[m]
        objectives.append(obj)
    
    # find optimal solution to the Lyapunov optimization as the currently selected bitrate
    m = np.argmax(objectives)
    curr_rate = int(available_bitrates_sorted[m][0])
    # find the bitrate of the previous download
    prev_rate = int(get_prev_bitrate(prev_bitrate, available_bitrates_sorted)[0])
    # BOLA-O: reduce bitrate oscillations
    if curr_rate > prev_rate:
        m_list = []
        # check if the higher bitrate (curr_rate) is sustainable
        r = prev_band
        for m in range(0, len(S)):
            if S[m]/p <= max(r, S[0]/p):
                m_list.append(m)
        i = max(m_list)
        m_prime = int(available_bitrates_sorted[i][0])
        if m_prime >= curr_rate:
            m_prime = curr_rate
        # avoid dropping down to a lower level than the previous download
        elif m_prime < prev_rate:
            m_prime = prev_rate
    else:
        m_prime = curr_rate
    # find the optimal bitrate
    selected_bitrate = m_prime
    # print("Bitrate", selected_bitrate)
    return selected_bitrate

# BOLA-BASIC algorithm (under the assumption that the videos are infinite)
def BOLA_BASIC(buffer_occupancy, available_bitrates_sorted, chunk_args):
    p = chunk_args["time"] # chunk time (sec)
    # calculate average chunk size in the current manifest (Byte)
    average_chunk_size = sum(tup[1] for tup in available_bitrates_sorted)/len(available_bitrates_sorted)
    # calculate the approximated maximum buffer size in the unit of seconds
    Q_max = p * buffer_occupancy["size"]/average_chunk_size
    # print("Q_max", p, buffer_occupancy["size"], average_chunk_size, Q_max)
    gamma = 5.0/p # set weight parameter γ for prioritizing utility versus smoothness


    min_chunk = available_bitrates_sorted[0][1]
    S = [] # list of chunk size (bit)
    v = [] # list of utility

    for tup in available_bitrates_sorted:
        S.append(tup[1] * 8)
        v_m = calculate_utility(tup[1], min_chunk)
        v.append(v_m)

    # calculate parameter V based on Q_max, v_max, γ and p
    V = (Q_max - 1) * 1.75/ (v[-1] + gamma * p)
    # print("V",V)
    # current buffer occupancy (sec)
    Q = buffer_occupancy["time"]
    objectives = [] # list of values of objective functions corresponding to index m
    for m in range(0, len(S)):
        obj = (V * v[m] + V * gamma * p - Q) / S[m]
        objectives.append(obj)
    
    # print(objectives)
    m = np.argmax(objectives)
    selected_bitrate = available_bitrates_sorted[m][0]
    print("Bitrate", selected_bitrate)
    return selected_bitrate
