"""
this is part of Sprint 1 to find maximal repeat sets. 
To improve readability, doc strings and comments were added. 

ByoungKyu Park
"""
import copy


def post_processing_mr (mrs) : 
    mrs=[p.split(',') for p in mrs]
    new_mrs=[]
    for p in mrs:
        if '|' in p : p.remove('|')
        if p not in new_mrs : new_mrs.append(p) 
    new_mrs.sort()
    new_mrs.sort(key= lambda x : len(str(x)))
    return new_mrs
                    

def listSearch(T,keyword, start=0):
    """
    to find the keyword in the given Trace, recursive call makes continution. 

    input   T, keyword, start=0 
    output  list of indices

    """
    try :
        idx = T.index(keyword,start)
    except : 
        return []
    return [idx] + listSearch(T,keyword,idx+1)



def search (L, pl, pr, tl, tr, lm, rm, rep ):
    """
    to find maximal repeat, it check immediate left and immediate right
    if matched, it recursively call this function with the expanded string. 

    input   L = given string, 
            pl = left position of the pattern,
            pr = right position of the pattern, 
            tl = sting's left position to be compared,
            tr = string's right position to be compared,
            lm = if lhs true (immediate left is the same), it expands to the left. , 
            rm = if rhs true (immediate right is hte same), it expands to the right
            rep = # of repetition

    output  pattern
            target
            pattern_location 
    """

    pl = pl-lm
    pr = pr+rm
    tl = tl-lm
    tr = tr+rm
    pattern=[L[pl : pr], pl, pr]
    target=[L[tl :tr], tl, tr]
    pattern_location=[]
    # flag=False 
    # try : 
    #     idx_delimiter = (pattern[0].index("|"))
    #     if idx_delimiter == 0 or idx_delimiter == len() 
    # except : 
    #     idx_delimiter = -1



    flag=True  

    if ("|" in pattern[0]) : 
        idx=pattern[0].index("|")
        if (idx==0 or idx==len(pattern[0])-1) :
            flag=True
        else : 
            flag=False  
    
    if flag : 

        # lhs = check if the immediate left is same,   
        # rhs = immediate right check
        lhs = pattern[0][0] == target[0][0]
        rhs = pattern[0][-1] == target[0][-1]

        if (pattern[0][0]=='|' or target[0][0]=='|') : lhs=False
        if (pattern[0][-1]=='|' or target[0][-1]=='|') : rhs=False


        # print("\t"*rep, pattern, target, lhs, rhs)
        
        
        
        if(lhs or rhs) :
            #print("\t"*rep, "expand to the", "left"* lhs, "right" * rhs)
            pattern, target, pattern_location= search(L, pl,pr,tl,tr,lhs,rhs, rep+1) #recursion call
        else :
            pattern_location=[[pl+1,pr-2],[tl+1,tr-2]]
            #print("\t"*rep, "pattern found", pattern[0][1: -1],pattern_location,"\n")   
    else:
        print("*"*200)

        pattern_location=[[pl,pr],[tl,tr]]
        print(pattern, target, pattern_location)

    return pattern, target, pattern_location
    

    
def discover_maximal_repeat(L) : 
    
    """
    Desc    discover the maximal repeat 
    Used    in pattern_discover()
    Input   L = Log
    output  maximal_repeat_set
    """
    maximal_repeat_set={}
    pattern_locations=[]
    
    # getting the list of the activities by doing set operation 
    A=list(set(L))
    A.sort()
    A.sort(key=len)
    
    # print("set of Activities", A)
    # print("log with paading", L)

    # add paddings to the ends, to compare as per the definition of maximal repeat
    L=[''] + L + ['']
    
    
    # it starts with 
   
    for activity in A :
        idx=listSearch(L,activity,0)
        # print("\nto be started with activity", activity)
        # find idx, the indices for the position the activity
        # print("found activity", activity, "in", idx)
        
        # it repeats |idx| times 
        for l in range(len(idx)) :  #pi = the position of pattern
            indice=idx[l:]
            pl = indice[0] - 1  # pl = pattern -1 to include immediate left 
            pr = indice[0] + 2  # pattern + 1 to include immediate right 
            for ti in indice[1:]: # ti = the position of target 
                tl= ti-1  # adding padding to the immediate left of the target
                tr= ti+2 # adding padding to the immediate right of the target
                pattern, target,pattern_location = search(L, pl, pr, tl, tr, 0,0,0)

                # pattern includes the information on the paddings and  position.  
                # e.g  pattern =  [['', 'a', 'a'], 0, 3], then repeated_activity = 'a'
                repeated_activity=','.join(pattern[0][1:-1])
                # repeated_activity=','.join(pattern[0][1:-1]).replace("|,", "").replace(",|","")

                if repeated_activity == '|' : continue
                # print("repeated_activity",repeated_activity, "pattern",pattern)
                # if not exist, it crates the data set in the dictionary.
                if maximal_repeat_set.get(repeated_activity, False ) == False : 
                    maximal_repeat_set[repeated_activity]={'location' : pattern_location}

                #if already exist in the dic.  
                else : 
                    for location in pattern_location:
                        if location not in maximal_repeat_set[repeated_activity]['location'] : 
                            maximal_repeat_set[repeated_activity]['location'].append(location)
                #print("updated in the dictionary", repeated_activity, maximal_repeat_set[repeated_activity], "\n\n\n\n")
    for activities in maximal_repeat_set : 
        maximal_repeat_set[activities]['length']=activities.count(',')+1
    # return maximal_repeat_set
    return list(maximal_repeat_set.keys())





def discover_super_maximal_repeat(maximal_repeat_set) :
    """
    Desc    discover the super maximal repeat by recursive manner
    Used    in pattern_discover()
    Input   maximal_repeat_set
    output  Super_maximal_repeat_set
    """

    # stop the recursively running if the set is empty 
    if not maximal_repeat_set : 
        return []

    # sort in the order of alphabet and length in ascending. 
#     maximal_repeat_set.sort()
#     maximal_repeat_set.sort(key=len)
    #print("To be processed here", maximal_repeat_set)
    
    # indicator to check if the pattern is super maximal repeat.
    flag=True  
    super_maximal_repeat_set=[]

    # getting the first one as a candidate 
    candidate=maximal_repeat_set.pop(0)
    # print("candidate",candidate)
    
    for maximal_repeat in maximal_repeat_set : 
        #check if candidate is in the maximal_repeat_set
        if ','.join(candidate)  in ','.join(maximal_repeat) :
            # print (candidate, "is in the", maximal_repeat)
            # even a one case is found, stop the commparision and get out of the loop 
            flag=False 
            break

    # if the pattern is super maximal repeat, adding it to the set 
    if flag==True : 
        #print(candidate,"is the element of SuperMaximalRepeatSet")
        super_maximal_repeat_set.append(candidate)
        #print(candidate, "appended in", super_maximal_repeat_set)
    
    # recursive call to compare with the remaining and the result will acc.
    super_maximal_repeat_set+= discover_super_maximal_repeat(maximal_repeat_set)
    #print("returning super_maximal_repeat_set", super_maximal_repeat_set)
    
    # sort for better view. 
    super_maximal_repeat_set.sort()
    super_maximal_repeat_set.sort(key=len)
    return super_maximal_repeat_set



def discover_near_super_maximal_repeat_minus_sm(dic_mr_set, maximal_repeat_set, super_maximal_repeat_set) :
    """
    Desc    discover the near super maximal repeat by recursive manner
    Used    in pattern_discover()
    Input   maximal_repeat_set, super_maximal_repeat_set
    output  near_Super_maximal_repeat_set
    """

    # candidates are the ones in the MR, not in the SMR
    candidates=[]
    for maximal_repeat in maximal_repeat_set :
        if maximal_repeat not in super_maximal_repeat_set : 
            candidates.append(maximal_repeat)
#     print('candidate',candidates)
#     # sort in the order of alphabet and length in ascending. 
#     candidates.sort()
#     candidates.sort(key=len)

    near_super_maximum_repeat_sets_minus_sm=[]

    for i in candidates : 
        key=','.join(i)
        candidate_locations=dic_mr_set[str(key)]['location']
        # print(key, candidate_locations)
        target_locations=[]
        # print('dic_mr_set.key',dic_mr_set.keys())
        # print('key',key)
        # print('target',set(dic_mr_set.keys()) - {key})
        for k in set(dic_mr_set.keys()) - {key} :
            if ''.join(k).find(''.join(key)) > -1 : 
                location=dic_mr_set[k]['location']
                # print(''.join(k), location)
                
                target_locations+=location
        target_locations.sort()
        # print("target", target_locations)

        for cl in candidate_locations :
            # indicator to if it is NSMR
            Flag=True 
            for tl in target_locations : 
                # print("comparision",i,cl,tl)

                # if candidate is included in the pattern (=not NSMR)
                # stop the comparision and out of loop  
                if ((tl[0]<=cl[0]) & (cl[1]<=tl[1])) : 
                    # print("\toverlapped", i, cl, tl)
                    Flag=False 
                    break

            # if the candidate is not included in any of other maximal repeats (=NSMR)
            if (Flag==True) : 
                # print("\t\tfound nearmaximum", i,cl,tl)
                near_super_maximum_repeat_sets_minus_sm.append(i)
                break
                # no more comparision required in Candidate locations 
    return near_super_maximum_repeat_sets_minus_sm






#To discover Maximal Repeats
def pattern_discover(log) : 
    """
    Desc    to find maximal repeats and its variations 
            mr(s) = Maximal Repeat (Set)
            smr(s) = Super Maximal Repeat (Set)
            nsmr(s) = Near Super Maximal Repeat (Set)
    Used    by discover.py  
    input   log, object of log imported by  xes_import_factory.apply (Pm4PY)
    output  list of mr, smr, and nsmr
    """
    maximal_repeat_set =[]
    super_maximal_repeat_set=[]
    near_super_maximal_repeat_set=[]

    concatenated_log=[]
    activity_classifier='concept:name'
    for trace in log:
        sequence=[event[activity_classifier] for event in trace]   
        sequence.append('|')
        concatenated_log+=sequence
    
    # print(concatenated_log)
    dic_mr_set=discover_maximal_repeat(concatenated_log)
    
    maximal_repeat_set=list(dic_mr_set.keys())
    maximal_repeat_set=post_processing_mr(maximal_repeat_set)
    maximal_repeat_set_=copy.deepcopy(maximal_repeat_set)
    
    super_maximal_repeat_set=discover_super_maximal_repeat(maximal_repeat_set_)
    near_super_maximal_repeat_minus_sm=discover_near_super_maximal_repeat_minus_sm(dic_mr_set,maximal_repeat_set,super_maximal_repeat_set)
    near_super_maximal_repeat= near_super_maximal_repeat_minus_sm + super_maximal_repeat_set
    near_super_maximal_repeat.sort()
    near_super_maximal_repeat.sort(key=len)
    
    return maximal_repeat_set, super_maximal_repeat_set, near_super_maximal_repeat
