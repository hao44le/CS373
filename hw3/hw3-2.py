import urllib.request
my_url = "https://www.cs.purdue.edu/homes/ribeirob/courses/Fall2017/data/yelp2_train.csv"
# local_filename, headers = urllib.request.urlretrieve(my_url)
local_filename = "yelp2_train.csv"

line_number = 0
#d
goodForGroups_set,city_set,state_set,stars_set,is_open_set,alcohol_set,noiseLevel_set,attire_set,priceRange_set,delivery_set,waiterService_set,smoking_set,outdoorSeating_set,caters_set,goodForKids_set = set(),set(),set(),set(),set(),set(),set(),set(),set(),set(),set(),set(),set(),set(),set()
sets = [goodForGroups_set,city_set,state_set,stars_set,is_open_set,alcohol_set,noiseLevel_set,attire_set,priceRange_set,delivery_set,waiterService_set,smoking_set,outdoorSeating_set,caters_set,goodForKids_set]
set_names = ["goodForGroups_set","city_set","state_set","stars_set","is_open_set","alcohol_set","noiseLevel_set","attire_set","priceRange_set","delivery_set","waiterService_set","smoking_set","outdoorSeating_set","caters_set","goodForKids_set"]

#g
numerator_factor = 1
priceRange_dict = {'':[0,0],'1':[0,0],'2':[0,0],'3':[0,0],'4':[0,0]}
alcohol_dict = {'none':[0,0],'full_bar':[0,0],'beer_and_wine':[0,0]}
noiseLevel_dict = {'':[0,0],'very_loud':[0,0],'loud':[0,0],'quiet':[0,0],'average':[0,0]}
attire_dict = {'':[0,0],'dressy':[0,0],'casual':[0,0],'formal':[0,0]}
all_dict = {'priceRange_dict':priceRange_dict,'alcohol_dict':alcohol_dict,'noiseLevel_dict':noiseLevel_dict,'attire_dict':attire_dict}

with open(local_filename) as in_file:
    for line in in_file.readlines():
        line_number += 1
        line = line.strip()
        if line_number == 1:
            #header.skip
            continue
        else:
            [goodForGroups,city,state,_,_,stars,_,_,is_open,alcohol,noiseLevel,attire,priceRange,delivery,waiterService,smoking,outdoorSeating,caters,goodForKids] = line.split(",")
            #d
            goodForGroups_set.add(goodForGroups)
            city_set.add(city)
            state_set.add(state)
            stars_set.add(stars)
            is_open_set.add(is_open)
            alcohol_set.add(alcohol)
            noiseLevel_set.add(noiseLevel)
            attire_set.add(attire)
            priceRange_set.add(priceRange)
            delivery_set.add(delivery)
            waiterService_set.add(waiterService)
            smoking_set.add(smoking)
            outdoorSeating_set.add(outdoorSeating)
            caters_set.add(caters)
            goodForKids_set.add(goodForKids)

            #g
            if goodForGroups == '0':
                priceRange_dict[priceRange] = [priceRange_dict[priceRange][0]+1,priceRange_dict[priceRange][1]]
                alcohol_dict[alcohol] = [alcohol_dict[alcohol][0]+1,alcohol_dict[alcohol][1]]
                noiseLevel_dict[noiseLevel] = [noiseLevel_dict[noiseLevel][0]+1,noiseLevel_dict[noiseLevel][1]]
                attire_dict[attire] = [attire_dict[attire][0]+1,attire_dict[attire][1]]
            else:
                priceRange_dict[priceRange] = [priceRange_dict[priceRange][0],priceRange_dict[priceRange][1]+1]
                alcohol_dict[alcohol] = [alcohol_dict[alcohol][0],alcohol_dict[alcohol][1]+1]
                noiseLevel_dict[noiseLevel] = [noiseLevel_dict[noiseLevel][0],noiseLevel_dict[noiseLevel][1]+1]
                attire_dict[attire] = [attire_dict[attire][0],attire_dict[attire][1]+1]

#d
# total_number_of_distinct_values = 0
# for index,m_set in enumerate(sets):
    # print("{} {}:{}".format(set_names[index],len(m_set),m_set))
    # total_number_of_distinct_values += len(m_set)
# print("total_number_of_distinct_values:{}".format(total_number_of_distinct_values))

#g
# for key in all_dict:
#     print("{}:{}".format(key,all_dict[key]))
