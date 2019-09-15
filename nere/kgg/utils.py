def create_triple(basic_fact, re_result):
    triple = set()
    name_dict = {}
    defendants = basic_fact['被告']
    plain_insurance_company_idx = 1
    plain_company_idx = 1
    defend_insurance_company_idx = 1
    defend_company_idx = 1
    for defend in defendants:
        name = defend['名字']
        if "保险" in name:
            fan_name = "被告保险公司_" + str(defend_insurance_company_idx)
            defend_insurance_company_idx += 1
        elif "公司" in name:
            fan_name = "被告公司_" + str(defend_company_idx)
            defend_company_idx += 1
        else:
            fan_name = '被告_' + str(defend['编号'])
        name_dict[name] = fan_name
    plaintiff = basic_fact['原告']
    for plain in plaintiff:
        name = plain['名字']
        if "保险" in name:
            fan_name = "原告保险公司_" + str(plain_insurance_company_idx)
            plain_insurance_company_idx += 1
        elif "公司" in name:
            fan_name = "原告公司_" + str(plain_company_idx)
            plain_company_idx += 1
        else:
            fan_name = '原告_' + str(plain['编号'])
        name_dict[name] = fan_name
    entity_norm = determine_role(re_result, name_dict)
    for res in re_result:
        e1, e2, rel = res[0], res[1], res[2]
        if rel == "指代":
            continue
        e1_value, e1_label = e1[0], e1[1]
        e2_value, e2_label = e2[0], e2[1]
        head_entity = entity_norm[e1_value]
        if e2_value in entity_norm:
            end_entity = entity_norm[e2_value]
        elif e1_label == "责任认定":
            end_entity = e2_value
        else:
            end_entity = e2_label
        triple.add((head_entity, rel, end_entity))
    return triple


def determine_role(triples, basic_role):
    person = []
    car = []
    entity_norm = {}
    drive = []
    accident = []
    owner = []
    take = []
    refer = []
    for trip in triples:
        e1, e2, rel = trip[0], trip[1], trip[2]
        e1_value, e1_label = e1[0], e1[1]
        e2_value, e2_label = e2[0], e2[1]
        if e1_value not in entity_norm:
            if (e1_label == "自然人主体" or e1_label == "非自然人主体") and e1_value not in person:
                if e1_value in basic_role:
                    entity_norm[e1_value] = basic_role[e1_value]
                else:
                    person.append(e1_value)
            elif (e1_label == "机动车" or e1_label == "非机动车") and e1_value not in car:
                car.append(e1_value)
        if e2_value not in entity_norm:
            if (e2_label == "自然人主体" or e2_label == "非自然人主体") and e2_value not in person:
                if e1_value in basic_role:
                    entity_norm[e2_value] = basic_role[e2_value]
                else:
                    person.append(e2_value)
            elif (e2_label == "机动车" or e2_label == "非机动车") and e2_value not in car:
                car.append(e2_value)
        if rel == "驾驶":
            drive.append(trip)
        elif rel == "发生事故":
            accident.append(trip)
        elif rel == "所有":
            owner.append(trip)
        elif rel == "搭乘":
            take.append(trip)
        elif rel == "指代":
            refer.append(trip)
    driver = 1
    owners = 1
    for trip in drive:
        person_name, car_name = trip[0][0], trip[1][0]
        if person_name in entity_norm and car_name not in entity_norm:
            # car_name_norm = norm_car(car_name)
            car_name_norm = trip[1][1]
            entity_norm[car_name] = entity_norm[person_name] + '-' + car_name_norm
            car.remove(car_name)
        elif person_name not in entity_norm and car_name in entity_norm:
            entity_norm[person_name] = entity_norm[car_name] + '-驾驶者'
            person.remove(person_name)
        elif person_name not in entity_norm and car_name not in entity_norm:
            # car_name_norm = norm_car(car_name)
            car_name_norm = trip[1][1]
            person_name_norm = "驾驶人_" + str(driver)
            driver += 1
            entity_norm[person_name] = person_name_norm
            entity_norm[car_name] = person_name_norm + '-' + car_name_norm
            car.remove(car_name)
            person.remove(person_name)
    if len(person) == 0 and len(car) == 0:
        return entity_norm
    for trip in owner:
        person_name, car_name = trip[0][0], trip[1][0]
        if person_name in entity_norm and car_name not in entity_norm:
            # car_name_norm = norm_car(car_name)
            car_name_norm = trip[1][1]
            entity_norm[car_name] = entity_norm[person_name] + '-' + car_name_norm
            car.remove(car_name)
        elif person_name not in entity_norm and car_name in entity_norm:
            entity_norm[person_name] = entity_norm[car_name] + '-拥有者'
            person.remove(person_name)
        elif person_name not in entity_norm and car_name not in entity_norm:
            # car_name_norm = norm_car(car_name)
            car_name_norm = trip[1][1]
            person_name_norm = "拥有者_" + str(owners)
            owners += 1
            entity_norm[person_name] = person_name_norm
            entity_norm[car_name] = person_name_norm + '-' + car_name_norm
            car.remove(car_name)
            person.remove(person_name)
    if len(person) == 0 and len(car) == 0:
        return entity_norm
    passengers_role = {}
    for trip in accident:
        car_name, entity_name, entity_label = trip[0][0], trip[1][0], trip[1][1]
        if car_name in entity_norm and entity_name not in entity_norm:
            if entity_label == "自然人主体":
                person_name_norm = entity_norm[car_name] + '-受害人'
                if person_name_norm not in passengers_role:
                    passengers_role[person_name_norm] = 0
                    entity_norm[entity_name] = person_name_norm
                else:
                    passengers_role[person_name_norm] += 1
                    entity_norm[entity_name] = person_name_norm + '_' + str(passengers_role[person_name_norm])
                person.remove(entity_name)
            else:
                # car_name_norm = norm_car(entity_name)
                car_name_norm = trip[1][1]
                person_name_norm = '事故车-' + car_name_norm
                if person_name_norm not in passengers_role:
                    passengers_role[person_name_norm] = 0
                    entity_norm[entity_name] = person_name_norm
                else:
                    passengers_role[person_name_norm] += 1
                    entity_norm[entity_name] = '事故车' + str(passengers_role[person_name_norm]) + '-' + car_name_norm
                car.remove(entity_name)
        elif car_name not in entity_norm and entity_name in entity_norm:
            # car_name_norm = norm_car(car_name)
            car_name_norm = trip[0][1]
            accident_car = '肇事车-' + car_name_norm
            if accident_car not in passengers_role:
                passengers_role[accident_car] = 0
                entity_name[car_name] = accident_car
            else:
                passengers_role[accident_car] += 1
                entity_name[car_name] = '肇事车' + str() + car_name_norm
            car.remove(car_name)
        elif car_name not in entity_norm and entity_name not in entity_norm:
            # car_name_norm = norm_car(car_name)
            car_name_norm = trip[0][1]
            accident_car = '肇事车-' + car_name_norm
            if accident_car not in passengers_role:
                passengers_role[accident_car] = 0
                entity_name[car_name] = accident_car
            else:
                passengers_role[accident_car] += 1
                entity_name[car_name] = '肇事车' + str() + car_name_norm
            car.remove(car_name)
            if entity_label == "自然人主体":
                person_name_norm = entity_norm[car_name] + '-受害人'
                if person_name_norm not in passengers_role:
                    passengers_role[person_name_norm] = 0
                    entity_norm[entity_name] = person_name_norm
                else:
                    passengers_role[person_name_norm] += 1
                    entity_norm[entity_name] = person_name_norm + '_' + str(passengers_role[person_name_norm])
                person.remove(entity_name)
            else:
                # car_name_norm = norm_car(entity_name)
                car_name_norm = trip[1][1]
                person_name_norm = '事故车-' + car_name_norm
                if person_name_norm not in passengers_role:
                    passengers_role[person_name_norm] = 0
                    entity_norm[entity_name] = person_name_norm
                else:
                    passengers_role[person_name_norm] += 1
                    entity_norm[entity_name] = '事故车' + str(passengers_role[person_name_norm]) + '-' + car_name_norm
                car.remove(entity_name)
    if len(person) == 0 and len(car) == 0:
        return entity_norm
    for trip in take:
        car_name, person_name = trip[0][0], trip[1][0]
        if person_name not in entity_norm and car_name in entity_norm:
            person_name_norm = entity_norm[car_name] + '-乘车人'
            if person_name_norm not in passengers_role:
                passengers_role[person_name_norm] = 0
                entity_norm[person_name] = person_name_norm
            else:
                passengers_role[person_name_norm] += 1
                entity_norm[person_name] = person_name_norm + '_' + str(passengers_role[person_name_norm])
            person.remove(person_name)
        elif person_name in entity_norm and car_name not in entity_norm:
            # car_norm = norm_car(car_name)
            car_norm = trip[0][1]
            car_name_norm = entity_norm[person_name] + '所乘_' + car_norm
            entity_norm[person_name] = car_name_norm
            car.remove(car_name)
        elif person_name not in entity_norm and car_name not in entity_norm:
            # car_norm = norm_car(car_name)
            car_norm = trip[0][1]
            car_name_norm = '其他_' + car_norm
            if car_name_norm not in passengers_role:
                passengers_role[car_name_norm] = 0
                entity_norm[car_name] = car_name_norm
            else:
                passengers_role[car_name_norm] += 1
                entity_norm[car_name] = '其他' + str(passengers_role[car_name_norm]) + "_" + car_norm
            car.remove(car_name)
            person_name_norm = entity_norm[car_name] + '-乘车人'
            if person_name_norm not in passengers_role:
                passengers_role[person_name_norm] = 0
                entity_norm[person_name] = person_name_norm
            else:
                passengers_role[person_name_norm] += 1
                entity_norm[person_name] = person_name_norm + '_' + str(passengers_role[person_name_norm])
            person.remove(person_name)
    if len(person) == 0 and len(car) == 0:
        return entity_norm
    for trip in refer:
        car_name1, car_name2 = trip[0][0], trip[1][0]
        if car_name1 not in entity_norm and car_name2 in entity_norm:
            entity_norm[car_name1] = entity_norm[car_name2]
            car.remove(car_name1)
        elif car_name1 in entity_norm and car_name2 not in entity_norm:
            entity_norm[car_name2] = entity_norm[car_name1]
            car.remove(car_name2)
        elif car_name1 not in entity_norm and car_name2 not in entity_norm:
            # car_norm = norm_car(car_name2)
            car_norm = trip[1][1]
            car_name_norm = '其他_' + car_norm
            if car_name_norm not in passengers_role:
                passengers_role[car_name_norm] = 0
                entity_norm[car_name2] = car_name_norm
            else:
                passengers_role[car_name_norm] += 1
                entity_norm[car_name2] = '其他' + str(passengers_role[car_name_norm]) + "_" + car_norm
            entity_norm[car_name1] = entity_norm[car_name2]
            car.remove(car_name1)
            car.remove(car_name2)
    return entity_norm


def norm_car(car_name):
    ss = ['x', 'X', '×', '的', '号', '牌']
    gen_list = list('Ｃ/京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼QWERTYUIOPASDFGHJKLZXCVBNMqwertyiopasdfghjklzxcvbnm0123456789')
    new_name = car_name
    for gen in ss:
        if gen in new_name:
            new_name = new_name[new_name.find(gen):].replace(gen, '')
    for gen in gen_list:
        new_name = new_name.replace(gen, '')
    return new_name
