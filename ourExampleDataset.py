import itertools, csv


def writeToFile(dataset, header, filename):
    try:
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for row in dataset:
                writer.writerow(row)
    except IOError:
        print("I/O Error occured here!!")


fire_hazard_class_factor = [0.00026, 0.00051, 0.00073, 0.00116, 0.00145, 0.00189, 0.00305, 0.00585]
contribution_margin = [i for i in range(10000, 99000000, 10000000)]
liability_time = [6, 12, 18]
benefit_cover = [0, 1]
benefit_gastro = [0, 1]
benefit_trade = [0, 1]
elementary_liability = [3, 6, 9]
special_discount = [i for i in range(-100, 30, 10)]
vpc_discount = [i for i in range(1, 100, 10)]

fire_bonus_header = [
    "Garage Square Feet", "Total Property Area", "Has Fireplace", "Has Pool", "County factor", "Land Price"]
elementary_bonus_header = ["Garage Square Feet", "Garage Condition", "Garage Price"]
special_discount_header = fire_bonus_header[:-1] + ["Garage Condition"] + \
                          ["Number of rooms", "No. of stories", "Property Price"]

elementary_bonus_dataset = list()
elementary_bonus_values = list()
elementary_bonus_values_map = dict()
for element in itertools.product(contribution_margin, elementary_liability):
    if element[1] == 3:
        X = element[0] * 0.00044 * 1.15
    elif element[1] == 6:
        X = element[0] * 0.00058 * 1.15
    elif element[1] == 9:
        X = element[0] * 0.00073 * 1.15

    X = float("{:.3f}".format(X))
    elementary_bonus_dataset.append((element[0], element[1], X))
    elementary_bonus_values.append(X)
    elementary_bonus_values_map[str(X)] = (element[0], element[1])

fire_bonus_dataset = list()
fire_bonus_values = list()
fire_bonus_values_map = dict()
for element in itertools.product(contribution_margin, liability_time, benefit_trade, benefit_gastro,
                                 fire_hazard_class_factor):
    if element[1] == 6:
        if element[2]:
            if element[3]:
                X = element[0] * element[4] * 0.75 * (1.15 + 0.06 + 0.06)
            else:
                X = element[0] * element[4] * 0.75 * (1.15 + 0.06)
        else:
            if element[3]:
                X = element[0] * element[4] * 0.75 * (1.15 + 0.06)
            else:
                X = element[0] * element[4] * 0.75 * 1.15

    elif element[1] == 12:
        if element[2]:
            if element[3]:
                X = element[0] * element[4] * (1.15 + 0.06 + 0.06)
            else:
                X = element[0] * element[4] * (1.15 + 0.06)
        else:
            if element[3]:
                X = element[0] * element[4] * (1.15 + 0.06)
            else:
                X = element[0] * element[4] * 1.15
    elif element[1] == 18:
        if element[2]:
            if element[3]:
                X = element[0] * element[4] * 1.25 * (1.15 + 0.06 + 0.06)
            else:
                X = element[0] * element[4] * 1.25 * (1.15 + 0.06)
        else:
            if element[3]:
                X = element[0] * element[4] * 1.25 * (1.15 + 0.06)
            else:
                X = element[0] * element[4] * 1.25 * 1.15
    X = float("{:.3f}".format(X))
    fire_bonus_dataset.append((element[0], element[1], element[2], element[3], element[4], X))
    fire_bonus_values.append(X)
    fire_bonus_values_map[str(X)] = (element[0], element[1], element[2], element[3], element[4])

# print(len(elementary_bonus_dataset))
# print(len(fire_bonus_dataset))
# writeToFile(fire_bonus_dataset, fire_bonus_header, "firebonus_dataset_new.csv")
# writeToFile(elementary_bonus_dataset, elementary_bonus_header, "elementary_dataset_new.csv")
try:
    with open("Datasets/special_discount_new.csv", 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(special_discount_header)
        for element in itertools.product(fire_bonus_values, elementary_bonus_values, special_discount, vpc_discount):
            X = (element[2] + element[3]) * ((100 - element[0]) * (100 - element[1]) / 100)
            X = float("{:.2f}".format(X))
            fb = fire_bonus_values_map[str(element[0])]
            eb = elementary_bonus_values_map[str(element[1])]
            w.writerow((fb[0], fb[1], fb[2], fb[3], eb[1], element[2], element[3], X))
except IOError:
    print("I/O Error occurred here!!")
