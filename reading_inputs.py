from glob import glob

def read():
    filename = glob('inputs.txt')[0]

    with open(filename, 'r') as f:
        data = f.readlines()
        d = [i.split()[1] for i in data[2:]]

    d[7] = None if d[7] == 'None' else d[7]
    d[8] = None if d[8] == 'None' else d[8]
    d[9] = None if d[9] == 'None' else d[9]
    d[11] = None if d[11] == 'None' else d[11]
    d[12] = True if d[12] == 'True' else False

    info =  [int(d[0]), int(d[1]), int(d[2]), int(d[3]), int(d[4]), float(d[5]), float(d[6]),
             d[7], d[8], d[9], d[10], d[11], d[12], int(d[13])]
    return info
