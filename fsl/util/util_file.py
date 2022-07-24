# def read_tsv_data(filename, skip_first=True):
#     sequences = []
#     labels = []
#     with open(filename, 'r') as file:
#         if skip_first:
#             next(file)
#         for line in file:
#             if line[-1] == '\n':
#                 line = line[:-1]
#             list = line.split('\t')
#             sequences.append(list[2])
#             labels.append(int(list[1]))
#     return [sequences, labels]


def read_tsv_data(filename, skip_first=True, meta=True, inference=False, test=False):
    sequences = []
    with open(filename, 'r') as file:
        if skip_first:
            next(file)
        for line in file:
            line = line.strip().split('\n')
            sequences = sequences + line
    if meta:
        if inference:
            labels = [1] * int((len(sequences) / 2)) + [0] * int((len(sequences) / 2))
        else:
            labels = [1] * len(sequences)
    else:
        if not test:
            print(len(sequences))
            labels = [1] * int((len(sequences)/2)) + [0] * int((len(sequences)/2))
        else:
            print(len(sequences))
            labels = [1] * 38 + [0] * 90

    return [sequences, labels]