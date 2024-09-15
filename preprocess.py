import json

from supar import Parser


def GetTree_heads(t):    # t -> con_tree of current sentence
    heads = [0] * len(t)
    mapnode = [0] * len(t)

    def Findheads(cidx, temp, headidx):    # (0, t, -1)
        if cidx >= len(temp):
            return cidx
        mapnode[cidx] = temp[cidx].lhs()    # lhs -> left child
        heads[cidx] = headidx

        if temp[cidx].lhs().__str__() == '_':
            mapnode[cidx] = temp[cidx].rhs()[0]

            return cidx + 1

        nidx = cidx + 1   # next_index = current_idx + 1
        for _ in temp[cidx].rhs():
            nidx = Findheads(nidx, temp, cidx)

        return nidx

    Findheads(0, t, -1)
    return heads, mapnode


def add_items(text_raw, dep_parser, con_parser):
    text_raw = text_raw.replace('(', '<').replace(')', '>').strip()
    token_temp = text_raw.split(' ')
    token = []
    for i in token_temp:
        if i != '':
            token.append(i)

    dep_parsed = dep_parser.predict(token, verbose=False)
    dep_info = dep_parsed.rels[0]
    dep_head = dep_parsed.arcs[0]
    dep_head = [i - 1 for i in dep_head]

    con_parsed = con_parser.predict(token, verbose=False)

    t = con_parsed.trees[0]
    con_head, temp = GetTree_heads(t.productions())
    con_info = []
    con_mark = '[N]'
    for info in temp:
        if info not in token:
            info = str(info) + con_mark
        con_info.append(info)
    return token, dep_head, dep_info, con_head, con_info


def create_new_dataset(fname):
    fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()

    new_data = []
    dep_parser = Parser.load('biaffine-dep-en')
    con_parser = Parser.load('crf-con-en')
    polarity_dict = {'1': 0, '-1': 1, '0': 2}
    context_dict = {}
    # flag = 0
    for i in range(0, len(lines), 3):
        # flag += 1
        # if flag == 10:
        #     break
        text_left, _, text_right = [s.lower().strip() for s in lines[i].partition('$T$')]
        aspect = lines[i+1].lower().strip().replace('\xa0', '').replace('(', '<').replace(')', '>')
        polarity = lines[i+2].strip()    # -1, 0, 1
        polarity = polarity_dict[polarity]
        text_left = text_left.strip().replace('\xa0', '').replace('(', '<').replace(')', '>').replace('   ', ' ')
        text_right = text_right.strip().replace('\xa0', '').replace('(', '<').replace(')', '>').replace('   ', ' ')
        text_raw = (text_left + ' ' + aspect + ' ' + text_right).strip().replace('\xa0', '').replace('(', '<').replace(')', '>')
        tll, trl = len(text_left.split(' ')), len(text_right.split(' '))
        tll = 0 if text_left == '' else tll
        asp_from, asp_to = tll, tll + len(aspect.strip().split(' '))

        if text_raw not in context_dict:
            context_dict[text_raw] = {'aspect': [], 'polarity': []}
            context_dict[text_raw]['aspect'] = [{'term': aspect, 'from': asp_from, 'to': asp_to}]
            context_dict[text_raw]['polarity'] = [polarity]
        else:
            context_dict[text_raw]['aspect'].append({'term': aspect, 'from': asp_from, 'to': asp_to})
            context_dict[text_raw]['polarity'].append(polarity)

    # context_dict = {'text...': {'aspect': [{'term', 'asp_frm', 'asp_to'}, {...}], polarity: [0, 1]}, ...}
    for text_raw, info in context_dict.items():
        token, dep_head, dep_info, con_head, con_info = add_items(text_raw, dep_parser, con_parser)
        aspect_list, polarity_list = info['aspect'], info['polarity']

        d = {'token': token, 'length': len(token),
             'dep_head': dep_head, 'dep_info': dep_info, 'con_head': con_head, 'con_info': con_info,
             'aspect_list': aspect_list, 'polarity': polarity_list}

        new_data.append(d)

    output_file = open(fname.replace('.raw', '_new.json'), 'w')
    json.dump(new_data, output_file)
    output_file.close()


if __name__ == '__main__':
    data_file = {  # original data
        'laptop14': {
            'train': './datasets/lap14/laptop_train.raw',
            'test': './datasets/lap14/laptop_test.raw'
        },
        'res14': {
            'train': './datasets/res14/restaurant_train.raw',
            'test': './datasets/res14/restaurant_test.raw'
        },
        'res15': {
            'train': './datasets/res15/restaurant_train.raw',
            'test': './datasets/res15/restaurant_test.raw'
        },
        'res16': {
            'train': './datasets/res16/restaurant_train.raw',
            'test': './datasets/res16/restaurant_test.raw'
        },
        'twitter': {
            'train': './datasets/twitter/twitter_train.raw',
            'test': './datasets/twitter/twitter_test.raw'
        }
    }
    datasets = ['laptop14', 'res14', 'res15', 'res16', 'twitter']
    # datasets = ['res15']
    for dataset in datasets:
        print('\nPreprocessing ' + dataset)
        create_new_dataset(data_file[dataset]['train'])
        create_new_dataset(data_file[dataset]['test'])
    print('\nDone !')
