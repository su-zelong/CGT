import json
from supar import Parser

from preprocess import add_items

from string import punctuation

def clean_text(text_left, aspect, text_right):
    text_left = text_left.strip().replace('\xa0', '').replace('(', '<').replace(')', '>').replace('   ', ' ')
    text_right = text_right.strip().replace('\xa0', '').replace('(', '<').replace(')', '>').replace('   ', ' ')

    left_token = text_left.split(' ')
    left = []
    for t in left_token:
        contain = set(t) & set(punctuation)  # 包含的结果是一个符号
        if contain:
            c = ''.join(contain)  # punct
            w = ''.join(t.split(c))  # word
            if w != '':
                left.append(w)
            left.append(c)
        else:
            left.append(t)

    right_token = text_right.split(' ')
    right = []
    for t in right_token:
        contain = set(t) & set(punctuation)  # 包含的结果是一个符号
        if contain:
            c = ''.join(contain)  # punct
            w = ''.join(t.split(c))  #
            if w != '':
                right.append(w)
            right.append(c)
        else:
            right.append(t)

    text_left, text_right = ' '.join(left), ' '.join(right)

    aspect = aspect.lower().strip().replace('\xa0', '').replace('(', '<').replace(')', '>')
    text_raw = (text_left + ' ' + aspect + ' ' + text_right).strip().replace('\xa0', '').replace('(', '<').replace(')',
                                                                                                                   '>')
    return text_raw

def create_new_data(fname):
    with open(fname, 'r', encoding='utf-8', errors='ignore') as f:
        all_data = json.load(f)

    dep_parser = Parser.load('biaffine-dep-en')
    con_parser = Parser.load('crf-con-en')
    polarity_dict = {'positive': 0, 'negative': 1, 'neutral': 2}
    context_dict = {}
    new_data = []

    # 找到所有id
    all_id = []
    for key, d in all_data.items():    # 每一条数据
        k_id = d['id']
        all_id.append(k_id)
    all_id = list(set(all_id))

    adv_data = []
    for a_id in all_id:
        adv1_id = a_id + '_adv3'
        if adv1_id in all_data.keys():
            adv_data.append(all_data[adv1_id])
        else:
            adv_data.append(all_data[a_id])

    for d in adv_data:
        text_raw, aspect_term, a_from, a_to, polarity = d['sentence'], d['term'], d['from'], d['to'], d['polarity']
        text_left, text_right = text_raw[0: a_from].strip(), text_raw[a_to:].strip()
        text_raw = clean_text(text_left, aspect_term, text_right)
        tl_len = len(text_left.split(' ')) if text_left != '' else 0
        asp_from, asp_to = tl_len, tl_len + len(aspect_term.split(' '))
        polarity = polarity_dict[polarity]

        if text_raw not in context_dict:
            context_dict[text_raw] = {'aspect': [], 'polarity': []}
            context_dict[text_raw]['aspect'] = [{'term': aspect_term.lower(), 'from': asp_from, 'to': asp_to}]
            context_dict[text_raw]['polarity'] = [polarity]
        else:
            context_dict[text_raw]['aspect'].append({'term': aspect_term.lower(), 'from': asp_from, 'to': asp_to})
            context_dict[text_raw]['polarity'].append(polarity)

    for text_raw, info in context_dict.items():
        token, dep_head, dep_info, con_head, con_info = add_items(text_raw, dep_parser, con_parser)
        aspect_list, polarity_list = info['aspect'], info['polarity']

        d = {'token': token, 'length': len(token),
             'dep_head': dep_head, 'dep_info': dep_info, 'con_head': con_head, 'con_info': con_info,
             'aspect_list': aspect_list, 'polarity': polarity_list}

        new_data.append(d)

    output_file = open(fname.replace('enriched', 'ARTS_adv3'), 'w')
    json.dump(new_data, output_file)
    output_file.close()


if __name__ == '__main__':
    datasets = ['lap14', 'res14']
    ARTS_file = {
        'lap14': './datasets/ARTS/laptop_test_enriched.json',
        'res14': './datasets/ARTS/rest_test_enriched.json'
    }
    for data in datasets:
        print('\nCreating {0} ARTS dataset\n'.format(data))
        create_new_data(ARTS_file[data])
        print('\nSuccessful create {0} ARTS dataset\n'.format(data))
    print('\nDone\n')
