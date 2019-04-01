import json


def has(key, obj):
    if key in obj:
        return True
    else:
        return False

if __name__ == "__main__":
    with open('./evaluation-tables.json', 'r') as f:
        myjson = json.load(f)

    # Json data is organized in this hierarhcy task->datasets->sota->sota_rows

    paper_seen = set()

    for i in myjson:
        category = i['categories']
        task = i['task']

        for dataset in i['datasets']:
            if has('sota', dataset):
                #print(dataset['sota']['sota_rows'])
                sota = dataset['sota']
                for sota_row in sota['sota_rows']:
                    if sota_row['paper_url']:
                        if sota_row['paper_url'] not in paper_seen:
                            paper_seen.add(sota_row['paper_url'])
                            print('%s, %s' % (category[0], sota_row['paper_url']))



