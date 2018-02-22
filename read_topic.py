import pprint

if __name__ == '__main__':
    ret = {}
    with open("topWords.txt", "r") as f:
        for i, line in enumerate(f):
            words = line.split(":")[1].strip()
            ret["Topic %d" % i] = {}
            for w in words.split("\t"):
                key = w[ : w.find('(')]
                value = float(w[w.find('(')+1 : w.find(')')])
                ret["Topic %d" % i][key] = value

    print("topics = \\")
    pprint.pprint(ret)
