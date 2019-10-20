from bert_serving.client import BertClient


class Bert:
    def __init__(self, ip="127.0.0.1"):
        try:
            self.bc = BertClient(ip)
        except Exception as e:
            print(str(e))

    def getBertEncoding(self, data):
        return self.bc.encode(data.values.tolist())