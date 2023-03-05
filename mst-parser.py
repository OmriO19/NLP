from networkx import DiGraph
from networkx.algorithms import maximum_spanning_arborescence
from nltk.corpus import dependency_treebank
from itertools import combinations
from collections import defaultdict
import time

WORD = "word"
TAG = "tag"
DEPS = "deps"
ADDRESS = "address"


def get_word_set(data):
    """
    :param data:
    :return: a set of all the words from the data
    """
    s = set()
    for sent in data:
        for node in sent.nodes.values():
            s.add(node[WORD])
    return s


def get_tags_set(data):
    """
    :param data:
    :return: a set of all the tags from the data
    """
    s = set()
    for sent in data:
        for node in sent.nodes.values():
            s.add(node[TAG])
    return s


def get_set_ind(sett):
    """
    :param sett: a set
    :return: a dict with all elements
    """
    set_ind = defaultdict(lambda: -1)
    for i, elem in enumerate(sett):
        set_ind[elem] = i
    return set_ind


class Arc:
    def __init__(self, head, tail, weight):
        self.head = head
        self.tail = tail
        self.weight = weight


def gold_standard_tree(sent):
    """
    :param sent: a sentence
    :return:  the feature sum of all the sentence
    """
    arcs = []
    for node in sent.nodes.values():
        deps = list(node[DEPS].values())
        if len(deps) > 0:
            tails = deps[0]
            for tail in tails:
                arcs.append((node[ADDRESS], tail))
    return arcs


class Sparse:
    """
    implementation of a sparse vector
    """
    def __init__(self):
        self.vec = defaultdict(lambda: 0)

    def __add__(self, other):
        vec_sum = Sparse()
        vec_sum.vec = self.vec
        for key, val in other.vec.items():
            summ = vec_sum[key] + val
            if summ == 0:  # remove the item when the val is zero
                vec_sum.vec.pop(key)
            else:
                vec_sum[key] = summ
        return vec_sum

    def __iadd__(self, other):
        for key, val in other.vec.items():
            summ = self.vec[key] + val
            if summ == 0:  # remove the item when the val is zero
                self.vec.pop(key)
            else:
                self.vec[key] = summ
        return self

    def __sub__(self, other):
        vec_diff = Sparse()
        vec_diff.vec = self.vec
        for key, val in other.vec.items():
            diff = vec_diff[key] - val
            if diff == 0:  # remove the item when the val is zero
                vec_diff.vec.pop(key)
            else:
                vec_diff[key] = diff
        return vec_diff

    def __isub__(self, other):
        for key, val in other.vec.items():
            diff = self.vec[key] - val
            if diff == 0:  # remove the item when the val is zero
                self.vec.pop(key)
            else:
                self.vec[key] = diff
        return self

    def __setitem__(self, key, value):
        self.vec[key] = value

    def __getitem__(self, item):
        return self.vec[item]

    def __mul__(self, other):
        mu = Sparse()
        for key, val in self.vec.items():
            mu.vec[key] = val * other
        return mu

    def __truediv__(self, other):
        return self * (1/other)

    def dot(self, other):
        dot_product = 0
        for key, val in other.vec.items():
            dot_product += val * self.vec[key]
            if self.vec[key] == 0:  # remove the item when the val is zero
                self.vec.pop(key)
        return dot_product


class MST:
    def __init__(self):
        self.parsed_sents = dependency_treebank.parsed_sents()
        len_data = len(dependency_treebank.parsed_sents())
        self.training = self.parsed_sents[:int(len_data * 0.9)]
        self.test = self.parsed_sents[int(len_data * 0.9):]
        self.word_set = get_word_set(self.parsed_sents)
        self.tag_set = get_tags_set(self.parsed_sents)
        self.words_ind = get_set_ind(self.word_set)
        self.tags_ind = get_set_ind(self.tag_set)
        self.offset = len(self.word_set)**2
        self.dim = self.offset + len(self.tag_set)**2
        self.w = Sparse()

    def feature(self, v, u):
        """
        :param v: first node
        :param u: second node
        :return: a feature vector that match the given arc
        """
        vec = Sparse()
        v_word_ind = self.words_ind[v[WORD]]
        u_word_ind = self.words_ind[u[WORD]]
        v_tag_ind = self.tags_ind[v[TAG]]
        u_tag_ind = self.tags_ind[u[TAG]]
        if v_word_ind != -1 and u_word_ind != -1:
            words_i = v_word_ind * len(self.word_set) + u_word_ind
            vec[words_i] = 1
        if v_tag_ind != -1 and u_tag_ind != -1:
            tags_i = v_tag_ind * len(self.tag_set) + u_tag_ind
            vec[tags_i + self.offset] = 1
        return vec

    def gold_standard_feature(self, sent):
        """
        :param sent: a sentence
        :return:  the feature sum of all the sentence
        """
        feature_sum = Sparse()
        for node in sent.nodes.values():
            deps = list(node[DEPS].values())
            if len(deps) > 0:
                tails = deps[0]
                for tail in tails:
                    feature_sum += self.feature(node, sent.nodes[tail])
        return feature_sum

    def tree_feature(self, tree, sent):
        """
        :param tree: the tree that represent the sentence
        :param sent: a sentence
        :return: the feature sum of all the sentence
        """
        feature_sum = Sparse()
        for arc in tree.values():
            feature_sum += self.feature(sent.nodes[arc.head], sent.nodes[arc.tail])
        return feature_sum

    def perceptron_iter(self, lr):
        """
        an iteration of the perceptron algorithm
        :param lr: learning rate
        """
        j = 0
        for sent in self.training:
            print("___ Sentence number " + str(j))
            # if(j == 100):
            #     continue
            j += 1
            nodes_indices = list(sent.nodes.keys())
            root, nodes = nodes_indices[0], nodes_indices[1:]
            all_combs = combinations(nodes, 2)
            arcs = []
            for (head, tail) in all_combs:
                weight = self.w.dot(self.feature(sent.nodes[head], sent.nodes[tail]))
                arcs.append(Arc(head, tail, weight))
                weight = self.w.dot(self.feature(sent.nodes[tail], sent.nodes[head]))
                arcs.append(Arc(tail, head, weight))
            for node in nodes:
                weight = self.w.dot(self.feature(sent.nodes[root], sent.nodes[node]))
                arcs.append(Arc(root, node, weight))
            t_tag = max_spanning_arborescence_nx(arcs)
            if lr != 1:  # for efficiency reason
                self.w += (self.gold_standard_feature(sent) - self.tree_feature(t_tag, sent)) * lr
            else:
                self.w += (self.gold_standard_feature(sent) - self.tree_feature(t_tag, sent))

    def perceptron(self, lr=1, iter=2):
        """
        :param lr: learning rate, 1 as default
        :param iter: number of iterations, 2 as default
        """
        for i in range(iter):
            print("____ Iteration number " + str(i + 1))
            self.perceptron_iter(lr)
        self.w = self.w / (len(self.training) * iter)

    def evaluation(self):
        """
        :return: the evaluation of the model based on the learning
        """
        print("____ Evaluation")
        eqs = []
        j = 0
        for sent in self.test:
            eq = 0
            print("___ Sentence number " + str(j))
            j += 1
            nodes_indices = list(sent.nodes.keys())
            root, nodes = nodes_indices[0], nodes_indices[1:]
            all_combs = combinations(nodes, 2)
            arcs = []
            for (head, tail) in all_combs:
                weight = self.w.dot(self.feature(sent.nodes[head], sent.nodes[tail]))
                arcs.append(Arc(head, tail, weight))
                weight = self.w.dot(self.feature(sent.nodes[tail], sent.nodes[head]))
                arcs.append(Arc(tail, head, weight))
            for node in nodes:
                weight = self.w.dot(self.feature(sent.nodes[root], sent.nodes[node]))
                arcs.append(Arc(root, node, weight))
            arcs_tag = list(max_spanning_arborescence_nx(arcs).values())
            given_arcs = gold_standard_tree(sent)
            for correct_arc in given_arcs:
                head, tail = correct_arc
                for arc in arcs_tag:
                    if arc.head == head and arc.tail == tail:
                        eq += 1
            eqs.append(eq / len(sent.nodes))
        return sum(eqs) / len(self.test)


def max_spanning_arborescence_nx(arcs):
    """
    Wrapper for the networkX min_spanning_tree to follow the original API
    :param arcs: list of Arc tuples
    :param sink: unused argument. We assume that 0 is the only possible root over the set of edges given to
     the algorithm.
    """
    print("Applying on " + str(len(arcs)) + " arcs")
    G = DiGraph()
    for arc in arcs:
        G.add_edge(arc.head, arc.tail, weight=arc.weight)
    ARB = maximum_spanning_arborescence(G)
    result = {}
    headtail2arc = {(a.head, a.tail): a for a in arcs}
    for edge in ARB.edges:
        tail = edge[1]
        result[tail] = headtail2arc[(edge[0], edge[1])]
    return result


if __name__ == '__main__':
    t0 = time.time()
    mst = MST()
    mst.perceptron()
    # print(mst.w.vec)
    eval_mst = mst.evaluation()
    print("* Evaluation score: " + str(eval_mst))
    t1 = time.time()
    print(str((t1 - t0) / 60) + " min")
