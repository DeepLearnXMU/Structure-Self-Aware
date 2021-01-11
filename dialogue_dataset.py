import re
from itertools import product
from operator import itemgetter

import torch
from torch.utils.data import Dataset
import json
import numpy as np


class DiscourseGraph:
    def __init__(self, dialogue, pairs):
        self.dialogue = dialogue
        self.pairs = pairs
        self.edu_num = len(self.dialogue['edus'])
        self.paths = self.get_graph(pairs, self.edu_num)
        self.speaker_paths = self.get_speaker_paths(dialogue)
        self.turn_paths = self.get_turn_paths(dialogue)

    @staticmethod
    def print_path(path):
        for row in path:
            print([col for col in row])

    @staticmethod
    def get_speaker_paths(dialogue):
        speaker_size = len(dialogue['edus']) + 1
        speaker_4edu = ['None']
        for edu in dialogue['edus']:
            if isinstance(edu['speaker'], str):
                speaker_4edu.append(edu['speaker'])
            else:
                speaker_4edu.append('None')
        speaker_4edu = np.array(speaker_4edu)
        speaker_4edu_Aside = speaker_4edu.repeat(speaker_size).reshape(speaker_size, speaker_size)
        speaker_4edu_Bside = speaker_4edu_Aside.transpose()
        return (speaker_4edu_Aside == speaker_4edu_Bside).astype(np.long)

    @staticmethod
    def get_turn_paths(dialogue):
        turn_size = len(dialogue['edus']) + 1
        turns = [0] + [edu['turn'] for edu in dialogue['edus']]
        turns = np.array(turns)
        turn_Aside = turns.repeat(turn_size).reshape(turn_size, turn_size)
        turn_Bside = turn_Aside.transpose()
        return (turn_Aside == turn_Bside).astype(np.long)

    @staticmethod
    def get_coreference_path(dialogue):
        coreferences = []
        edu_num = len(dialogue['edus'])
        path = np.zeros((edu_num + 1, edu_num + 1), dtype=np.long)
        if 'solu' in dialogue:
            for cluster in dialogue['solu']:
                coreferences.append([k for (k, v) in cluster])
            for cluster in coreferences:
                for (x, y) in list(product(cluster, cluster)):
                    if x != y:
                        x, y = x + 1, y + 1
                        path[x][y] = 1
        return path.tolist()

    @staticmethod
    def get_graph(pairs, edu_num):
        node_num = edu_num + 1
        graph = np.zeros([node_num, node_num], dtype=np.long)
        for (x, y), label in pairs.items():
            graph[y + 1][x + 1] = label
        return graph.tolist()


class DialogueDataset(Dataset):
    def __init__(self, args, filename, mode, tokenizer):
        with open(filename, 'r') as file:
            print('loading {} data from {}'.format(mode, filename))
            dialogues = json.load(file)
        self.tokenizer = tokenizer
        self.padding_value = tokenizer.pad_token_id
        self.dialogues, self.relations = self.format_dialogue(dialogues)
        self.type2ids, self.id2types = None, None

    # dependency graph --> dependency tree; accumulate all relation types
    # return formatted dialogue: List [dialogue_dict_1, dialogue_dict_2, ...]
    #        relation types: set {relation type}
    def format_dialogue(self, dialogues):
        print('format dataset..')
        relation_types = set()
        for dialogue in dialogues:
            last_speaker = None
            turn = 0
            for edu in dialogue['edus']:
                text = edu['text']
                # web page link to special token
                while text.find("http") >= 0:
                    i = text.find("http")
                    j = i
                    while j < len(text) and text[j] != ' ': j += 1
                    text = text[:i] + " [url] " + text[j + 1:]
                # delete invalid char
                invalid_chars = ["/", "\*", "^", ">", "<", "\$", "\|", "=", "@"]
                for ch in invalid_chars:
                    text = re.sub(ch, "", text)
                text = self.tokenizer.encode(text)
                edu['text'] = text

                if edu["speaker"] != last_speaker:
                    last_speaker = edu["speaker"]
                    turn += 1
                edu["turn"] = turn

            dialogue['relations'] = sorted(dialogue['relations'], key=itemgetter('y', 'x'))
            for relation in dialogue['relations']:
                relation['type'] = relation['type'].strip().lower()
                if relation['type'] not in relation_types:
                    relation_types.add(relation['type'])

        return dialogues, relation_types

    @staticmethod
    def format_relations(relations: set):
        id2types = ['None'] + sorted(list(relations))
        type2ids = {type: i for i, type in enumerate(id2types)}
        return type2ids, id2types

    def get_relations(self, relations, type2ids, id2types):
        self.relations, self.type2ids, self.id2types = relations, type2ids, id2types

    def get_discourse_graph(self):
        for dialogue in self.dialogues:
            pairs = {(relation['x'], relation['y']): self.type2ids[relation['type']]
                     for relation in dialogue['relations']}
            discourse_graph = DiscourseGraph(dialogue, pairs)
            dialogue['graph'] = discourse_graph

    @staticmethod
    def nest_padding(sequence):
        max_cols = max([len(row) for batch in sequence for row in batch])
        max_rows = max([len(batch) for batch in sequence])
        sequence = [batch + [[0] * (max_cols)] * (max_rows - len(batch)) for batch in sequence]
        sequence = torch.tensor([row + [0] * (max_cols - len(row)) for batch in sequence for row in batch])
        return sequence.reshape(-1, max_rows, max_cols)

    @staticmethod
    def padding(sequence: torch.Tensor, padding_value):
        return (sequence != padding_value).byte()

    def __len__(self):
        return len(self.dialogues)

    def __getitem__(self, index):
        dialogue = self.dialogues[index]
        texts = [edu['text'] for edu in dialogue['edus']]
        lengths = torch.tensor([len(sent) for sent in texts])

        graph = dialogue['graph']
        pairs = graph.pairs
        paths = graph.paths
        speakers = graph.speaker_paths.tolist()
        turns = graph.turn_paths.tolist()
        if 'id' not in dialogue:
            dialogue['id'] = 'none'
        return texts, pairs, paths, lengths, speakers, turns, graph.edu_num, dialogue['id']