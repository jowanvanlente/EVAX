from typing import Dict, List
from enum import Enum


class Literal:

    def __init__(self, feature_name: str, feature_value: int, feature_name_digit: int):
        self.name = feature_name
        self.feature_name_digit = feature_name_digit
        self.value = feature_value
        self.outcomes = []

    def __eq__(self, other):
        return (self.name == other.name) & (self.value == other.value)

    def add_outcome(self, outcome):
        self.outcomes.append(outcome)


class Argument:

    def __init__(self, name: str, literal: Literal, output_class: int, probability, coverage):
        self.name = name
        self.premise_name = literal.name
        self.premise_value = literal.value
        self.conclusion = output_class
        self.strength = probability
        self.coverage = coverage
        self.responsibility: int
        self.feature_name_digit = literal.feature_name_digit
        self.strength_altered: int
        self.attacks: Argument = []
        self.is_attacked_by: Argument = []

    def add_attack(self, other: 'Argument'):
        if other not in self.attacks:
            self.attacks.append(other)
            other.is_attacked_by.append(self)


class AF:
    def __init__(self):
        self.arguments: Dict[str, Argument] = {}

    def add_argument(self, argument: Argument):
        self.arguments[argument.name] = argument


class Label(Enum):
    UNDECIDED = 1
    IN = 2
    OUT = 3


class Labeler:

    def get_labels(self, af: AF, semantics: str) -> Dict[Argument, Label]:
        if semantics == 'grounded':

            labeled_arguments = {argument: Label.UNDECIDED for argument in af.arguments.values()}
            todo_round = []

            for argument in af.arguments.values():
                if not argument.is_attacked_by:
                    todo_round.append(argument)

            while todo_round:
                todo_new_round = []

                # if attacker of this argument is IN, this argument is OUT
                for argument in todo_round:
                    if any(labeled_arguments.get(attacker) == Label.IN for attacker in argument.is_attacked_by):
                        labeled_arguments[argument] = Label.OUT

                    elif all(labeled_arguments.get(attacker) == Label.OUT for attacker in argument.is_attacked_by):
                        labeled_arguments[argument] = Label.IN

                    if labeled_arguments[argument] in [Label.IN, Label.OUT]:
                        todo_new_round += [attacked for attacked in argument.attacks
                                           if labeled_arguments.get(attacked) == Label.UNDECIDED]
                todo_round = todo_new_round

            return labeled_arguments

    def get_extension(self, af: AF, semantics: str) -> List[Argument]:
        labeled_arguments = self.get_labels(af, semantics)

        return [argument for argument, label in labeled_arguments.items() if label == Label.IN]


class ReadAF:

    def read_af(file_path : str) -> AF:

        af = AF()

        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if line.startswith('arg'):
                    arg_name = line.split('(', 1)[1].split(')')[0]
                    new_argument = Argument(arg_name)
                    af.add_argument(new_argument)

                elif line.split('att'):
                    attacker = af.arguments[line[line.index('(') + 1: line.index(',')]]
                    attacked = af.arguments[line[line.index(',') + 1: line.index(')')]]
                    attacker.add_attack(attacked)
            file.close()
        return af



