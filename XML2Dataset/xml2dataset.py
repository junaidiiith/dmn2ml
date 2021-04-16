"""
Takes an XML file and produces datasets from the decision tables of the decisions.

Inputs: xml file, metadata mapper file of elements with allowed ranges

"""

import csv
import itertools

import json
import xmltodict
import re

DEFINITION = 'definitions'
DECISIONS = 'decision'
INPUT_DATA = 'inputData'
REQUIRED_INPUT = 'requiredInput'
REQUIRED_DECISION = 'requiredDecision'
INFORMATION_REQUIREMENT = 'informationRequirement'
ID = '@id'
NAME = '@name'
LINK = '@href'
LABEL = '@label'
DECISION_TABLE = 'decisionTable'
DECISION_LITERAL_EXPRESSION = 'literalExpression'
DECISION_INPUTS = 'input'
DECISION_OUTPUTS = 'output'
INPUT_VARIABLE_EXPRESSION = 'inputExpression'
INPUT_VARIABLE_VALUES = 'inputValues'
VARIABLE_TYPE = '@typeRef'
RULES = 'rule'
INPUT_ENTRY = 'inputEntry'
OUTPUT_ENTRY = 'outputEntry'
DECIMAL_PLACE = 3


boolean_values = {
    "false": [0, "false", "False"],
    "true": [1, "true", "True"]
}


def load_dmn_file_as_json(filename):
    json.dump(xmltodict.parse(open(filename).read()), open(filename.split(".")[0] + '.json', 'w'))
    return json.load(open(filename.split(".")[0] + '.json'))


def get_decisions(json_dmn):
    return json_dmn[DEFINITION][DECISIONS]


def get_input_variables(json_dmn):
    input_data = dict()
    for i in json_dmn[DEFINITION][INPUT_DATA]:
        input_data[i[ID]] = i[NAME]

    return input_data


def get_required_variable(info_req):
    try:
        return info_req[REQUIRED_INPUT][LINK]
    except KeyError:
        return info_req[REQUIRED_DECISION][LINK]


def get_new_computable_decisions(decisions, calculated_inputs):
    new_decisions = list()
    for d in decisions:
        if "".join(d[ID].split("#")) not in calculated_inputs:
            informationRequirement = d[INFORMATION_REQUIREMENT]

            if isinstance(informationRequirement, list):
                if all(["".join(get_required_variable(infoReq).split("#")) in calculated_inputs for infoReq in
                        informationRequirement]):
                    new_decisions.append(d)
            elif "".join(get_required_variable(informationRequirement).split("#")) in calculated_inputs:
                new_decisions.append(d)

    return new_decisions


def get_variable_input_data(variable, metadata=None):
    variable_data = dict()
    variable_data[LABEL] = variable[LABEL]
    variable_data['expr'] = variable[INPUT_VARIABLE_EXPRESSION]['text']
    variable_data[VARIABLE_TYPE] = variable[INPUT_VARIABLE_EXPRESSION][VARIABLE_TYPE]

    if variable_data[VARIABLE_TYPE] == "string":
        variable_data['values'] = [i[1:-1] for i in variable[INPUT_VARIABLE_VALUES]['text'].split(',')]
    elif variable_data[VARIABLE_TYPE] == "integer" or variable_data[VARIABLE_TYPE] == "double":
        if metadata:
            start, end, interval = [int(i.strip()) for i in metadata[variable_data[LABEL]].strip()[1:-1].split("..")]
        else:
            sei = input("Please enter the start, end and interval of the allowed values of " + variable_data[
                LABEL] + " for dataset creation (Comma separated. For example 3, 10, 5 where 3 is start, 10 is the "
                           "end and 5 is the interval\n")
            start, end, interval = [int(i.strip()) for i in sei.split(",")]
        variable_data['values'] = [i for i in range(start, end, interval)]
    elif variable_data[VARIABLE_TYPE] == "boolean":
        variable_data['values'] = [0, 1]

    return variable_data


def get_variable_output_data(variable):
    variable_data = dict()
    variable_data[LABEL] = variable[LABEL]
    variable_data[NAME] = variable[NAME]
    variable_data[VARIABLE_TYPE] = variable[VARIABLE_TYPE]
    return variable_data


def get_table_inputs(d):
    input_data = d[DECISION_TABLE][DECISION_INPUTS]
    input_vars = list()
    if isinstance(input_data, list):
        for variable in input_data:
            input_vars.append(variable)
    else:
        input_vars.append(input_data)

    return input_vars


def get_table_outputs(d):
    output_data = d[DECISION_TABLE][DECISION_OUTPUTS]
    output_vars = list()
    if isinstance(output_data, list):
        for variable in output_data:
            output_vars.append(variable)
    else:
        output_vars.append(output_data)

    return output_vars


''' 
A rule has the following structure 
[
    ["inputVar1", "inputVar2", .. , "inputVarN", "Output1", .. "OutputM"]
    ["InValue1", "InValue2", .. , "InValueN", "OutValue1",.. ,"OutValueM"]   
]
'''


def clean_attribute(expr, datatype):
    if datatype == 'string':
        if expr[0] in ["\"", "'"]:
            expr = expr[1:-1]

    return expr


def evaluate_rule(rule_inputs, rule, data):
    if isinstance(rule[INPUT_ENTRY], dict):
        rule[INPUT_ENTRY] = [rule[INPUT_ENTRY]]
    if isinstance(rule[OUTPUT_ENTRY], dict):
        rule[OUTPUT_ENTRY] = [rule[OUTPUT_ENTRY]]
    try:
        assert len(data) == len(rule[INPUT_ENTRY]) and len(rule[INPUT_ENTRY]) == len(rule_inputs)
    except AssertionError:
        print("Incompatible size of rules, inputs, and values!")

    for element, input_data, input_value in zip(data, rule_inputs, rule[INPUT_ENTRY]):
        if not evaluate_decision_column(
                element,
                input_data[INPUT_VARIABLE_EXPRESSION][VARIABLE_TYPE],
                input_value['text']
        ):
            return False
    return True


def evaluate_range(expression, value):
    lower, upper = [i.strip() for i in expression[1:-1].split("..")]
    if expression[0] == "(":
        lower_limit = eval(str(value) + ">" + lower)
    else:
        lower_limit = eval(str(value) + ">=" + lower)

    if expression[-1] == ")":
        upper_limit = eval(str(value) + "<" + upper)
    else:
        upper_limit = eval(str(value) + "<=" + upper)

    return upper_limit * lower_limit


def evaluate_logical_expression(expression, value):
    # Assuming a single logical operator because range option is present
    result = eval(str(value) + expression)
    return result


def evaluate_expression_set(expression, value):
    expression_set = [i for i in expression.strip()[1:-1].split(",")]
    result = str(value) in expression_set
    return result


def evaluate_decision_column(value, data_type, expression):
    expression = clean_attribute(expression, data_type)
    result = False
    if data_type == "integer":  # Patterns: Relational operators, Ranges, Sets
        expression = expression.strip()
        if ".." in expression:
            # It is a range
            result = evaluate_range(expression, value)
        elif ">" in expression or "<" in expression:
            # Assuming other alternative to be a relational expression
            result = evaluate_logical_expression(expression, value)
        elif "," in expression:
            # It is a set
            result = evaluate_expression_set(expression, value)

    elif data_type == "double":
        pass
    elif data_type == "string":  # Assuming string variable will have fixed categories
        result = value == expression
    elif data_type == 'boolean':
        result = value in boolean_values[expression]

    return result


def evaluate_output_expression(expression, variables):
    variable_keys = variables.keys()
    for key in variable_keys:
        key_for_value = key
        key = "".join([i.lower() for i in key.split()])
        occurrences = [i.start() for i in re.finditer(key, expression)]
        if occurrences:
            for occurrence in occurrences:
                expression = expression[: occurrence] + \
                             str(variables[key_for_value]) + \
                             expression[occurrence + len(key):]
    # print(expression)
    value = eval(expression)
    if isinstance(value, float):
        decimal_place = "{:." + str(DECIMAL_PLACE) + "f}"
        value = float(decimal_place.format(value))
    return value


def get_extra_output_columns(decision_rules, decision_inputs_, values):
    new_columns = set()
    input_labels = [i[LABEL].lower() for i in decision_inputs_]
    pattern = r"[A-Za-z]\w+"
    for rule in decision_rules:
        if isinstance(rule[INPUT_ENTRY], dict):
            rule[INPUT_ENTRY] = [rule[INPUT_ENTRY]]
        if isinstance(rule[OUTPUT_ENTRY], dict):
            rule[OUTPUT_ENTRY] = [rule[OUTPUT_ENTRY]]
        for output_entry in rule[OUTPUT_ENTRY]:
            variables = re.findall(pattern, output_entry['text'])
            for variable in variables:
                for key in values.keys():
                    key_for_value = key.lower()
                    key = "".join([i.lower() for i in key.split()])
                    if key == variable and key_for_value not in input_labels:
                        new_columns.add(key_for_value)
    return list(new_columns)


def create_dataset_for_decision(decision_, values):
    if DECISION_TABLE in decision_.keys():
        rules = decision[DECISION_TABLE][RULES]
        if isinstance(rules, dict):
            rules = [rules]
        output_datasets = create_dataset_for_decision_table(decision, rules, values)
    else:
        literal = decision[DECISION_LITERAL_EXPRESSION]
        output_datasets = create_dataset_for_decision_literal(decision, literal, values)
    print("Writing datasets for decision: " + decision_[NAME])
    write_dataset_to_csv(output_datasets)
    print("Written datasets: " + decision_[NAME])
    return output_datasets


"""
Returns output datasets for each output variable in the decision table
"""


def create_dataset_for_decision_table(decision_, decision_rules, values):
    output_datasets = dict()
    decision_inputs_ = decision_[DECISION_TABLE][DECISION_INPUTS]
    if not isinstance(decision_inputs_, list):
        decision_inputs_ = [decision_inputs_]

    decision_outputs_ = decision_[DECISION_TABLE][DECISION_OUTPUTS]

    if not isinstance(decision_outputs_, list):
        decision_outputs_ = [decision_outputs_]

    input_lists = [values[i[LABEL].lower()]['values'] for i in decision_inputs_]
    extra_output_columns = get_extra_output_columns(decision_rules, decision_inputs_, values)
    for _ in extra_output_columns:
        input_lists.append(values[_]['values'])
    # print(input_lists)
    for output in decision_outputs_:
        output_datasets[output[LABEL].lower()] = [
                [i[LABEL] for i in decision_inputs_] +
                extra_output_columns +
                [output[LABEL]]
            ]
        values[output[LABEL].lower()]['values'] = set()

    for elements in itertools.product(*input_lists):
        for decision_rule in decision_rules:
            variable_dict = dict()
            for element, column in zip(elements[:len(decision_inputs_)], decision_inputs_):
                variable_dict[column[LABEL].lower()] = element
            for element, column in zip(elements[len(decision_inputs_):], extra_output_columns):
                variable_dict[column.lower()] = element
            if evaluate_rule(decision_inputs_, decision_rule, elements[:len(decision_inputs_)]):
                for column_value, column_name in zip(decision_rule[OUTPUT_ENTRY], decision_outputs_):
                    row = list(elements)
                    output_value = evaluate_output_expression(column_value['text'], variable_dict)
                    values[column_name[LABEL].lower()]['values'].add(output_value)
                    output_datasets[column_name[LABEL].lower()].append(row+[output_value])

                break  # Assuming only single rules matches
    for output in decision_outputs_:
        values[output[LABEL].lower()]['values'] = list(values[output[LABEL].lower()]['values'])
    return output_datasets


def create_dataset_for_decision_literal(decision_, decision_literal, values):
    output_datasets = dict()
    input_lists = list()
    information_requirements = decision_[INFORMATION_REQUIREMENT]
    if not isinstance(information_requirements, list):
        information_requirements = [information_requirements]

    for information_requirement in information_requirements:
        input_list_variable = \
            input_variables["".join(get_required_variable(information_requirement[REQUIRED_INPUT]).split("#"))]
        input_lists.append(values[input_list_variable.lower()])

    output_datasets[decision_[NAME]] = [
        [input_variables["".join(get_required_variable(i[REQUIRED_INPUT]).split("#"))]
         for i in information_requirements]
        + [decision_[NAME]]
    ]

    for elements in itertools.product(*input_lists):
        literal_expression = decision_literal['text']
        variables = {
            input_variables["".join(get_required_variable(i[REQUIRED_INPUT]).split("#"))]: value
            for i, value in zip(information_requirements, elements)
        }
        row = list(elements)
        row.append(evaluate_output_expression(literal_expression, variables))
        output_datasets[decision_[NAME]].append(row)
    return output_datasets


def write_dataset_to_csv(output_datasets):
    for filename, data_rows in output_datasets.items():
        with open(filename + ".csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(data_rows[0])
            for row in data_rows[1:]:
                writer.writerow(row)


if __name__ == "__main__":
    dmn = load_dmn_file_as_json('Datafiles/insurance.dmn')
    metadata_file = json.load(open('Datafiles/metadata.json'))
    all_decisions = get_decisions(dmn)
    input_variables = get_input_variables(dmn)
    new_computable_decisions = get_new_computable_decisions(all_decisions, list(input_variables.keys()))
    table_variables = dict()
    datasets = dict()
    while len(new_computable_decisions) > 0:
        print("New computable decisions are: ")
        for decision in new_computable_decisions:

            decision_inputs = get_table_inputs(decision)
            decision_outputs = get_table_outputs(decision)
            for di in decision_inputs:
                if di[LABEL] not in table_variables.keys():
                    table_variables[di[LABEL].lower()] = get_variable_input_data(di, metadata_file)
            for do in decision_outputs:
                if do[LABEL] not in table_variables.keys():
                    table_variables[do[LABEL].lower()] = get_variable_output_data(do)

            input_variables[decision[ID]] = decision[NAME]
            datasets[decision[NAME].lower()] = create_dataset_for_decision(decision, table_variables)

        new_computable_decisions = get_new_computable_decisions(all_decisions, list(input_variables.keys()))
