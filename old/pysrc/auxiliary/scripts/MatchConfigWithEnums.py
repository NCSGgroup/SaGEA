"""
This script is to associate the string in the configuration file with the enumeration type in the program,
and use the corresponding function in the configuration program to pass parameters.
"""


def match_config(json_dict, json_keys, enum_classes, setting_functions):
    """

    :param json_dict:
    :param json_keys:
    :param enum_classes:
    :param setting_functions:
    :return:
    """
    for i in range(len(enum_classes)):
        this_enum_class = enum_classes[i]
        this_json_key = json_keys[i]
        this_setting_function = setting_functions[i]

        input_parameter = json_dict[this_json_key]
        optional_parameters = this_enum_class.__members__.keys()
        assert input_parameter in optional_parameters  # ensure that the parameters in the JSON file are legal

        this_setting_function(this_enum_class.__members__[input_parameter])

