import pandas as pd

dict_for_map_many_func = {
    "alzheimer": "gen_mols_alzheimer",
    "dyslipidemia": "gen_mols_dyslipidemia",
    "lung cancer": "gen_mols_lung_cancer",
    "sclerosis": "gen_mols_multiple_sclerosis",
    "Drug_Resistance": "gen_mols_acquired_drug_resistance",
    "Parkinson": "gen_mols_parkinson",
    "nothing": "nothing",
}


def validate_decompose(
    idx: int, decompose_lst: list, validation_path="experiment3.xlsx"
) -> bool:

    lines = pd.read_excel(validation_path)
    columns = lines.columns
    lines = lines.values.tolist()

    num_tasks_true = len(lines[idx][0].split(","))
    lines[idx][2] = decompose_lst

    if len(decompose_lst) == num_tasks_true:
        lines[idx][3] = True

        pd.DataFrame(lines, columns=columns).to_excel(validation_path, index=False)
        return True
    else:
        lines[idx][3] = False
        pd.DataFrame(lines, columns=columns).to_excel(validation_path, index=False)
        return False


def validate_conductor(
    idx: int,
    func: dict,
    sub_task_number: int,
    path_total_val="experiment3_example.xlsx",
) -> bool:
    """
    Validate conductors agent answer. File must consist of next columns =
    'case', 'content', 'decomposers_tasks', 'is_correct_context', 'task 1',
    'task 2', 'task 3', 'task 4', 'task 5'

    Parameters
    ----------
    idx : int
        Number of line for validation
    func : dict
        Dict with function name and parameters
    sub_task_number : int
        Number of subtask (from decompose agent)

    Returns
    -------
    answer : bool
        Validation passed or not
    """
    lines = pd.read_excel(path_total_val)
    columns = lines.columns
    lines = lines.values.tolist()

    try:
        target_name = lines[idx][0].split(", ")[sub_task_number]
    except:
        target_name = "nothing"
    if isinstance(func, bool):
        return False

    # if call chat model for answer in free form (сos no such case exists in the file)
    if func["name"].replace(" ", "") == "make_answer_chat_model":
        lines[idx][4 + sub_task_number] = func["name"].replace(" ", "")
        pd.DataFrame(
            lines,
            columns=columns,
        ).to_excel(path_total_val, index=False)
        return False
    else:
        if func["name"].replace(" ", "") == dict_for_map_many_func[target_name]:
            lines[idx][4 + sub_task_number] = func["name"].replace(" ", "")
            pd.DataFrame(
                lines,
                columns=columns,
            ).to_excel(path_total_val, index=False)
            return True
        else:
            lines[idx][4 + sub_task_number] = func["name"].replace(" ", "")
            pd.DataFrame(
                lines,
                columns=columns,
            ).to_excel(path_total_val, index=False)
            return False


def compute_metrics(
    model_name: str = "no_name_model", file_path: str = "experiment3_example.xlsx"
):
    """
    Compute pipeline metrics

    Parameters
    ----------

    file_path : str
        Path to excel file with next columns:
        case, content, decompose_tasks, is_correct, task 1, task 2, task 3, task 4
    model_name : str
        Name of model with which testing was carried out
    """
    just_1_case_in_all_smpls = True
    dfrm = pd.read_excel(file_path)

    number_subtasks = 0
    number_tasks = 0

    correct_subtasks = 0
    correct_tasks = 0

    decomposer_true = 0

    # add zeros columns for result
    dfrm["conductors_score"] = 0
    dfrm["score_from"] = 0
    dfrm["total_score"] = 0
    columns = dfrm.columns

    lst = dfrm.values.tolist()

    for row in lst:
        try:
            cases = (
                row[0]
                .replace("Parkinson ", "Parkinson")
                .replace("Drug_Resistance ", "Drug_Resistance")
                .split(", ")
            )
            decomposer_true += row[3]

            row[11 - 1] = len(cases)

            # for every subtask in main task(query)
            for n, case in enumerate(cases):
                is_correct = dict_for_map_many_func[case] == row[4 + n]
                row[10 - 1], correct_subtasks = (
                    row[10 - 1] + int(is_correct),
                    correct_subtasks + int(is_correct),
                )

            # if all subtasks are defined correctly
            if row[10 - 1] == row[11 - 1]:
                correct_tasks += 1
                row[12 - 1] = 1
            else:
                row[12 - 1] = 0

            if just_1_case_in_all_smpls:
                if len(cases) > 1:
                    just_1_case_in_all_smpls = False

            number_subtasks, number_tasks = (
                number_subtasks + len(cases),
                number_tasks + 1,
            )

        except:
            continue

    pd.DataFrame(lst, columns=columns).to_excel(f"result.xlsx", index=False)

    if not (just_1_case_in_all_smpls):
        print(
            "Percentage true subtasks (accuracy of whole pipeline): ",
            100 / (number_subtasks) * correct_subtasks,
        )
        print(
            "Percentage true tasks by Decomposer: ",
            100 / number_tasks * decomposer_true,
        )
        print(
            "Percentage true tasks by Conductor: ",
            100 / (number_subtasks) * correct_subtasks,
        )
    else:
        print("Percentage true tasks: ", 100 / (number_tasks) * correct_tasks)
