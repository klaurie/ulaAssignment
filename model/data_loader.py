import pandas as pd
import numpy as np


def load_data(model, input_file_name: str):
    """
    Load input, student, studio, reference sheets from excel file into corresponding dataframes

    Parameters:
        file_name (str): Excel sheet containing input data
    Returns:
        df: ULA information
        df: studio information
        df: optimization weights from input
        df: reference info for ULA and studio info
    """
    student_info = pd.read_excel(input_file_name, sheet_name="Student Info")
    studio_info = pd.read_excel(input_file_name, sheet_name="Studio Info")
    weights = pd.read_excel(input_file_name, sheet_name="User Input")
    reference = pd.read_excel(input_file_name, sheet_name="Reference")

    return studio_info, student_info, weights, reference


def calculate_unique_counts(model):
    """
    Calculates the number of unique values in each specified column and assigns them to instance variables.
    """
    index_columns = {
        'Studio Index': 'num_start_times',
        'Gender Index': 'num_genders',
        'School Index': 'num_schools',
        'Major Index': 'num_majors',
        'Language Index': 'num_languages',
        'Race/Ethnicity Index': 'num_races',
        'Course Index': 'num_courses',
    }

    for column, attr_name in index_columns.items():
        try:
            unique_values = pd.unique(model.reference[column])
            count = len(unique_values[~np.isnan(unique_values)]) if np.issubdtype(unique_values.dtype,
                                                                                  np.number) else len(unique_values)
        except KeyError:
            count = 0  # In case the column is missing

        setattr(model, attr_name, count)


def categorize_studios(model):
    S1 = model.studio_info.index[model.studio_info['1 ULA?'] == 1]  # this is a list of the studio section number-1 that have 1 ULA
    S2 = model.studio_info.index[
        model.studio_info['1 ULA?'] == 0]  # this is a list of the studio section number-1 that have 2 ULA's
    S3 = model.studio_info.index[
        model.studio_info['Language'] != -1]  # this is a list of the studio section number-1 that have a language specified

    return S1, S2, S3


def categorize_student_years(model):
    seniors = (model.student_info['Year'] == 4).astype(int)
    juniors = (model.student_info['Year'] == 3).astype(int)
    sophomores = (model.student_info['Year'] == 2).astype(int)
    freshman = (model.student_info['Year'] == 1).astype(int)

    return freshman, sophomores, juniors, seniors


def create_studio_time_binary(model):
    studio_start_times_binary = pd.DataFrame(np.zeros((model.num_studios, model.num_start_times)))
    for i in range(model.num_studios):
        studio_start_times_binary.iloc[i, model.studio_info.loc[i, 'Studio Time']] = 1

    return studio_start_times_binary


def create_instructor_studio_binary(model):
    instruct = pd.DataFrame(np.zeros((model.num_studios, model.num_instructors)))
    for i in range(model.num_studios):
        instruct.iloc[i, int(np.where(model.unique_instructors == model.studio_info.loc[i, 'Instructor Email'])[0])] = 1

    return instruct

def create_engr_103_lang_binary(model):
    lang = pd.DataFrame()
    if (len(model.S3) > 0):
        lang = pd.DataFrame(np.zeros((len(model.S3), model.num_languages)))
        lang.index = model.S3
        for i in model.S3:
            lang.loc[i, model.studio_info.loc[i, 'Language']] = 1

    return lang


def create_student_exp_binary(model):
    lan = pd.DataFrame()
    if len(model.S3) > 0:
        lan = pd.DataFrame(np.zeros((model.num_ulas, model.num_languages)))
        temp = model.student_info['Languages']
        for i in range(model.num_ulas):
            if str(temp[i]).isdigit():
                if temp[i] != -1:
                    lan.loc[i, temp[i]] = 1
            elif temp[i] != -1:
                temp1 = temp[i].split(',')
                for j in range(len(temp1)):
                    lan.loc[i, int(temp1[j])] = 1

    return lan


def create_gender_binary(model):
    gen = pd.DataFrame(np.zeros((model.num_ulas, model.num_genders)))
    for i in range(model.num_ulas):
        gen.loc[i, model.student_info.loc[i, 'Gender']] = 1

    return gen


def create_race_binary(model):
    race = pd.DataFrame(np.zeros((model.num_ulas, model.num_races)))
    for i in range(model.num_ulas):
        race.loc[i, model.student_info.loc[i, 'Race']] = 1

    return race


def create_major_binary(model):
    maj = pd.DataFrame(np.zeros((model.num_ulas, model.num_majors)))
    for i in range(model.num_ulas):
        maj.loc[i, model.student_info.loc[i, 'Major']] = 1

    return maj


def create_school_binary(model):
    school = pd.DataFrame(np.zeros((model.num_ulas, model.num_schools)))
    for i in range(model.num_ulas):
        school.loc[i, model.student_info.loc[i, 'School']] = 1

    return school

def create_instruct_pref_list(model):
    # Create a dictionary to map student emails to indices
    email_to_index = {email: idx for idx, email in enumerate(model.student_info['Email'])}
    for instructor in range(model.num_instructors):
        model.instructor_sections[instructor] = model.instruct.index[model.instruct.iloc[:, instructor] == 1].tolist()
        # Get one location in studio_info where the instructor is listed
        preferred_ulas = \
            model.studio_info.loc[model.studio_info['Instructor Email'] == model.unique_instructors[instructor], 'Preferences'].iloc[
                0]
        if preferred_ulas != 'None':
            preferred_ulas_list = [email_to_index[email.strip()] for email in preferred_ulas.split(',') if
                                   email.strip() in email_to_index]
            model.instructor_preferences[model.unique_instructors[instructor]] = preferred_ulas_list
        else:
            model.instructor_preferences[model.unique_instructors[instructor]] = []


def create_instruct_pref_binary(model):
    temp = model.studio_info.index[model.studio_info['Preferences'] != 'None']
    temp = model.studio_info[model.studio_info['Preferences'] != 'None']
    temp = temp.drop_duplicates(subset=['Instructor Email'])

    # use this to refer to which preferences were not satisfied in output
    pref = pd.DataFrame(np.zeros((model.num_instructors, model.num_ulas)))
    if temp.shape[0] > 0:
        for i in temp.index:
            temp1 = model.studio_info.loc[i, 'Preferences'].split(',')
            for j in range(len(temp1)):
                if temp1[j] not in model.student_info['Email'].values:
                    # notify of preferences that arent in student info
                    print(
                        f"Email {temp1[j]} preferred by instructor {temp.loc[i, 'Instructor Email']} is not listed in student emails.")
                else:
                    pref.iloc[int(np.where(model.unique_instructors == model.studio_info.loc[i, 'Instructor Email'])[0]),
                    model.student_info.index[model.student_info['Email'] == temp1[j]].tolist()[0]] = 1

    return pref

