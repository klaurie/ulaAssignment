import pandas as pd
import numpy as np
from . import data_loader as dl
from . import utils
import time
import sys
import json
import pickle

import gurobipy as gp
from gurobipy import GRB

class ULAModel:
    def __init__(self, input_file_name: str, output_ULA_file_name: str, output_studio_file_name: str, model_name: str):
            self.output_ULA_file_name = output_ULA_file_name
            self.output_studio_file_name = output_studio_file_name

            self.start_preprocess = time.time()
            self.studio_info, self.student_info, self.weights, self.reference = dl.load_data(self, input_file_name)
            self.studio_info = self.studio_info.fillna('None')

            # Identify counts of all inputs
            self.num_ulas = self.student_info.shape[0]
            self.num_studios = self.studio_info.shape[0]
            self.unique_instructors = pd.unique(self.studio_info['Instructor Email'])
            self.num_instructors = len(self.unique_instructors)

            # Initialize counts
            self.num_start_times = 0
            self.num_genders = 0
            self.num_schools = 0
            self.num_majors = 0
            self.num_languages = 0
            self.num_races = 0
            self.num_courses = 0

            dl.calculate_unique_counts(self)

            # Categorize Different Sets of Studios
            self.S1, self.S2, self.S3 = dl.categorize_studios(self)

            # Categorize ULAs into their respective years
            self.freshman, self.sophomores, self.juniors, self.seniors = dl.categorize_student_years(self)

            # studio start binary
            self.studio_start_times_binary = dl.create_studio_time_binary(self)
            
            # instructors associated studios binary
            self.instruct = dl.create_instructor_studio_binary(self)

            # engr 103 recommended language experience
            self.lang = dl.create_engr_103_lang_binary(self)

            # languages students have experience in
            self.lan = dl.create_student_exp_binary(self)

            self.gen = dl.create_gender_binary(self)

            self.race = dl.create_race_binary(self)

            self.maj = dl.create_major_binary(self)

            self.school = dl.create_school_binary(self)

            self.instructor_sections = {}
            self.instructor_preferences = {}

            dl.create_instruct_pref_list(self)

            self.pref = dl.create_instruct_pref_binary(self)
            print(self.instructor_preferences)
            self.m = gp.Model(model_name)

            self.end_preprocess = time.time()


    def define_variables(self):
        #############################################################################################
        ## Variables:                                                                              ##
        ## x : ULA assigned to studio               eps_g : only 1 gender assigned to instructor   ##
        ## h : ULA hired                            eps_r : only 1 race assigned to instructor     ##
        ## alpha : ULA assigned to instructor       eps_m : only 1 major assigned to instructor    ##
        ## w : instructor has >=1 of gender         eps_p : instructor not assigned pref           ##
        ## y : instructor has >=1 of race           eps_C : ULA not assigned course pref           ##
        ## z : instructor has >=1 of major          eps_hc : non-HC ULAs assigned HC studio        ##
        ## N_major : hired ULAs major count         eps_E :  non-E ULAs assigned E studio          ##
        ## N_school : hired ULAs school count       eps_L : hired ULAs with non-match lang req     ##
        ## P_major : linearization var              P_school : linearization var                   ##
        ## eps_1, eps_2, eps_3, eps_4 : ULAs of year n not hired                                   ##
        #############################################################################################
        self.define_assignment_variables()
        self.define_instructor_variables()
        self.define_preference_variables()
        self.define_mismatch_variables()
        self.define_linearization_variables()
        self.define_hiring_status_variables()

    # x, h, alpha
    def define_assignment_variables(self):
        ## Binary variables to indicate whether ULA i assigned to studio s ##
        self.x = (self.m.addVars([a for a in list(range(self.num_ulas))],
                            [a for a in list(range(self.num_studios))],
                            ub=1, vtype=GRB.BINARY, name='x'))

        ## Binary variables to indicate whether to hire ULA i or not ##
        self.h = (self.m.addVars([a for a in list(range(self.num_ulas))],
                       ub=1, vtype=GRB.BINARY, name='h'))

        ## Binary variables to indicate whether ULA i assigned to instructor j ##
        self.alpha = (self.m.addVars([a for a in list(range(self.num_ulas))],
                           [a for a in list(range(self.num_instructors))],
                           ub=1, vtype=GRB.BINARY, name='alpha'))
        
    # w, eps_g, y, eps_r, z, eps_m, 
    def define_instructor_variables(self):
        ## Binary variables to indicate if instructor j has >=1 ULA of gender g ##
        self.w = (self.m.addVars([a for a in list(range(self.num_instructors))],
                       [a for a in list(range(self.num_genders))],
                       ub=1, vtype=GRB.BINARY, name='w'))

        ## Binary variables to indicate if only 1 type of gender is assigned to instructor j ##
        self.eps_g = (self.m.addVars([a for a in list(range(self.num_instructors))],
                           ub=1, vtype=GRB.BINARY, name='eps_g'))

        ## Binary variables to indicate if instructor j has >=1 ULA of race r ##
        self.y = (self.m.addVars([a for a in list(range(self.num_instructors))],
                       [a for a in list(range(self.num_races))],
                       ub=1, vtype=GRB.BINARY, name='y'))
        
        ## Binary variables to indicate if only 1 type of race is assigned to instructor j ##
        self.eps_r = (self.m.addVars([a for a in list(range(self.num_instructors))],
                           ub=1, vtype=GRB.BINARY, name='eps_r'))
        
        ## Binary variables to indicate if instructor j has >=1 ULA of major m ##
        self.z = (self.m.addVars([a for a in list(range(self.num_instructors))],
                       [a for a in list(range(self.num_majors))],
                       ub=1, vtype=GRB.BINARY, name='z'))
        
        ## Binary variables to indicate if only 1 type of major is assigned to instructor j ##
        self.eps_m = (self.m.addVars([a for a in list(range(self.num_instructors))],
                           ub=1, vtype=GRB.BINARY, name='eps_m'))

    # eps_p, eps_C
    def define_preference_variables(self):
        ## Binary variables to indicate if instructor $j$ prefers ULA $i$ but is not assigned that ULA ##
        self.eps_p = (self.m.addVars([a for a in list(range(self.num_ulas))],
                           [a for a in list(range(self.num_instructors))],
                           ub=1, vtype=GRB.BINARY, name='eps_p'))
        
        ## Binary variable to indicate if ULA is assigned to a course that is not their preferred course number ##
        self.eps_C = (self.m.addVars([a for a in list(range(self.num_ulas))], ub=1, vtype=GRB.BINARY, name='eps_C'))

    # eps_hc, eps_E, eps_L
    def define_mismatch_variables(self):
        ## Number of non-HC ULAS's assigned to HC studio s ##
        self.eps_hc = (self.m.addVars([a for a in list(range(self.num_studios))],
                            ub=1, vtype=GRB.CONTINUOUS, name='eps_hc'))

        ## Number of ULA's who prefer e-campus assigned to non-ecampus studio s ##
        self.eps_E = (self.m.addVars([a for a in list(range(self.num_studios))],
                           vtype=GRB.CONTINUOUS, name='eps_E'))

        ## Number of studios without matching language requirement ##
        self.eps_L = (self.m.addVars([a for a in list(self.S3)],
                           vtype=GRB.CONTINUOUS, name='eps_L'))

    # P_major, P_school
    def define_linearization_variables(self):
        ## Linearization variable ##
        self.P_major = (self.m.addVars([a for a in list(range(self.num_majors))],
                             [a for a in list(range(self.num_majors))],
                             vtype=GRB.CONTINUOUS, name='P_major'))

        ## Linearization variable ##
        self.P_school = (self.m.addVars([a for a in list(range(self.num_schools))],
                              [a for a in list(range(self.num_schools))],
                              vtype=GRB.CONTINUOUS, name='P_school'))

    # N_maj, N_schoo, eps_U, eps_(1, 2, 3, 4)
    def define_hiring_status_variables(self):
        # Variables for ULA and class year hiring status
        ## Number of ULA's hired that are major m ##
        self.N_maj = (self.m.addVars([a for a in list(range(self.num_majors))],
                           vtype=GRB.CONTINUOUS, name='N_maj'))

        ## Number of ULA's hired that are school s ##
        self.N_school = (self.m.addVars([a for a in list(range(self.num_schools))],
                              vtype=GRB.CONTINUOUS, name='N_school'))
        
        ## Number of returning ULAs not hired ##
        self.eps_U = (self.m.addVars([a for a in list(range(self.num_ulas))], ub=1, vtype=GRB.BINARY, name='eps_U'))
        
        ## Number of Seniors not hired ##
        self.eps_4 = (self.m.addVars([a for a in list(range(self.num_ulas))], ub=1, vtype=GRB.BINARY, name='eps_4'))

        ## Number of Juniors not hired ##
        self.eps_3 = (self.m.addVars([a for a in list(range(self.num_ulas))], ub=1, vtype=GRB.BINARY, name='eps_3'))

        ## Number of Sophomores not hired ##
        self.eps_2 = (self.m.addVars([a for a in list(range(self.num_ulas))], ub=1, vtype=GRB.BINARY, name='eps_2'))

        ## Number of Freshman not hired ##
        self.eps_1 = (self.m.addVars([a for a in list(range(self.num_ulas))], ub=1, vtype=GRB.BINARY, name='eps_1'))


    def set_objective(self):
        # This is multi-objective function weighted by user inputs
        self.m.setObjective(gp.quicksum((10 ** (int(6 - self.weights.iloc[1, 2]))) * self.eps_L[s] for s in self.S3) +
                       gp.quicksum((10 ** (int(6 - self.weights.iloc[2, 2]))) * self.eps_hc[s] +
                                   (10 ** (int(6 - self.weights.iloc[3, 2]))) * self.eps_E[s] for s in range(self.num_studios)) +
                       gp.quicksum((10 ** (int(6 - self.weights.iloc[4, 2]))) * self.eps_g[j] +
                                   (10 ** (int(6 - self.weights.iloc[5, 2]))) * self.eps_r[j] +
                                   (10 ** (int(6 - self.weights.iloc[6, 2]))) * self.eps_m[j] +
                       gp.quicksum((10 ** (int(6 - self.weights.iloc[0, 2]))) * self.eps_p[i, j] for i in range(self.num_ulas))
                                   for j in range(self.num_instructors)) +
                                   (10 ** (int(6 - self.weights.iloc[7, 2]))) * gp.quicksum(
                       gp.quicksum(self.P_major[m1, m2] for m2 in range(m1 + 1, self.num_majors))
                                   for m1 in range(self.num_majors)) +
                                    (10 ** (int(6 - self.weights.iloc[8, 2]))) * gp.quicksum(
                       gp.quicksum(self.P_school[k1, k2] for k2 in range(k1 + 1, self.num_schools))
                                    for k1 in range(self.num_schools)) +
                       gp.quicksum((10 ** (int(6 - self.weights.iloc[10, 2]))) * self.eps_C[i] +
                                   (10 ** (int(6 - self.weights.iloc[9, 2]))) * self.eps_U[i] +
                                   (10 ** (int(6 - self.weights.iloc[11, 2]))) * self.eps_4[i] +
                                   (10 ** (int(6 - self.weights.iloc[12, 2]))) * self.eps_3[i] +
                                   (10 ** (int(6 - self.weights.iloc[13, 2]))) * self.eps_2[i] +
                                   (10 ** (int(6 - self.weights.iloc[14, 2]))) * self.eps_1[i] for i in range(self.num_ulas)))
 

    def add_constraints(self):
        # Call individual constraint methods
        self.constraint_hiring()
        self.constraint_demographics()
        self.constraint_studio_mismatch()
        self.constraint_instructor()
        self.constraint_preference()
        self.constraint_demographics()
        self.constraint_linearization()
        self.constraint_time_conflict()
        self.constraint_grade_level_assignment()
    
    # rerunConstraint, Only2_Studios, OneULA_Only_Studios, TwoULA_Studios, ULA_Availability, eps_U_constraints
    def constraint_hiring(self):
        # TODO: this is for rerun, need to adjust 
        # Constraints to ensure studios with positions filled are not assigned a new ULA
        # (1 - student_info. loc[i, "Not Available"]) = 0 if not available, because we dont want to change them
        # (1 - studio_info.loc[s, "Position Filled"]) = 0 if studio needs no new ULA
        #self.m.addConstrs((self.x[i,s] <= (1 - self.student_info. loc[i, "Not Available"]) * (1 - self.studio_info.loc[s, "Position Filled"])
        #             for i in range(self.num_ulas) for s in range(self.num_studios)), "rerunConstraint")

        # Constraints ensuring each hired ULA is assigned to exactly 2 studios
        self.m.addConstrs((gp.quicksum(self.x[i, s] for s in range(self.num_studios)) == 2 * self.h[i]
                      for i in range(self.num_ulas)), "Only2_Studios")
        
        # Constraints ensures that each studio that needs only one ULA is assigned a single ULA.
        self.m.addConstrs((gp.quicksum(self.x[i, s] for i in range(self.num_ulas)) == 1
                      for s in self.S1), "OneULA_Only_Studios")

        # Constraints ensures that each studio that needs two ULAs is assigned a two ULAs
        self.m.addConstrs((gp.quicksum(self.x[i, s] for i in range(self.num_ulas)) == 2
                      for s in self.S2), "TwoULA_Studios")

        # Constraints ensures that you can only assign a ULA to a studio if they
        # are available when that studio is being offered
        self.m.addConstrs((self.x[i, s] <= self.student_info.loc[i, self.studio_info.loc[s, 'Studio Time']]
                      for i in range(self.num_ulas)
                      for s in range(self.num_studios)), "ULA_Availability")
        
        # This ensures eps_U takes on appropriate value
        self.m.addConstrs((self.eps_U[i] == (1 - self.h[i]) * self.student_info.loc[i, 'Return ULA'] for i in range(self.num_ulas)), "eps_U_constraints")

   
    # only_one_instructor_per_constraints, w_constraints, y_constraints, z_constraints
    def constraint_instructor(self):
        # Next three constraints ensures that each ULA is assigned to
        # AT MOST 1 instructor and the alpha's take on appropriate values.
        self.m.addConstrs((gp.quicksum(self.x[i, s] * self.instruct.iloc[s, j] for s in range(self.num_studios)) >= self.alpha[i, j]
                      for i in range(self.num_ulas)
                      for j in range(self.num_instructors)), "Only1_InstructorPerULA_1")

        self.m.addConstrs((gp.quicksum(self.x[i, s] * self.instruct.iloc[s, j] for s in range(self.num_studios)) <= 10000 * self.alpha[i, j]
                      for i in range(self.num_ulas)
                      for j in range(self.num_instructors)), "Only1_InstructorPerULA_2")

        self.m.addConstrs((gp.quicksum(self.alpha[i, j] for j in range(self.num_instructors)) <= 1
                      for i in range(self.num_ulas)), "Only1_InstructorPerULA_3")
        
        # Constraints to ensure that w variables take on appropriate values
        self.m.addConstrs((gp.quicksum(self.alpha[i, j] * self.gen.iloc[i, g] for i in range(self.num_ulas)) >= self.w[j, g]
                      for j in range(self.num_instructors)
                      for g in range(self.num_genders)), "w_constraints")


        # Constraints to ensure that y variables take on appropriate values
        self.m.addConstrs((gp.quicksum(self.alpha[i, j] * self.race.iloc[i, r] for i in range(self.num_ulas)) >= self.y[j, r]
                      for j in range(self.num_instructors)
                      for r in range(self.num_races)), "y_constraints")


        # Constraints to ensure that z variables take on appropriate values
        self.m.addConstrs((gp.quicksum(self.alpha[i, j] * self.maj.iloc[i, m] for i in range(self.num_ulas)) >= self.z[j, m]
                      for j in range(self.num_instructors)
                      for m in range(self.num_majors)), "z_constraints")

    # OnCampus_constraints, eps_L_constraints, , eps_hc_constraints, eps_E_constraints
    def constraint_studio_mismatch(self):
        # This constraint ensures that if studio $s$ is on-campus ($oc_s=1$)
        # then ULA $i$ cannot be assigned to that studio if they are not 
        # available for an on-campus course
        self.m.addConstrs((self.x[i, s] <= 1 - (1 - self.studio_info.loc[s, 'Ecampus']) * (self.student_info.loc[i, 'Ecampus'])
                      for i in range(self.num_ulas)
                      for s in range(self.num_studios)), "OnCampus_Constraints")

        # This ensures eps_L takes on appropriate value
        self.m.addConstrs((self.eps_L[s] == gp.quicksum(
            self.x[i, s] * gp.quicksum(self.lang.iloc[s - min(self.S3), l] * (1 - self.lan.iloc[i, l]) for l in range(self.num_languages))
            for i in range(self.num_ulas)) for s in self.S3), "eps_L_constraints")

        # This ensures eps_hc takes on appropriate value
        self.m.addConstrs((self.eps_hc[s] == self.studio_info.loc[s, 'HC'] * gp.quicksum(self.x[i, s] * (1 - self.student_info.loc[i, 'HC'])
                                                                          for i in range(self.num_ulas))
                      for s in range(self.num_studios)), "eps_hc_constraints")

        # This ensures eps_E takes on appropriate value
        self.m.addConstrs(
            (self.eps_E[s] == (1 - self.studio_info.loc[s, 'Ecampus']) * gp.quicksum(self.x[i, s] * self.student_info.loc[i, 'Ecampus']
                                                                           for i in range(self.num_ulas))
             for s in range(self.num_studios)), "eps_E_constraints")

    # eps_g_constraints, eps_r_constraints, N_maj_constraints, N_school_constraints, eps_m_constraints
    def constraint_demographics(self):   
        # Constraint 1 to ensure that eps_g variables take on appropriate values
        self.m.addConstrs((2 - self.eps_g[j] <= gp.quicksum(self.w[j, g] for g in range(self.num_genders))
                      for j in range(self.num_instructors)), "eps_g_constraints1")

        # Constraint 2 to ensure that eps_g variables take on appropriate values
        self.m.addConstrs((gp.quicksum(self.w[j, g] for g in range(self.num_genders)) <= 1 + 10 * (1 - self.eps_g[j])
                      for j in range(self.num_instructors)), "eps_g_constraints2")   
        
        # Constraint 1 to ensure that eps_r variables take on appropriate values
        self.m.addConstrs((2 - self.eps_r[j] <= gp.quicksum(self.y[j, r] for r in range(self.num_races))
                      for j in range(self.num_instructors)), "eps_r_constraints1")

        # Constraint 2 to ensure that eps_r variables take on appropriate values
        self.m.addConstrs((gp.quicksum(self.y[j, r] for r in range(self.num_races)) <= 1 + 10 * (1 - self.eps_r[j])
                      for j in range(self.num_instructors)), "eps_r_constraints2")    
        
        # Constraint counts the number of major m that are hired across all courses
        self.m.addConstrs((self.N_maj[m] == gp.quicksum(self.h[i] * self.maj.iloc[i, m] for i in range(self.num_ulas))
                      for m in range(self.num_majors)), "N_maj_constraints")
      
        # Constraint counts the number of school k that are hired across all courses
        self.m.addConstrs((self.N_school[k] == gp.quicksum(self.h[i] * self.school.iloc[i, k] for i in range(self.num_ulas))
                      for k in range(self.num_schools)), "N_school_constraints")
        
        # Constraint 1 to ensure that eps_m variables take on appropriate values
        self.m.addConstrs((2 - self.eps_m[j] <= gp.quicksum(self.z[j, m] for m in range(self.num_majors))
                      for j in range(self.num_instructors)), "eps_m_constraints1")

        # Constraint 2 to ensure that eps_m variables take on appropriate values
        self.m.addConstrs((gp.quicksum(self.z[j, m] for m in range(self.num_majors)) <= 1 + 10 * (1 - self.eps_m[j])
                      for j in range(self.num_instructors)), "eps_m_constraints2")

    # studio_time_constraints
    def constraint_time_conflict(self):
         # Constraint to ensure that the two studios ULAs are assigned to are not at the same time.
        self.m.addConstrs((gp.quicksum(self.x[i, s1] * self.x[i, s2] * (self.studio_info['Studio Time'].iloc[s1] == self.studio_info['Studio Time'].iloc[s2])
                                for s1 in range(self.num_studios)
                                for s2 in range(self.num_studios) if s1 != s2) == 0
                    for i in range(self.num_ulas)), "studio_time_constraints")

    # eps_(1, 2, 3, 4)_constraints
    def constraint_grade_level_assignment(self):
        # Constraints ensuring grade levels are assigned correctly
        self.m.addConstrs((self.eps_4[i] == (1 - self.h[i]) * self.seniors[i] for i in range(self.num_ulas)), "eps_4_constraints")
        self.m.addConstrs((self.eps_3[i] == (1 - self.h[i]) * self.juniors[i] for i in range(self.num_ulas)), "eps_3_constraints")
        self.m.addConstrs((self.eps_2[i] == (1 - self.h[i]) * self.sophomores[i] for i in range(self.num_ulas)), "eps_2_constraints")
        self.m.addConstrs((self.eps_1[i] == (1 - self.h[i]) * self.freshman[i] for i in range(self.num_ulas)), "eps_1_constraints")

    # eps_C_constraint, InstructorPreference, 
    def constraint_preference(self):
        for i in range(self.num_ulas):
            for s in range(self.num_studios):
                # Now that we've checked for None values, we can safely add the constraint
                self.m.addConstr(self.eps_C[i] >= self.x[i, s] *
                            abs(self.studio_info.loc[s, 'Course Index'] - self.student_info.loc[i, 'Course Preference']),
                            name=f"eps_C_constraint_{i}_{s}")
                
        # This constraint ensures that if instructor $j$ is ALWAYS assigned 
        # ULA $i$ if they have specified that it is a preference.
        self.m.addConstrs((self.alpha[i, j] + self.eps_p[i, j] >= self.pref.iloc[j, i]
                      for i in range(self.num_ulas)
                      for j in range(self.num_instructors)), "InstructorPreference")

    # p_maj_constraints, p_school_constraints
    def constraint_linearization(self):
        # The next 2 constraints (alongside the minimization of 
        # $P_{m_1m_2}^{maj}$) ensure that 
        # $P_{m_1m_2}^{maj} = |N_{m_1}^{maj}-N_{m_2}^{maj}|$
        self.m.addConstrs((self.P_major[a, b] >= self.N_maj[a] - self.N_maj[b] for a in range(self.num_majors)
                      for b in range(self.num_majors)), "p_maj_constraints1")

        self.m.addConstrs((self.P_major[a, b] >= self.N_maj[b] - self.N_maj[a] for a in range(self.num_majors)
                      for b in range(self.num_majors)), "p_maj_constraints2")

        # The next 2 constraints (alongside the minimization of 
        # $P_{k_1k_2}^{school}$) ensure that 
        # $P_{k_1k_2}^{school} = |N_{k_1}^{school}-N_{k_2}^{school}|$
        self.m.addConstrs((self.P_school[a, b] >= self.N_school[a] - self.N_school[b] for a in range(self.num_schools)
                      for b in range(self.num_schools)), "p_school_constraints1")

        self.m.addConstrs((self.P_school[a, b] >= self.N_school[b] - self.N_school[a] for a in range(self.num_schools)
                      for b in range(self.num_schools)), "p_school_constraints2")
        
    def set_rerun_constraints(self, rerun_filename : str):
        rerun_info = pd.read_csv(rerun_filename) 

        # rerun file needs 3 columns: index, studio 1, studio 2, not available 
        # Add constraints: If a student is not available, skip assignment to their studios
        for student, row in rerun_info.iterrows():
            studio1 = row['Studio 1']
            studio2 = row['Studio 2']
            not_available = row['not available']

            if not_available == 0:  # Only enforce assignment if 'not available' is 0
                self.m.addConstr(self.x[student, studio1] == 1, f"Assign_Studio1_{student}")
                self.m.addConstr(self.x[student, studio2] == 1, f"Assign_Studio2_{student}")
            else:   # this student is not available; make sure they arent hired
                # Prevent the student from being assigned to any studio
                for studio in range(self.num_studios):
                    self.m.addConstr(self.x[student, studio] == 0, f"Prevent_All_Studios_{student}")

    def infeasible_model(self):
        # create output file with flags to help format input
        df_ula = self.student_info.copy()
        df_studios = self.studio_info.copy()

        ula_index = {self.student_info.loc[i, 'Email']: i for i in range(self.num_ulas)}

        df_studios['TwoULA_Studios'] = 0
        df_studios['OneULA_Only_Studios'] = 0
        df_studios['Conflicting Availability'] = 0
        df_studios['ULA_Availability'] = 0

        df_ula['Only2_Studios'] = 0
        df_ula['Returning Not Hired'] = 0
        df_ula['Conflicting Availability'] = 0
        df_ula['ULA_Availability'] = 0
        df_ula['Only1_InstructorPerULA_1'] = 0
        df_ula['Only1_InstructorPerULA_2'] = 0
        df_ula['Only1_InstructorPerULA_3'] = 0

        for c in self.m.getConstrs():
            if c.IISConstr:
                print(f"{c.ConstrName}: {c}")
                if "Only2_Studios" in c.constrname:
                    df_ula.loc[utils.extract_indices(c.constrname)[0], 'Not Exactly 2 Studios'] = 1

                elif "OneULA_Only_Studios" in c.constrname:
                    df_studios.loc[utils.extract_indices(c.constrname)[0], 'Not Exactly 1 ULA'] = 1

                elif "TwoULA_Studios" in c.constrname:
                    df_studios.loc[utils.extract_indices(c.constrname)[0], 'Less Than 2 ULAs'] = 1

                elif "ULA_Availability" in c.constrname:
                    temp = utils.extract_indices(c.constrname)
                    df_ula.loc[temp[0], 'ULA_Availability'] = 1
                    df_studios.loc[temp[1], 'ULA_Availability'] = 1

                elif "Only1_InstructorPerULA_1" in c.constrname:
                    df_ula.loc[utils.extract_indices(c.constrname)[0], 'Only1_InstructorPerULA_1'] = 1

                elif "Only1_InstructorPerULA_2" in c.constrname:
                    df_ula.loc[utils.extract_indices(c.constrname)[0], 'Only1_InstructorPerULA_2'] = 1

                elif "Only1_InstructorPerULA_3" in c.constrname:
                    df_ula.loc[utils.extract_indices(c.constrname)[0], 'Only1_InstructorPerULA_3'] = 1

        for v in self.m.getVars():
            if v.IISLB: print(f'\t{v.varname} ≥ {v.LB}')
            if v.IISUB: print(f'\t{v.varname} ≤ {v.UB}')

        # Relax the bounds and try to make the model feasible
        print('The model is infeasible; relaxing the bounds')
        orignumvars = self.m.NumVars
        # relaxing only variable bounds
        self.m.feasRelaxS(0, False, True, False)
        # for relaxing variable bounds and constraint bounds use
        # m.feasRelaxS(0, False, True, True)

        self.m.optimize()

        status = self.m.Status
        if status in (GRB.INF_OR_UNBD, GRB.INFEASIBLE, GRB.UNBOUNDED):
            print('The relaxed model cannot be solved \
                   because it is infeasible or unbounded')

            df_ula.to_csv("ULA_error")
            df_studios.to_csv("Studio_error")

            sys.exit(1)

        # print the values of the artificial variables of the relaxation
        print('\nSlack values:')
        slacks = self.m.getVars()[orignumvars:]
        for sv in slacks:
            if sv.X > 1e-9:
                print('%s = %g' % (sv.VarName, sv.X))


    def display_results(self):
        print('\nRun Time Report:')
        print('  Total Run Time')
        print('    Seconds: ' + str((self.end_preprocess - self.start_preprocess) +
                                    (self.end_var_obj - self.start_var_obj) +
                                    (self.end_constraints - self.start_constraints) +
                                    (self.end_solve - self.start_solve)))
        print('    Minutes: ' + str(((self.end_preprocess - self.start_preprocess) +
                                     (self.end_var_obj - self.start_var_obj) +
                                     (self.end_constraints - self.start_constraints) +
                                     (self.end_solve - self.start_solve)) / 60))
        print('  Pre-processing')
        print('    Seconds: ' + str((self.end_preprocess - self.start_preprocess)))
        print('    Minutes: ' + str((self.end_preprocess - self.start_preprocess) / 60))
        print('  Adding Variables andself. Objecitve')
        print('    Seconds: ' + str((self.end_var_obj - self.start_var_obj)))
        print('    Minutes: ' + str((self.end_var_obj - self.start_var_obj) / 60))
        print('  Adding Constraints')
        print('    Seconds: ' + str((self.end_constraints - self.start_constraints)))
        print('    Minutes: ' + str((self.end_constraints - self.start_constraints) / 60))
        print('  Solving MILP')
        print('    Seconds: ' + str((self.end_solve - self.start_solve)))
        print('    Minutes: ' + str((self.end_solve - self.start_solve) / 60))


    def write_solution(self):
        temp = [self.m.getVarByName(name) for name in
                (f"x[{i},{s}]" for i in range(self.num_ulas) for s in range(self.num_studios))]  # Get name of variables
        x_sol = self.m.getAttr("X", temp)  # Retrieve variables with the names above

        temp = [self.m.getVarByName(name) for name in (f"x[{i},{s}]" for i in range(self.num_ulas) for s in range(1, 2))]
        test = self.m.getAttr("VarName", temp)

        # Get h variable solutions
        temp = [self.m.getVarByName(name) for name in (f"h[{i}]" for i in range(self.num_ulas))]  # Get name of variables
        h_sol = self.m.getAttr("X", temp)  # Retrieve variables with the names above

        df_ula = self.student_info.copy()
        df_studios = self.studio_info.copy()     

        # if this is a rerun these columns will already exist, don't overwrite them
        df_ula['Hire?'] = 0
        df_ula['Studio 1'] = 0
        df_ula['Studio 2'] = 0
        df_ula['IP_Req'] = 0
        df_studios['IP_Req'] = 0
        df_ula['Lang_Req'] = 0
        df_studios['Lang_Req'] = 0
        df_ula['Return_Req'] = 0
        df_ula['SP_Req'] = 0
        df_ula['Non-Ecampus'] = 0
        df_studios['Non-Ecampus'] = 0
        df_ula['Non-Honors'] = 0
        df_studios['Non-Honors'] = 0
        df_studios['ReturnULA'] = 0

        self.write_ula_solutions(x_sol, h_sol, df_ula, df_studios)


        df_studios['ULA 1'] = -1
        df_studios['ULA 2'] = -1

        self.write_studio_solutions(df_ula, df_studios)

        # flag if instructor preference was not met
        self.write_instructor_solutions(df_ula, df_studios)

        # rerun stuff
        hired_students = self.write_rerun_df(df_ula)
        print(hired_students)

        hired_students.to_csv('rerun_output.csv')
        df_ula.to_csv(self.output_ULA_file_name)
        df_studios.to_csv(self.output_studio_file_name)

    def write_rerun_df(self, df_ula):
        # Initialize DataFrame with hired students
        hired_students = pd.DataFrame()
        
        # Get mask for hired students
        hired_mask = df_ula['Hire?'] == 1
        
        # Initialize all columns with numeric default values
        hired_students['Studio 1'] = -1
        hired_students['Studio 2'] = -1
        hired_students['Languages'] = -1
        hired_students['Gender'] = -1
        hired_students['Race'] = -1
        hired_students['School'] = -1
        hired_students['Major'] = -1
        
        # Assign numeric values for hired students
        hired_students['Studio 1'] = df_ula[hired_mask]['Studio 1']
        hired_students['Studio 2'] = df_ula[hired_mask]['Studio 2']
        hired_students['Languages'] = df_ula[hired_mask]['Languages']  # Keep as string for comma-separated values
        hired_students['Gender'] = df_ula[hired_mask]['Gender'].astype(int)
        hired_students['Race'] = df_ula[hired_mask]['Race'].astype(int)
        hired_students['School'] = df_ula[hired_mask]['School'].astype(int)
        hired_students['Major'] = df_ula[hired_mask]['Major'].astype(int)
        
        return hired_students

    def write_instructor_solutions(self, df_ula, df_studios):
        for i in range(self.num_instructors):
            instructor_email = self.unique_instructors[i]
            remaining_students = self.instructor_preferences[instructor_email].copy()  # Create a copy of preferences
            
            # Loop through each studio associated with this instructor
            for studio_idx in self.instructor_sections.get(i, []):
                # Get ULAs assigned to this studio
                ula_1 = df_studios.loc[studio_idx, 'ULA 1']
                ula_2 = df_studios.loc[studio_idx, 'ULA 2']

                # Create a dictionary to map indices back to student emails
                email_to_index = {email: idx for idx, email in enumerate(self.student_info['Email'])}
                index_to_email = {idx: email for email, idx in email_to_index.items()}

                # Remove assigned ULAs from remaining students if they were in preferences
                if ula_1 != -1 and ula_1 in remaining_students:
                    remaining_students.remove(ula_1)
                if ula_2 != -1 and ula_2 in remaining_students:
                    remaining_students.remove(ula_2)

            # After checking all studios, if there are still remaining preferred students
            if remaining_students:
                # Map remaining students' indices back to emails
                remaining_students_emails = [index_to_email[student] for student in remaining_students]
                
                # Set IP_Req flag for all studios of this instructor
                for studio_idx in self.instructor_sections.get(i, []):
                    df_studios.loc[studio_idx, 'IP_Req'] = ', '.join(remaining_students_emails)
                    
                    # Set IP_Req flag for assigned ULAs
                    ula_1 = df_studios.loc[studio_idx, 'ULA 1']
                    ula_2 = df_studios.loc[studio_idx, 'ULA 2']
                    
                    if ula_1 != -1:
                        df_ula.loc[ula_1, 'IP_Req'] = 1
                    if ula_2 != -1:
                        df_ula.loc[ula_2, 'IP_Req'] = 1

    def write_studio_solutions(self, df_ula, df_studios):
        for s in range(self.num_studios):
            # Get x variables for studio "s"
            temp = [self.m.getVarByName(name) for name in (f"x[{i},{j}]" for i in range(self.num_ulas) for j in range(s, s + 1))]
            temp1 = self.m.getAttr("X", temp)
            temp2 = [x for x in range(len(temp1)) if temp1[x] > 0.1]

            # match assigned ULA to studio
            if df_studios.loc[s, '1 ULA?'] == 1:
                df_studios.loc[s, 'ULA 1'] = temp2[0]

                if df_ula.loc[temp2[0], 'Return ULA'] == 1:
                    df_studios.loc[s, 'ReturnULA'] = 1
            else:
                df_studios.loc[s, 'ULA 1'] = temp2[0]
                df_studios.loc[s, 'ULA 2'] = temp2[1]

            # check if ULA language meets req if there are any
            if df_studios.loc[s, 'Language'] != -1:
                req_lang = df_studios.loc[s, 'Language']
                ula_index = df_studios.loc[s, 'ULA 1']

                # flag if ula languages do not meet studio req
                ula_lang = list(map(int, str(df_ula.loc[ula_index, 'Languages']).split(',')))
                if req_lang not in ula_lang:
                    df_ula.loc[ula_index, 'Lang_Req'] = 1

                # Repeat for second ula if any
                if df_studios.loc[s, '1 ULA?'] != 1:
                    ula_index = df_studios.loc[s, 'ULA 2']

                    # flag if ula languages do not meet studio req
                    ula_lang = list(map(int, str(df_ula.loc[ula_index, 'Languages']).split(',')))
                    if req_lang not in ula_lang:
                        df_ula.loc[ula_index, 'Lang_Req'] = 1
                        df_studios.loc[s, 'Lang_Req'] = 1

                # Flag if ULA course preference is not met
                ula_index = df_studios.loc[s, 'ULA 1']
                assigned_course = df_studios.loc[s, 'Course Index']
                preferred_course = df_ula.loc[ula_index, 'Course Preference']
                is_course_mismatch = assigned_course != preferred_course
                if is_course_mismatch:
                    df_ula.loc[ula_index, 'SP_Req'] = 1

                if df_studios.loc[s, 'ULA 2'] != -1:
                    if df_studios.loc[s, 'Course Index'] != df_ula.loc[df_studios.loc[s, 'ULA 2'], 'Course Preference']:
                        df_ula.loc[df_studios.loc[s, 'ULA 2'], 'SP_Req'] = 1

                # Flag is ecampus is assigned to non-ecampus course
                if df_studios.loc[s, 'Ecampus'] != 1:
                    ula1 = df_studios.loc[s, 'ULA 1']
                    ula2 = df_studios.loc[s, 'ULA 2']

                    if df_ula.loc[ula1, 'Ecampus'] == 1:
                        df_ula.loc[ula1, 'Non-Ecampus'] = 1

                    if df_studios.loc[s, 'ULA 2'] != -1:
                        if df_ula.loc[ula2, 'Ecampus'] == 1:
                            df_ula.loc[ula2, 'Non-Ecampus'] = 1
                            df_studios.loc[s, 'Non-Ecampus'] = 1

                # Flag is Non-honors is assigned to honors studio
                if df_studios.loc[s, 'HC'] == 1:
                    ula1 = df_studios.loc[s, 'ULA 1']
                    ula2 = df_studios.loc[s, 'ULA 2']

                    if df_ula.loc[ula1, 'HC'] != 1:
                        df_ula.loc[ula1, 'Non-Honors'] = 1
                        df_studios.loc[s, 'Non-Honors'] = 1

                    if df_studios.loc[s, 'ULA 2'] != -1:
                        if df_ula.loc[ula2, 'HC'] != 1:
                            df_ula.loc[ula2, 'Non-Honors'] = 1
                            df_studios.loc[s, 'Non-Honors'] = 1

    def write_ula_solutions(self, x_sol, h_sol, df_ula, df_studios):
        for i in range(self.num_ulas):
            if h_sol[i] > 0.1:
                df_ula.loc[i, 'Hire?'] = h_sol[i]

                # temp = x_sol[(i*100):(i*100+num_studios)]
                temp = x_sol[(i * self.num_studios):(i * self.num_studios + self.num_studios)]
                temp1 = [x for x in range(len(temp)) if temp[x] > 0.1]
                df_ula.loc[i, 'Studio 1'] = temp1[0] + 1
                df_ula.loc[i, 'Studio 2'] = temp1[1] + 1

                # Mark positions as filled
                df_studios.loc[temp1[0] + 1, 'Position Filled'] = 1
                df_studios.loc[temp1[1] + 1, 'Position Filled'] = 1
            else:
                df_ula.loc[i, 'Hire?'] = -1
                df_ula.loc[i, 'Studio 1'] = -1
                df_ula.loc[i, 'Studio 2'] = -1

                # mark if ULA was returning ULA and wasn't hired
                if df_ula.loc[i, 'Return ULA'] == 1:
                    df_ula.loc[i, 'Return_Req'] = 1


    def build_model(self):
        """Build the optimization model by loading data, defining variables, constraints, and objective."""
        
        self.start_var_obj = time.time()
        self.define_variables()
        self.set_objective()
        self.end_var_obj = time.time()

        self.start_constraints = time.time()
        self.add_constraints()
        self.end_constraints = time.time()


    def solve(self):
        """Solve the optimization model and display results."""
        self.start_solve = time.time()
        self.m.optimize()
        self.end_solve = time.time()
        self.display_results()

        # infeasibility handling
        if self.m.Status == GRB.INFEASIBLE:
            # Find what constraints are unsatisfied
            self.m.write('iismodel.ilp')

            """ query the IIS attributes for each constraint and variable
                using IISConstr, IISLB, IISUB, IISSOS, IISQConstr, and IISGenConstr. 
                Each indicates whether the corresponding model element is a member of the computed IIS.
                Here is an example of printing out the variable limits and linear constraints in the IIS 
                
                source: https://support.gurobi.com/hc/en-us/articles/15656630439441-How-do-I-use-compute-IIS-to-find-a-subset-of-constraints-that-are-causing-model-infeasibility"""
            
            self.infeasible_model()

        self.write_solution()


    def export_model(self):
        self.m.write("optimized_model/model_output.lp")   # LP format
        self.m.write("optimized_model/model_output.mps")  # MPS format   


if __name__== "__main__":
    inputFileName = 'Sample_Input_Data 1.xlsx'
    # inputFileLocation = 'C:/Users/agorj/Box/ULA Assignment Project'
    # inputFileLocation = 'C:/Users/Joe/Box/ULA Assignment Project/'

    # Specify Name and Location of Output File

    output_ULA_FileName = 'ULA_output 1.csv'
    output_studio_FileName = 'studio_output 1.csv'
    error_ULA_FileName = 'ULA_error 1.csv'
    error_studio_FileName = 'studio_error.csv'

    ula_model = ULAModel(inputFileName, output_ULA_FileName, output_studio_FileName, "ula_model")

    ula_model.build_model()
    ula_model.solve()

        
