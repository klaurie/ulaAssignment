# ULA Placement Optimization Program

Overview

This optimization program is designed to allocate Undergraduate Learning Assistants (ULAs) to studios for a given set of constraints and preferences, considering several factors such as ULA availability, instructor preferences, 
studio requirements. The program uses Gurobi optimization to determine the best assignments based on multiple objectives while ensuring fairness and adherence to constraints.

Features

Studio Assignment: Assign ULAs to studios ensuring they meet the studio's required number of ULAs and matching language preferences.

Instructor Preferences: Ensure ULAs are assigned to instructors they prefer, based on ULA preferences.

Student Availability: Ensure ULAs are assigned to studios where they are available.

Demographic Constraints: Place the hired ULAs in studio sections that provide distributed representation across schools/majors.

E-campus Preferences: Manage the assignment of e-campus and on-campus students to corresponding studio types.

Variable Adjustment: The model allows for variable redefinition to accommodate re-runs, such as adjusting for rejections and reassignments.
