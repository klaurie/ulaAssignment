# ULA Placement Optimization Program

Overview

This optimization program is designed to allocate Undergraduate Learning Assistants (ULAs) to studios for a given set of constraints and preferences, considering several factors such as ULA availability, instructor preferences, 
studio requirements, and demographic considerations to emphasize fairness placement. The program uses Gurobi optimization to determine the best assignments based on multiple objectives while ensuring fairness and adherence to constraints.

Features

Studio Assignment: Assign ULAs to studios ensuring they meet the studio's required number of ULAs and matching language preferences.

Instructor Preferences: Ensure ULAs are assigned to instructors they prefer, based on ULA preferences.

Student Availability: Ensure ULAs are assigned to studios where they are available.

Demographic Constraints: Enforce fairness in the assignment process by considering ULA demographics (e.g., gender, race, major, school, and honors college).

E-campus Preferences: Manage the assignment of e-campus and on-campus students to corresponding studio types.

Variable Adjustment: The model allows for variable redefinition to accommodate re-runs, such as adjusting for rejections and reassignments.

Requirements

Python 3.x

Gurobi Optimizer (for solving the optimization problem)

pandas (for data manipulation)

numpy (for mathematical operations)

Setup

Install Gurobi (if you don't have it already):

Download and install from Gurobi's website.
You’ll need a Gurobi license (a free academic license is available).
Install required Python libraries:


`pip install pandas numpy gurobipy`
Ensure you have all necessary data (student, studio, and instructor information) in a compatible format (e.g., CSV or Excel files).


